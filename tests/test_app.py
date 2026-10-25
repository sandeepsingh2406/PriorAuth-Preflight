"""Tests for the Gradio screen's logic in src/app.py (fake models, temp audit log, no browser)."""
import datetime
import json
import re
from pathlib import Path

import pytest

from src.app import (
    BANNER,
    CaseForm,
    build_app,
    case_id_for,
    decide,
    lab_recency,
    list_cases,
    load_case_into_form,
    main,
    render_requirements,
    run_case,
)
from src.models import ChecklistItem, DraftText, ExtractedField, ExtractionResult, Order, PayerChecklist
from src.run import load_case
from src.tools import check_packet, lookup_payer_checklist

APPROVE_LABEL = "Approve draft (logs decision only)"

PREFIXES = {
    "diagnosis": ("Diagnosis:",),
    "stage": ("Stage:", "Staging:"),
    "biomarker": ("Biomarkers",),
    "prior_therapies": ("Prior therapy:",),
    "recent_lab": ("Labs:",),
    "treatment_plan": ("Plan:",),
}


def _extraction(note: str, conflicts: dict[str, list[str]] | None = None):
    """A fake model that quotes whole lines for the items it can find in a hand-written note."""
    fields = []
    for name, prefixes in PREFIXES.items():
        line = next((l for l in note.splitlines() if l.startswith(prefixes)), None)
        date = None
        if line and name == "recent_lab":
            date = datetime.date.fromisoformat(re.search(r"\d{4}-\d{2}-\d{2}", line).group())
        fields.append(
            ExtractedField(
                name=name, value=line, quote=line, found=line is not None, date=date,
                conflicting_quotes=(conflicts or {}).get(name, []),
            )
        )
    return lambda prompt: ExtractionResult(fields=fields)


def _outreach(prompt: str) -> DraftText:
    return DraftText(subject="Items needed for the packet", body="Please send the listed items.")


def _form(case_id: str, **changes: str) -> CaseForm:
    form = load_case_into_form(case_id)
    return form.model_copy(update=changes) if changes else form


def _run(case_id: str, tmp_path: Path, conflicts=None, **changes: str):
    form = _form(case_id, **changes)
    return run_case(
        form,
        llm=_extraction(form.note, conflicts),
        outreach_llm=_outreach,
        audit_path=tmp_path / "audit.jsonl",
    )


def _audit(tmp_path: Path) -> list[dict]:
    path = tmp_path / "audit.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


# --- cases and the form ---------------------------------------------------------


def test_cases_are_listed_hand_labeled_first() -> None:
    cases = list_cases()
    assert cases[:10] == [f"hand_{i:02d}" for i in range(1, 11)]
    assert len(cases) == 40 and cases[10] == "gen_001"


def test_choosing_a_case_fills_the_whole_form() -> None:
    form = load_case_into_form("hand_07")
    order, note = load_case("hand_07")
    assert form.case_id == "hand_07" and form.note == note
    assert form.payer_id == order.payer_id
    assert form.order_date == order.order_date.isoformat()
    assert (form.drug, form.dose, form.schedule) == (order.drug, order.dose, order.schedule)


def test_an_unknown_case_id_is_an_error_not_a_crash() -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        load_case_into_form("../hand_01")


def test_audit_case_id_follows_the_note_actually_analysed() -> None:
    """A chosen case keeps its id only while the note is unchanged; edited or pasted notes say so."""
    _, note = load_case("hand_01")
    assert case_id_for("hand_01", note) == "hand_01"
    assert case_id_for("hand_01", note + " edited") == "pasted_note"
    assert case_id_for("", note) == "pasted_note"
    assert case_id_for(None, note) == "pasted_note"


# --- running a case ---------------------------------------------------------------


def test_ready_case_shows_status_evidence_for_every_item_and_a_review_button(tmp_path: Path) -> None:
    view = _run("hand_01", tmp_path)
    assert view.error is None
    assert "READY" in view.status_html and "NOT_READY" not in view.status_html
    for fragment in ["Stage IIIB", "EGFR exon 19 deletion", "Start Examplumab 200 mg IV every 21 days"]:
        assert fragment in view.evidence_html
    assert view.draft is None and view.show_draft is False
    assert view.show_reviewed is True
    assert len(_audit(tmp_path)) == 1


def test_not_ready_case_lists_the_gaps_and_shows_an_editable_draft(tmp_path: Path) -> None:
    view = _run("hand_07", tmp_path)
    assert "NOT_READY" in view.status_html
    assert "biomarker" in view.evidence_html and "prior_therapies" in view.evidence_html
    assert "no evidence found in the note" not in view.evidence_html  # the gap text already says it
    assert view.show_draft is True and view.show_reviewed is False
    assert (view.subject, view.body) == ("Items needed for the packet", "Please send the listed items.")
    assert "not sent" in view.draft_label.lower()


def test_needs_review_case_shows_both_conflicting_quotes_and_the_reason_without_a_draft(tmp_path: Path) -> None:
    view = _run("hand_08", tmp_path, conflicts={"stage": ["stage IV non-small cell lung cancer"]})
    assert "NEEDS_REVIEW" in view.status_html
    assert "Staging: Stage II (cT2 N0 M0)" in view.evidence_html
    assert "stage IV non-small cell lung cancer" in view.evidence_html
    assert "Escalation reason" not in view.evidence_html  # the conflicting gap already explains it
    assert view.draft is None and view.show_draft is False and view.show_reviewed is True


def test_the_trace_lists_the_tools_in_order_with_the_run_id(tmp_path: Path) -> None:
    view = _run("hand_07", tmp_path)
    order = ["lookup_payer_checklist", "extract_fields", "validate_citations", "check_packet", "draft_request"]
    positions = [view.trace.index(name) for name in order]
    assert positions == sorted(positions)
    assert view.session.run.run_id in view.trace


def test_no_gaps_text_for_a_ready_case(tmp_path: Path) -> None:
    assert "No gaps" in _run("hand_01", tmp_path).evidence_html


# --- input checks: friendly errors, no model call, no audit line -------------------


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"order_date": "05/04/2026"}, "date"),
        ({"order_date": ""}, "date"),
        ({"payer_id": "payer_z"}, "payer"),
        ({"note": "   "}, "note"),
        ({"note": "x" * 20001}, "long"),
        ({"drug": " "}, "drug"),
        ({"dose": ""}, "dose"),
        ({"schedule": ""}, "schedule"),
    ],
)
def test_bad_input_gives_a_message_and_runs_nothing(tmp_path: Path, changes: dict, expected: str) -> None:
    def must_not_run(prompt: str) -> ExtractionResult:
        raise AssertionError("the model was called")

    form = _form("hand_01", **changes)
    view = run_case(form, llm=must_not_run, outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    assert view.error and expected in view.error.lower()
    assert view.session is None
    assert _audit(tmp_path) == []


# --- escaping: note text is data, never markup --------------------------------------


def test_note_text_is_escaped_before_it_reaches_the_page(tmp_path: Path) -> None:
    """A quote containing HTML must show as text, not run as markup."""
    evil = "Labs: <script>alert(1)</script><img src=x onerror=alert(2)> 2026-04-22."
    _, note = load_case("hand_01")
    form = _form("hand_01", note=note + evil + "\n")

    def llm(prompt: str) -> ExtractionResult:
        return ExtractionResult(
            fields=[ExtractedField(name="recent_lab", value=evil, quote=evil, found=True, date=datetime.date(2026, 4, 22))]
        )

    view = run_case(form, llm=llm, outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    page = view.status_html + view.evidence_html + view.trace
    assert "<script>" not in page and "<img" not in page
    assert "&lt;script&gt;" in view.evidence_html


def test_unreadable_text_is_shown_to_the_reviewer_and_escaped(tmp_path: Path) -> None:
    bad = "<i>EG?R ex@n</i> [unreadable]"
    _, note = load_case("hand_01")
    form = _form("hand_01", note=note + bad + "\n")
    fields = _extraction(note)(None).fields
    view = run_case(
        form,
        llm=lambda p: ExtractionResult(fields=fields, unreadable_passages=[bad]),
        outreach_llm=_outreach,
        audit_path=tmp_path / "audit.jsonl",
    )
    assert "NEEDS_REVIEW" in view.status_html
    assert "&lt;i&gt;EG?R ex@n&lt;/i&gt; [unreadable]" in view.evidence_html
    assert "<i>" not in view.evidence_html


# --- decisions ----------------------------------------------------------------------


def test_approving_the_draft_as_it_is_records_approved(tmp_path: Path) -> None:
    view = _run("hand_07", tmp_path)
    message = decide(view.session, "approve", view.subject, view.body, audit_path=tmp_path / "audit.jsonl")
    assert [l["human_decision"] for l in _audit(tmp_path)] == [None, "approved"]
    assert "approved" in message and "nothing was sent" in message.lower()


def test_approving_after_changing_the_text_records_edited_then_approved(tmp_path: Path) -> None:
    view = _run("hand_07", tmp_path)
    decide(view.session, "approve", view.subject, view.body + " Thanks!", audit_path=tmp_path / "audit.jsonl")
    assert [l["human_decision"] for l in _audit(tmp_path)] == [None, "edited", "approved"]


def test_edit_records_edited_only_when_something_changed(tmp_path: Path) -> None:
    view = _run("hand_07", tmp_path)
    log = tmp_path / "audit.jsonl"
    message = decide(view.session, "edit", view.subject, view.body, audit_path=log)
    assert "no changes" in message.lower()
    assert len(_audit(tmp_path)) == 1
    decide(view.session, "edit", view.subject, "A different body.", audit_path=log)
    assert _audit(tmp_path)[-1]["human_decision"] == "edited"


def test_discard_records_discarded(tmp_path: Path) -> None:
    view = _run("hand_07", tmp_path)
    message = decide(view.session, "discard", view.subject, view.body, audit_path=tmp_path / "audit.jsonl")
    assert _audit(tmp_path)[-1]["human_decision"] == "discarded"
    assert "nothing was sent" in message.lower()


@pytest.mark.parametrize("case_id", ["hand_01", "hand_08"])
def test_mark_reviewed_for_cases_without_a_draft(tmp_path: Path, case_id: str) -> None:
    view = _run(case_id, tmp_path, conflicts={"stage": ["x"]} if case_id == "hand_08" else None)
    message = decide(view.session, "reviewed", "", "", audit_path=tmp_path / "audit.jsonl")
    assert _audit(tmp_path)[-1]["human_decision"] == "reviewed"
    assert "reviewed" in message


def test_a_decision_that_does_not_fit_the_run_is_refused_with_a_message(tmp_path: Path) -> None:
    ready = _run("hand_01", tmp_path)
    message = decide(ready.session, "approve", "", "", audit_path=tmp_path / "audit.jsonl")
    assert "not" in message.lower() and len(_audit(tmp_path)) == 1  # no decision line written
    draft = _run("hand_07", tmp_path)
    message = decide(draft.session, "reviewed", "", "", audit_path=tmp_path / "audit.jsonl")
    assert len(_audit(tmp_path)) == 2  # only the second run's line, still no decision


def test_deciding_before_running_anything_is_a_message_not_an_error(tmp_path: Path) -> None:
    message = decide(None, "approve", "", "", audit_path=tmp_path / "audit.jsonl")
    assert "run a case" in message.lower()
    assert _audit(tmp_path) == []


def test_decision_lines_never_contain_draft_or_edited_text(tmp_path: Path) -> None:
    view = _run("hand_07", tmp_path)
    decide(view.session, "approve", "Zebra subject", "Zebra-striped edited body text", audit_path=tmp_path / "audit.jsonl")
    text = (tmp_path / "audit.jsonl").read_text()
    assert "Zebra" not in text and "Please send" not in text


def test_each_run_has_its_own_session_and_decisions_link_to_it(tmp_path: Path) -> None:
    a, b = _run("hand_07", tmp_path), _run("hand_07", tmp_path)
    assert a.session is not b.session
    decide(a.session, "discard", a.subject, a.body, audit_path=tmp_path / "audit.jsonl")
    lines = _audit(tmp_path)
    decision = next(l for l in lines if l["human_decision"])
    assert decision["run_id"] == a.session.run.run_id != b.session.run.run_id


# --- the app itself -------------------------------------------------------------------


def test_app_builds_with_the_agreed_controls_and_no_telemetry() -> None:
    demo = build_app(llm=lambda p: ExtractionResult(fields=[]), outreach_llm=_outreach)
    labels = {getattr(b, "value", None) for b in demo.blocks.values()}
    for label in ["Run", APPROVE_LABEL, "Edit", "Discard", "Mark reviewed"]:
        assert label in labels, label
    assert demo.analytics_enabled is False


def test_the_app_launches_locally_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import gradio as gr

    captured: dict = {}
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: captured.update(kwargs))
    code = main(env_file=None, llm=lambda p: ExtractionResult(fields=[]), outreach_llm=_outreach)
    assert code == 0
    assert captured["share"] is False
    assert captured["server_name"] == "127.0.0.1"


def test_main_without_a_model_name_explains_and_exits_2(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MODEL_DEV", raising=False)
    monkeypatch.delenv("USE_DEMO_MODEL", raising=False)
    assert main(env_file=None) == 2
    assert "MODEL_DEV" in capsys.readouterr().err


def test_only_run_is_a_primary_button_and_approve_is_neutral() -> None:
    """Approve only logs a decision (nothing is sent), so it must not look like the main action."""
    import gradio as gr

    demo = build_app(llm=lambda p: ExtractionResult(fields=[]), outreach_llm=_outreach)
    variants = {b.value: b.variant for b in demo.blocks.values() if isinstance(b, gr.Button)}
    assert variants["Run"] == "primary"
    assert variants[APPROVE_LABEL] == "secondary"
    assert [v for v, variant in variants.items() if variant == "primary"] == ["Run"]


def test_the_title_block_explains_how_it_works_in_three_steps_and_states_the_limits() -> None:
    text = BANNER
    assert text.startswith("# PriorAuth Preflight")
    assert "How it works" in text
    steps = [line for line in text.splitlines() if line[:3] in ("1. ", "2. ", "3. ")]
    assert len(steps) == 3
    assert "quotes the evidence" in steps[0]
    assert "word for word" in steps[1] and "decides the status" in steps[1]
    assert "gap list only" in steps[2] and "person reviews everything" in steps[2]
    assert "Limits" in text
    for fragment in ["Synthetic notes", "invented payer rules", "not whether the treatment is right", "never sends", "check the quotes"]:
        assert fragment in text, fragment


def test_the_title_block_is_on_the_page() -> None:
    demo = build_app(llm=lambda p: ExtractionResult(fields=[]), outreach_llm=_outreach)
    assert BANNER in {getattr(b, "value", None) for b in demo.blocks.values()}


# --- spacing and redundancy in the gap list -----------------------------------------------


def test_a_missing_item_says_it_once(tmp_path: Path) -> None:
    """'Not documented in the note: ...' is the whole message; there is no second 'no evidence' line."""
    html = _run("hand_07", tmp_path).evidence_html
    assert html.count("Not documented in the note: Relevant biomarker or test result.") == 1
    assert html.count("Not documented in the note: Prior therapies tried, with outcomes.") == 1
    assert "no evidence found" not in html


def test_gap_text_and_its_evidence_sit_on_separate_lines(tmp_path: Path) -> None:
    """The detail ends its own block before the quotes start, so the two never run together."""
    html = _run("hand_08", tmp_path, conflicts={"stage": ["stage IV non-small cell lung cancer"]}).evidence_html
    assert re.search(r"about this item\.</div>\s*<div[^>]*>\s*<blockquote", html)
    assert "item.<blockquote" not in html


def test_the_reason_is_hidden_when_conflicting_gaps_already_explain_it(tmp_path: Path) -> None:
    view = _run("hand_08", tmp_path, conflicts={"stage": ["x"]})
    assert "NEEDS_REVIEW" in view.status_html
    assert "Escalation reason" not in view.evidence_html
    assert "[conflicting]" in view.evidence_html


def test_the_reason_is_shown_for_unreadable_text(tmp_path: Path) -> None:
    bad = "EG?R ex@n [unreadable]"
    _, note = load_case("hand_01")
    fields = _extraction(note)(None).fields
    view = run_case(
        _form("hand_01", note=note + bad + "\n"),
        llm=lambda p: ExtractionResult(fields=fields, unreadable_passages=[bad]),
        outreach_llm=_outreach,
        audit_path=tmp_path / "audit.jsonl",
    )
    assert "Escalation reason" in view.evidence_html and "Unreadable text in the note" in view.evidence_html


def test_the_reason_is_shown_when_the_model_failed(tmp_path: Path) -> None:
    def broken(prompt: str) -> ExtractionResult:
        raise RuntimeError("boom")

    view = run_case(_form("hand_01"), llm=broken, outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    assert "NEEDS_REVIEW" in view.status_html
    assert "Escalation reason" in view.evidence_html and "Extraction failed (RuntimeError)" in view.evidence_html


def test_the_reason_is_shown_when_unreadable_text_comes_with_a_conflict(tmp_path: Path) -> None:
    bad = "EG?R ex@n [unreadable]"
    _, note = load_case("hand_08")
    base = _extraction(note, {"stage": ["stage IV non-small cell lung cancer"]})(None).fields
    view = run_case(
        _form("hand_08", note=note + bad + "\n"),
        llm=lambda p: ExtractionResult(fields=base, unreadable_passages=[bad]),
        outreach_llm=_outreach,
        audit_path=tmp_path / "audit.jsonl",
    )
    assert "Escalation reason" in view.evidence_html


# --- the lab recency line --------------------------------------------------------------------

D = datetime.date


def test_lab_line_for_a_stale_lab() -> None:
    result = lab_recency(D(2026, 4, 10), D(2026, 5, 11), 30)
    assert (result.age, result.verdict) == (31, "stale")
    assert result.text == "Lab 2026-04-10, order 2026-05-11: 31 days, window 30, stale."


def test_lab_line_for_a_lab_exactly_at_the_window() -> None:
    result = lab_recency(D(2026, 4, 10), D(2026, 5, 25), 45)
    assert (result.age, result.verdict) == (45, "within window")
    assert result.text == "Lab 2026-04-10, order 2026-05-25: 45 days, window 45, within window."


def test_lab_line_uses_the_singular_for_one_day_and_zero_days_is_fine() -> None:
    assert lab_recency(D(2026, 5, 3), D(2026, 5, 4), 30).text == "Lab 2026-05-03, order 2026-05-04: 1 day, window 30, within window."
    assert lab_recency(D(2026, 5, 4), D(2026, 5, 4), 30).text.endswith("0 days, window 30, within window.")


def test_lab_line_for_a_lab_after_the_order() -> None:
    result = lab_recency(D(2026, 5, 5), D(2026, 5, 4), 30)
    assert (result.age, result.verdict) == (-1, "after the order date")
    assert result.text == "Lab 2026-05-05, order 2026-05-04: -1 days, window 30, after the order date."


def test_lab_line_when_the_date_is_missing() -> None:
    result = lab_recency(None, D(2026, 5, 4), 30)
    assert (result.age, result.verdict) == (None, "date missing or unreadable")
    assert result.text == "Lab date missing or unreadable."


@pytest.mark.parametrize("payer", ["payer_a", "payer_b", "payer_c"])
def test_the_lab_verdict_always_agrees_with_check_packet(payer: str) -> None:
    """The screen recomputes the age; this keeps it from ever disagreeing with the real check."""
    window = lookup_payer_checklist(payer).lab_window_days
    order_date = D(2026, 6, 1)
    checklist = PayerChecklist(
        payer_id=payer,
        lab_window_days=window,
        items=[ChecklistItem(name="recent_lab", description="Lab", kind="lab")],
    )
    order = Order(case_id="t", payer_id=payer, order_date=order_date, drug="d", dose="1 mg", schedule="s")
    expected = {"within window": [], "stale": ["stale"], "after the order date": ["conflicting"], "date missing or unreadable": ["conflicting"]}
    for age in [-3, -1, 0, 1, window - 1, window, window + 1, window + 30, None]:
        lab_date = None if age is None else order_date - datetime.timedelta(days=age)
        field = ExtractedField(name="recent_lab", value="Lab", quote="Lab drawn", found=True, date=lab_date)
        reasons = [g.reason for g in check_packet([field], checklist, order)]
        assert reasons == expected[lab_recency(lab_date, order_date, window).verdict], (payer, age)


def test_the_stale_lab_row_shows_the_recency_line(tmp_path: Path) -> None:
    html = _run("hand_04", tmp_path).evidence_html
    assert "Lab 2026-04-10, order 2026-05-11: 31 days, window 30, stale." in html


def test_the_within_window_lab_row_shows_the_recency_line(tmp_path: Path) -> None:
    view = _run("hand_05", tmp_path)
    assert "READY" in view.status_html
    assert "Lab 2026-04-10, order 2026-05-25: 45 days, window 45, within window." in view.evidence_html


def test_a_lab_without_a_date_says_so_in_its_row(tmp_path: Path) -> None:
    _, note = load_case("hand_01")
    fields = _extraction(note)(None).fields
    fields = [f.model_copy(update={"date": None}) if f.name == "recent_lab" else f for f in fields]
    view = run_case(_form("hand_01"), llm=lambda p: ExtractionResult(fields=fields), outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    assert "Lab date missing or unreadable." in view.evidence_html


def test_a_missing_lab_has_no_recency_line(tmp_path: Path) -> None:
    _, note = load_case("hand_01")
    fields = [f for f in _extraction(note)(None).fields if f.name != "recent_lab"]
    view = run_case(_form("hand_01"), llm=lambda p: ExtractionResult(fields=fields), outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    assert "window" not in view.evidence_html.split("Evidence for each item")[1]


def test_only_the_lab_row_gets_a_recency_line(tmp_path: Path) -> None:
    html = _run("hand_05", tmp_path).evidence_html
    assert html.count(", window ") == 1  # one recency line for the whole table
    assert "Lab date missing or unreadable" not in html  # non-lab rows have no date and must stay silent


# --- the payer requirements panel and Required marking ----------------------------------------


def _panel_items(html: str) -> list[str]:
    return re.findall(r"<strong>(\w+)</strong>", html)


@pytest.mark.parametrize("payer", ["payer_a", "payer_b", "payer_c"])
def test_the_panel_lists_exactly_the_payers_required_items_and_lab_window(payer: str) -> None:
    checklist = lookup_payer_checklist(payer)
    html = render_requirements(payer)
    assert _panel_items(html) == [i.name for i in checklist.items]
    assert f"within {checklist.lab_window_days} days of the order date" in html
    import html as html_lib

    for item in checklist.items:
        assert html_lib.escape(item.description) in html  # shown escaped, like all dynamic text


def test_the_panel_differs_between_payers() -> None:
    a, b, c = (render_requirements(p) for p in ["payer_a", "payer_b", "payer_c"])
    assert "prior_therapies" not in _panel_items(a) and "prior_therapies" in _panel_items(b)
    assert "stage" in _panel_items(a) and "stage" not in _panel_items(c)
    assert "30 days" in a and "45 days" in b and "60 days" in c


def test_the_panel_says_the_rules_are_invented() -> None:
    assert "invented" in render_requirements("payer_a").lower()


@pytest.mark.parametrize("bad", ["", None, "payer_z"])
def test_the_panel_handles_a_missing_or_unknown_payer(bad: object) -> None:
    assert "Choose a payer" in render_requirements(bad)  # type: ignore[arg-type]


def test_the_panel_is_a_collapsed_accordion_filled_for_the_default_payer_and_updated_on_change() -> None:
    import gradio as gr

    demo = build_app(llm=lambda p: ExtractionResult(fields=[]), outreach_llm=_outreach)
    accordions = [b for b in demo.blocks.values() if isinstance(b, gr.Accordion) and b.label == "Payer requirements"]
    assert len(accordions) == 1 and accordions[0].open is False
    assert render_requirements("payer_a") in {getattr(b, "value", None) for b in demo.blocks.values()}
    payer = next(b for b in demo.blocks.values() if isinstance(b, gr.Dropdown) and b.label == "Payer")
    targets = [tuple(t) for dep in demo.config["dependencies"] for t in dep["targets"]]
    assert (payer._id, "change") in targets


def _table(html: str) -> str:
    return html.split("Evidence for each item")[1]


def _row(html: str, item: str) -> str:
    return next(r for r in _table(html).split("<tr>") if f"<td>{item}</td>" in r)


def test_every_row_is_marked_required_or_not_for_the_payer(tmp_path: Path) -> None:
    html = _run("hand_01", tmp_path).evidence_html  # payer_a: no prior_therapies
    for item in ["diagnosis", "stage", "biomarker", "recent_lab", "treatment_plan"]:
        assert "Required" in _row(html, item) and "Not required" not in _row(html, item)
    row = _row(html, "prior_therapies")
    assert "Not required by this payer" in row and "not checked" in row


def test_stage_is_not_required_for_payer_c(tmp_path: Path) -> None:
    html = _run("hand_06", tmp_path).evidence_html  # payer_c
    row = _row(html, "stage")
    assert "Not required by this payer" in row and "not checked" in row
    assert "Stage IV (M1b)" not in row  # no quote is shown for an item nobody asked about


def test_payer_b_requires_all_six_items(tmp_path: Path) -> None:
    html = _run("hand_05", tmp_path).evidence_html
    assert "Not required" not in html
    assert len(re.findall(r"<td>Required</td>", html)) == 6


def test_a_required_item_the_model_left_out_shows_as_required_and_not_found(tmp_path: Path) -> None:
    _, note = load_case("hand_05")
    fields = [f for f in _extraction(note)(None).fields if f.name != "prior_therapies"]
    view = run_case(_form("hand_05"), llm=lambda p: ExtractionResult(fields=fields), outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    row = _row(view.evidence_html, "prior_therapies")
    assert "Required" in row and "not found" in row
    assert "prior_therapies" in view.evidence_html.split("Evidence for each item")[0]  # and it is a gap


def test_when_the_model_fails_required_items_say_not_checked_rather_than_not_found(tmp_path: Path) -> None:
    def broken(prompt: str) -> ExtractionResult:
        raise RuntimeError("boom")

    view = run_case(_form("hand_01"), llm=broken, outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    html = view.evidence_html
    assert "not found" not in html
    assert "not checked" in _row(html, "stage")
    assert "Required" in _row(html, "stage")


def test_the_table_lists_items_in_a_stable_order(tmp_path: Path) -> None:
    html = _table(_run("hand_01", tmp_path).evidence_html)
    names = re.findall(r"<tr><td>(\w+)</td>", html)
    assert names == ["diagnosis", "stage", "biomarker", "prior_therapies", "recent_lab", "treatment_plan"]
