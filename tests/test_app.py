"""Tests for the Gradio screen's logic in src/app.py (fake models, temp audit log, no browser)."""
import datetime
import json
import re
from pathlib import Path

import pytest

from src.app import (
    APPROVE_LABEL,
    BANNER,
    DISCARD_LABEL,
    EDIT_LABEL,
    PALETTE,
    REVIEWED_LABEL,
    RUN_LABEL,
    SOLID,
    CaseForm,
    build_css,
    build_theme,
    build_app,
    case_id_for,
    decide,
    lab_recency,
    list_cases,
    pill,
    load_case_into_form,
    main,
    record_number_from,
    render_requirements,
    run_case,
)
from src.models import ChecklistItem, DraftText, ExtractedField, ExtractionResult, Order, PayerChecklist
from src.run import load_case
from src.tools import check_packet, lookup_payer_checklist

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
    for label in [RUN_LABEL, APPROVE_LABEL, EDIT_LABEL, DISCARD_LABEL, REVIEWED_LABEL]:
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
    assert ".btn-approve" in captured["css"]  # the colour rules are handed to launch
    import gradio as gr

    assert isinstance(captured["theme"], gr.themes.Soft)


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
    assert variants[RUN_LABEL] == "primary"
    assert variants[APPROVE_LABEL] == "secondary"
    assert [v for v, variant in variants.items() if variant == "primary"] == [RUN_LABEL]


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
    return next(r for r in _table(html).split("<tr") if f"<td>{item}</td>" in r)


def _text(cell: str) -> str:
    """A table cell as a reader sees it: icons and markup removed, words kept."""
    cell = re.sub(r'<span class="icon"[^>]*>.*?</span>', "", cell)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cell)).strip()


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
    assert len(re.findall(r'pill-info"><span class="icon" aria-hidden="true">[^<]*</span>Required</span>', html)) == 6


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
    assert "not found" not in html.split("<table")[1]  # the legend names it; no row may claim it
    assert "not checked" in _row(html, "stage")
    assert "Required" in _row(html, "stage")


def test_the_table_lists_items_in_a_stable_order(tmp_path: Path) -> None:
    html = _table(_run("hand_01", tmp_path).evidence_html)
    names = re.findall(r'<tr class="row-\w+"><td>(\w+)</td>', html)
    assert names == ["diagnosis", "stage", "biomarker", "prior_therapies", "recent_lab", "treatment_plan"]


# --- the record number in the draft ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("SYNTHETIC NOTE\nRecord number: SYN-90004\nVisit date: 2026-05-08", "SYN-90004"),
        ("Patient: Riley Placeholder  MRN: SYN-00001\nVisit", "SYN-00001"),
        ("Record number: SYN-90004\nMRN: SYN-00002", "SYN-90004"),  # the first one wins
        ("no identifier here", None),
        ("MRN: 123456789", None),  # only synthetic SYN- numbers are ever carried onto the screen
        ("Record number: 555-12-3456", None),
        ("record number: SYN-90004", None),  # the note format is fixed; no guessing
        ("", None),
    ],
)
def test_record_number_is_read_from_the_note_but_only_if_synthetic(note: str, expected: str | None) -> None:
    assert record_number_from(note) == expected


class RecordingOutreach:
    """A fake Outreach model that writes the case id into its text, like the real one tends to."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> DraftText:
        self.prompts.append(prompt)
        case_id = re.search(r"case (\S+?),", prompt).group(1)
        return DraftText(subject=f"Items needed for case {case_id}", body=f"For case {case_id}, please send the listed items. (see {case_id}b too)")


def _run_recording(case_id: str, tmp_path: Path, **changes: str):
    form = _form(case_id, **changes)
    outreach = RecordingOutreach()
    view = run_case(form, llm=_extraction(form.note), outreach_llm=outreach, audit_path=tmp_path / "audit.jsonl")
    return view, outreach


def test_the_draft_shown_names_the_record_number_not_the_case_id(tmp_path: Path) -> None:
    view, _ = _run_recording("hand_04", tmp_path)
    assert view.subject == "Items needed for case SYN-90004"
    assert view.body.startswith("For case SYN-90004, please send")
    assert not re.search(r"(?<![\w-])hand_04(?![\w-])", view.subject + view.body)  # the exact id is gone (hand_04b is a different token)


def test_only_the_exact_case_id_is_replaced(tmp_path: Path) -> None:
    view, _ = _run_recording("hand_04", tmp_path)
    assert "(see hand_04b too)" in view.body  # a longer token that merely starts with the id is left alone


def test_the_outreach_model_never_sees_the_record_number(tmp_path: Path) -> None:
    """Rule 6: the number comes from the note, so it is swapped in only after the model has written."""
    view, outreach = _run_recording("hand_04", tmp_path)
    (prompt,) = outreach.prompts
    assert "SYN-" not in prompt and "90004" not in prompt
    assert "hand_04" in prompt  # it still works from the case id


def test_without_a_record_number_the_case_id_stays(tmp_path: Path) -> None:
    _, note = load_case("hand_04")
    stripped = "\n".join(l for l in note.splitlines() if not l.startswith("Record number"))
    view, _ = _run_recording("hand_04", tmp_path, note=stripped)
    assert "pasted_note" in view.subject  # the note changed, so the audit case id is pasted_note, and there is no number to swap in


def test_a_real_looking_record_number_is_not_swapped_in(tmp_path: Path) -> None:
    _, note = load_case("hand_04")
    real_looking = note.replace("Record number: SYN-90004", "Record number: 4432871")
    view, _ = _run_recording("hand_04", tmp_path, note=real_looking)
    assert "4432871" not in view.subject + view.body


def test_approving_the_untouched_draft_logs_only_approved(tmp_path: Path) -> None:
    """The comparison is against the text on screen (with the record number), not the model's original."""
    view, _ = _run_recording("hand_04", tmp_path)
    decide(view.session, "approve", view.subject, view.body, audit_path=tmp_path / "audit.jsonl")
    assert [l["human_decision"] for l in _audit(tmp_path)] == [None, "approved"]


def test_edit_with_no_changes_is_still_recognised_after_the_swap(tmp_path: Path) -> None:
    view, _ = _run_recording("hand_04", tmp_path)
    assert "no changes" in decide(view.session, "edit", view.subject, view.body, audit_path=tmp_path / "audit.jsonl").lower()
    assert len(_audit(tmp_path)) == 1


def test_changing_the_swapped_text_is_an_edit(tmp_path: Path) -> None:
    view, _ = _run_recording("hand_04", tmp_path)
    decide(view.session, "approve", view.subject, view.body + " Thanks!", audit_path=tmp_path / "audit.jsonl")
    assert [l["human_decision"] for l in _audit(tmp_path)] == [None, "edited", "approved"]


def test_the_record_number_never_reaches_the_audit_log(tmp_path: Path) -> None:
    view, _ = _run_recording("hand_04", tmp_path)
    decide(view.session, "approve", view.subject, view.body, audit_path=tmp_path / "audit.jsonl")
    assert "SYN-90004" not in (tmp_path / "audit.jsonl").read_text()


def test_the_run_itself_keeps_the_original_draft_for_the_record(tmp_path: Path) -> None:
    """The model's own text is untouched on the run; only what is shown to the person is swapped."""
    view, _ = _run_recording("hand_04", tmp_path)
    assert "hand_04" in view.session.run.draft.subject and "SYN-90004" not in view.session.run.draft.subject


# --- conflicting items are marked in the table, not just in the gap list ---------------------------


def _cells(row: str) -> list[str]:
    return re.findall(r"<td>(.*?)</td>", row, flags=re.S)


def test_a_conflicting_item_says_conflicting_and_shows_both_quotes_in_its_row(tmp_path: Path) -> None:
    html = _run("hand_08", tmp_path, conflicts={"stage": ["stage IV non-small cell lung cancer"]}).evidence_html
    cells = _cells(_row(html, "stage"))
    assert _text(cells[1]) == "Required" and _text(cells[2]) == "conflicting"
    assert "Staging: Stage II (cT2 N0 M0)" in cells[3]
    assert "stage IV non-small cell lung cancer" in cells[3]


def test_items_without_a_conflict_still_say_found(tmp_path: Path) -> None:
    html = _run("hand_08", tmp_path, conflicts={"stage": ["stage IV non-small cell lung cancer"]}).evidence_html
    assert _text(_cells(_row(html, "diagnosis"))[2]) == "found"
    assert _text(_cells(_row(html, "treatment_plan"))[2]) == "found"


def test_a_conflicting_lab_keeps_its_recency_line(tmp_path: Path) -> None:
    other = "WBC 6.1, hemoglobin 12.4, platelets 214, creatinine 0.9."  # really in the note, so it survives validation
    html = _run("hand_01", tmp_path, conflicts={"recent_lab": [other]}).evidence_html
    cells = _cells(_row(html, "recent_lab"))
    assert _text(cells[2]) == "conflicting"
    assert other in cells[3]
    assert "Lab 2026-04-22, order 2026-05-04: 12 days, window 30, within window." in cells[3]


def test_a_lab_with_no_date_is_marked_conflicting_with_the_reason(tmp_path: Path) -> None:
    _, note = load_case("hand_01")
    fields = [f.model_copy(update={"date": None}) if f.name == "recent_lab" else f for f in _extraction(note)(None).fields]
    view = run_case(_form("hand_01"), llm=lambda p: ExtractionResult(fields=fields), outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    cells = _cells(_row(view.evidence_html, "recent_lab"))
    assert _text(cells[2]) == "conflicting" and "Lab date missing or unreadable." in cells[3]


def test_a_failed_main_quote_does_not_hide_a_real_conflicting_quote(tmp_path: Path) -> None:
    """The model's main quote is invented and dropped; the real contradicting quote is still shown."""
    _, note = load_case("hand_08")
    real = "Assessment: 66-year-old with stage IV non-small cell lung cancer, here to start immunotherapy."
    fields = [
        ExtractedField(name="stage", value="Stage IX", quote="Stage IX per nobody", found=True, conflicting_quotes=[real])
        if f.name == "stage" else f
        for f in _extraction(note)(None).fields
    ]
    view = run_case(_form("hand_08"), llm=lambda p: ExtractionResult(fields=fields), outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl")
    cells = _cells(_row(view.evidence_html, "stage"))
    assert _text(cells[2]) == "conflicting"
    assert real in cells[3] and "Stage IX per nobody" not in cells[3]


def test_a_conflicting_row_is_escaped(tmp_path: Path) -> None:
    """The risky text is really in the note (so it survives validation) and must still be shown as text."""
    evil = "<script>alert(1)</script>"
    _, note = load_case("hand_08")
    html = _run("hand_08", tmp_path, conflicts={"stage": [evil]}, note=note + evil + "\n").evidence_html
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in _cells(_row(html, "stage"))[3]


# --- colours and icons: the palette --------------------------------------------------------------


def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    linear = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("state", sorted(PALETTE))
def test_pill_text_is_readable_on_its_background(state: str) -> None:
    """WCAG AA needs 4.5:1 for normal text; a future colour edit cannot quietly break it."""
    text, background, _edge = PALETTE[state]
    assert _contrast(text, background) >= 4.5, state


@pytest.mark.parametrize("name", sorted(SOLID))
def test_white_text_is_readable_on_every_solid_badge_and_button(name: str) -> None:
    assert _contrast("#ffffff", SOLID[name]) >= 4.5, name


def test_the_palette_covers_the_states_the_plan_names() -> None:
    assert {"ok", "warn", "bad", "muted", "info"} <= set(PALETTE)


def test_each_state_has_its_own_colour() -> None:
    backgrounds = [bg for _, bg, _ in PALETTE.values()]
    assert len(set(backgrounds)) == len(backgrounds)


def test_approve_edit_and_discard_are_three_different_colours() -> None:
    colours = {SOLID["btn-approve"], SOLID["btn-edit"], SOLID["btn-discard"]}
    assert len(colours) == 3


def test_the_css_defines_every_class_the_page_uses() -> None:
    css = build_css()
    for state in PALETTE:
        for prefix in ["pill", "callout", "row"]:
            assert f".{prefix}-{state}" in css, (prefix, state)
    for name in SOLID:
        assert f".{name}" in css, name


def test_each_button_rule_uses_its_own_colour_and_wins_over_the_theme() -> None:
    css = build_css()
    for name in ["btn-approve", "btn-edit", "btn-discard", "btn-reviewed"]:
        rule = re.search(rf"\.{name}\b[^{{]*{{([^}}]*)}}", css).group(1)
        assert SOLID[name] in rule and "!important" in rule, name


def test_the_css_is_fixed_text_with_no_user_input() -> None:
    assert build_css() == build_css()
    assert "<" not in build_css()  # no markup can ride in a stylesheet


def test_the_action_buttons_carry_their_colour_classes() -> None:
    import gradio as gr

    demo = build_app(llm=lambda p: ExtractionResult(fields=[]), outreach_llm=_outreach)
    classes = {b.value: list(b.elem_classes or []) for b in demo.blocks.values() if isinstance(b, gr.Button)}
    assert "btn-approve" in classes[APPROVE_LABEL]
    assert "btn-edit" in classes[EDIT_LABEL]
    assert "btn-discard" in classes[DISCARD_LABEL]
    assert "btn-reviewed" in classes[REVIEWED_LABEL]


def test_the_button_labels_carry_an_icon_and_keep_their_words() -> None:
    assert RUN_LABEL.endswith("Run") and EDIT_LABEL.endswith("Edit") and DISCARD_LABEL.endswith("Discard")
    assert APPROVE_LABEL.endswith("Approve draft (logs decision only)")
    assert REVIEWED_LABEL.endswith("Mark reviewed")
    assert all(label[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for label in [RUN_LABEL, APPROVE_LABEL, EDIT_LABEL, DISCARD_LABEL, REVIEWED_LABEL])


def test_the_theme_is_the_soft_indigo_one() -> None:
    import gradio as gr

    assert isinstance(build_theme(), gr.themes.Soft)


# --- pills, status badges and coloured rows ---------------------------------------------------------


def test_a_pill_is_an_icon_plus_its_words() -> None:
    assert pill("ok", "✅", "found") == '<span class="pill pill-ok"><span class="icon" aria-hidden="true">✅</span>found</span>'


def test_a_pill_only_accepts_a_known_colour_state() -> None:
    """Class names come from the fixed palette, never from anything typed or read from a note."""
    with pytest.raises(ValueError):
        pill('ok"><script>', "✅", "found")


def test_pill_text_is_escaped() -> None:
    assert "<b>" not in pill("info", "📌", "<b>x</b>") and "&lt;b&gt;" in pill("info", "📌", "<b>x</b>")


@pytest.mark.parametrize(
    ("case_id", "conflicts", "status", "badge"),
    [
        ("hand_01", None, "READY", "badge-ready"),
        ("hand_07", None, "NOT_READY", "badge-not_ready"),
        ("hand_08", {"stage": ["stage IV non-small cell lung cancer"]}, "NEEDS_REVIEW", "badge-needs_review"),
    ],
)
def test_each_status_has_its_own_badge_with_an_icon_and_the_words(tmp_path: Path, case_id: str, conflicts, status: str, badge: str) -> None:
    html = _run(case_id, tmp_path, conflicts=conflicts).status_html
    assert f'class="badge {badge}"' in html
    assert status in _text(html)
    assert 'class="icon" aria-hidden="true"' in html


def _result_pill(html: str, item: str) -> tuple[str, str]:
    cell = _cells(_row(html, item))[2]
    state = re.search(r"pill-(\w+)", cell).group(1)
    return state, _text(cell)


def test_found_is_green_not_found_is_amber_conflicting_is_red_not_checked_is_gray(tmp_path: Path) -> None:
    conflict = _run("hand_08", tmp_path, conflicts={"stage": ["stage IV non-small cell lung cancer"]}).evidence_html
    assert _result_pill(conflict, "diagnosis") == ("ok", "found")
    assert _result_pill(conflict, "stage") == ("bad", "conflicting")
    assert _result_pill(conflict, "prior_therapies") == ("muted", "not checked")  # payer_a does not ask for it
    missing = _run("hand_07", tmp_path).evidence_html
    assert _result_pill(missing, "biomarker") == ("warn", "not found")
    assert _result_pill(missing, "prior_therapies") == ("warn", "not found")


def test_rows_carry_the_colour_of_their_state(tmp_path: Path) -> None:
    html = _run("hand_08", tmp_path, conflicts={"stage": ["stage IV non-small cell lung cancer"]}).evidence_html
    row_class = lambda item: re.search(r'class="(row-\w+)"', _row(html, item)).group(1)
    assert (row_class("diagnosis"), row_class("stage"), row_class("prior_therapies")) == ("row-ok", "row-bad", "row-muted")
    assert set(re.findall(r'class="(row-\w+)"', html)) <= {f"row-{s}" for s in PALETTE}


def test_required_is_blue_and_not_required_is_gray(tmp_path: Path) -> None:
    html = _run("hand_01", tmp_path).evidence_html  # payer_a
    required = _cells(_row(html, "diagnosis"))[1]
    not_required = _cells(_row(html, "prior_therapies"))[1]
    assert "pill-info" in required and _text(required) == "Required"
    assert "pill-muted" in not_required and _text(not_required) == "Not required by this payer"


def test_after_a_model_failure_required_rows_are_gray_not_checked(tmp_path: Path) -> None:
    def broken(prompt: str) -> ExtractionResult:
        raise RuntimeError("boom")

    html = run_case(_form("hand_01"), llm=broken, outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl").evidence_html
    assert _result_pill(html, "stage") == ("muted", "not checked")


def test_a_legend_explains_every_state_once(tmp_path: Path) -> None:
    html = _run("hand_01", tmp_path).evidence_html
    legend = re.findall(r'<p class="legend">(.*?)</p>', html, flags=re.S)
    assert len(legend) == 1
    for word in ["found", "not found", "stale", "conflicting", "not checked"]:
        assert word in legend[0]


def test_colour_is_never_the_only_signal(tmp_path: Path) -> None:
    """Every pill has words, and every icon is hidden from screen readers."""
    html = _run("hand_08", tmp_path, conflicts={"stage": ["stage IV non-small cell lung cancer"]}).evidence_html
    pills = re.findall(r'<span class="pill pill-\w+">.*?</span></span>|<span class="pill pill-\w+"><span class="icon"[^>]*>[^<]*</span>[^<]*</span>', html)
    assert pills
    for p in pills:
        assert _text(p), p
        assert 'class="icon" aria-hidden="true"' in p
