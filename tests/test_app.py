"""Tests for the Gradio screen's logic in src/app.py (fake models, temp audit log, no browser)."""
import datetime
import json
import re
from pathlib import Path

import pytest

from src.app import (
    CaseForm,
    build_app,
    case_id_for,
    decide,
    list_cases,
    load_case_into_form,
    main,
    run_case,
)
from src.models import DraftText, ExtractedField, ExtractionResult
from src.run import load_case

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
    assert "no evidence found in the note" in view.evidence_html
    assert view.show_draft is True and view.show_reviewed is False
    assert (view.subject, view.body) == ("Items needed for the packet", "Please send the listed items.")
    assert "not sent" in view.draft_label.lower()


def test_needs_review_case_shows_both_conflicting_quotes_and_the_reason_without_a_draft(tmp_path: Path) -> None:
    view = _run("hand_08", tmp_path, conflicts={"stage": ["stage IV non-small cell lung cancer"]})
    assert "NEEDS_REVIEW" in view.status_html
    assert "Staging: Stage II (cT2 N0 M0)" in view.evidence_html
    assert "stage IV non-small cell lung cancer" in view.evidence_html
    assert "Escalation reason" in view.evidence_html
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
