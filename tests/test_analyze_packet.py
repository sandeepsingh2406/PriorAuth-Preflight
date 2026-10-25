"""Tests for analyze_packet, the Packet Analyst pipeline (LLM always mocked)."""
import datetime
import json
from pathlib import Path

import pytest

from src.agents import AnalysisRun, analyze_packet
from src.models import AuditEntry, ExtractedField, ExtractionResult, Order

NOTE = (
    "SYNTHETIC NOTE\n"
    "Diagnosis: Lung adenocarcinoma (ICD-10 C34.90).\n"
    "Staging: Stage IIIB (cT3 N2 M0).\n"
    "Assessment: stage IV disease on review.\n"
    "Biomarkers: EGFR exon 19 deletion detected.\n"
    "Labs: CBC drawn 2026-04-22.\n"
    "Plan: Start Examplumab 200 mg IV every 21 days.\n"
)
ORDER = Order(
    case_id="t_001",
    payer_id="payer_a",
    order_date=datetime.date(2026, 5, 4),
    drug="Examplumab",
    dose="200 mg",
    schedule="every 21 days",
)
PIPELINE = ["lookup_payer_checklist", "extract_fields", "validate_citations", "check_packet"]


def _f(name: str, quote: str | None, **kw: object) -> ExtractedField:
    return ExtractedField(name=name, value=quote, quote=quote, found=quote is not None, **kw)


def _complete() -> list[ExtractedField]:
    """What a careful model would return for NOTE with payer_a: everything documented."""
    return [
        _f("diagnosis", "Diagnosis: Lung adenocarcinoma (ICD-10 C34.90)."),
        _f("stage", "Staging: Stage IIIB (cT3 N2 M0)."),
        _f("biomarker", "Biomarkers: EGFR exon 19 deletion detected."),
        _f("recent_lab", "Labs: CBC drawn 2026-04-22.", date=datetime.date(2026, 4, 22)),
        _f("treatment_plan", "Plan: Start Examplumab 200 mg IV every 21 days."),
    ]


def _replace(fields: list[ExtractedField], new: ExtractedField) -> list[ExtractedField]:
    return [new if f.name == new.name else f for f in fields]


def _llm(fields: list[ExtractedField]):
    return lambda prompt: ExtractionResult(fields=fields)


def _run(fields: list[ExtractedField], log: Path) -> AnalysisRun:
    return analyze_packet(ORDER, NOTE, llm=_llm(fields), audit_path=log)


def _audit(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_ready_packet(tmp_path: Path) -> None:
    """Every item documented with verbatim quotes and a recent lab: READY, no gaps."""
    run = _run(_complete(), tmp_path / "log.jsonl")
    assert run.report.status == "READY"
    assert run.report.gaps == []
    assert run.report.escalation_reason is None
    assert run.tools_called == PIPELINE


def test_not_ready_packet(tmp_path: Path) -> None:
    """A missing item gives NOT_READY with that gap, and no escalation."""
    fields = _replace(_complete(), _f("biomarker", None))
    run = _run(fields, tmp_path / "log.jsonl")
    assert run.report.status == "NOT_READY"
    assert [(g.item, g.reason) for g in run.report.gaps] == [("biomarker", "missing")]
    assert "escalate_to_human" not in run.tools_called


def test_conflict_is_escalated_with_a_code_written_reason(tmp_path: Path) -> None:
    """Conflicting evidence routes to NEEDS_REVIEW through escalate_to_human, with a reason."""
    stage = _f("stage", "Staging: Stage IIIB (cT3 N2 M0).", conflicting_quotes=["stage IV disease"])
    run = _run(_replace(_complete(), stage), tmp_path / "log.jsonl")
    assert run.report.status == "NEEDS_REVIEW"
    assert run.tools_called == PIPELINE + ["escalate_to_human"]
    assert run.report.escalation_reason and "stage" in run.report.escalation_reason
    assert "stage IV disease" not in run.report.escalation_reason  # no quoted note text


def test_invented_quote_is_dropped_and_code_decides_the_status(tmp_path: Path) -> None:
    """The model claims a biomarker with a quote that is not in the note; code marks it missing."""
    fake = _f("biomarker", "Biomarkers: ALK rearrangement detected.")
    run = _run(_replace(_complete(), fake), tmp_path / "log.jsonl")
    assert run.report.status == "NOT_READY"
    assert [(g.item, g.reason) for g in run.report.gaps] == [("biomarker", "missing")]
    raw = {f.name: f for f in run.raw_fields}
    assert raw["biomarker"].quote == "Biomarkers: ALK rearrangement detected."  # kept for the gate
    final = {f.name: f for f in run.report.fields}
    assert final["biomarker"].found is False and final["biomarker"].quote is None


def test_every_quote_in_the_report_is_verbatim(tmp_path: Path) -> None:
    fields = _replace(_complete(), _f("stage", "Stage 3B", conflicting_quotes=["not in note"]))
    run = _run(fields, tmp_path / "log.jsonl")
    for f in run.report.fields:
        assert f.quote is None or f.quote in NOTE
        assert all(q in NOTE for q in f.conflicting_quotes)


def test_one_audit_line_per_run_with_the_spec_fields(tmp_path: Path) -> None:
    """Each run appends exactly one line: case id, tools called, status, timestamp."""
    log = tmp_path / "log.jsonl"
    _run(_complete(), log)
    _run(_replace(_complete(), _f("biomarker", None)), log)
    stage = _f("stage", "Staging: Stage IIIB (cT3 N2 M0).", conflicting_quotes=["stage IV disease"])
    _run(_replace(_complete(), stage), log)
    lines = _audit(log)
    assert [line["status"] for line in lines] == ["READY", "NOT_READY", "NEEDS_REVIEW"]
    assert all(line["case_id"] == "t_001" for line in lines)
    assert lines[2]["tools_called"] == PIPELINE + ["escalate_to_human"]
    assert all(AuditEntry.model_validate(line) for line in lines)


def test_audit_log_never_contains_note_text(tmp_path: Path) -> None:
    """Not the note, not a quote, not a single distinctive line of it."""
    log = tmp_path / "log.jsonl"
    stage = _f("stage", "Staging: Stage IIIB (cT3 N2 M0).", conflicting_quotes=["stage IV disease"])
    _run(_replace(_complete(), stage), log)
    text = log.read_text()
    for line in NOTE.splitlines()[1:]:
        assert line not in text
    for fragment in ["EGFR", "Examplumab", "IIIB", "stage IV disease", "C34.90"]:
        assert fragment not in text


def test_model_failure_goes_to_a_person_and_is_still_audited(tmp_path: Path) -> None:
    """If the model call fails, the packet is NEEDS_REVIEW, the reason names only the error type."""

    def broken(prompt: str) -> ExtractionResult:
        raise RuntimeError("upstream said: Biomarkers: EGFR exon 19 deletion detected.")

    log = tmp_path / "log.jsonl"
    run = analyze_packet(ORDER, NOTE, llm=broken, audit_path=log)
    assert run.report.status == "NEEDS_REVIEW"
    assert "RuntimeError" in run.report.escalation_reason
    assert "EGFR" not in run.report.escalation_reason
    assert run.raw_fields == []
    assert run.tools_called == ["lookup_payer_checklist", "extract_fields", "escalate_to_human"]
    (line,) = _audit(log)
    assert line["status"] == "NEEDS_REVIEW"
    assert "EGFR" not in log.read_text()


def test_unknown_payer_is_an_input_error(tmp_path: Path) -> None:
    """A case file naming a payer we do not have is a broken input, not a review case."""
    order = ORDER.model_copy(update={"payer_id": "payer_z"})
    with pytest.raises(ValueError, match="payer_z"):
        analyze_packet(order, NOTE, llm=_llm(_complete()), audit_path=tmp_path / "log.jsonl")


def test_default_audit_path_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without audit_path, the line goes where AUDIT_LOG_PATH points (temp in tests)."""
    log = tmp_path / "env_log.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log))
    analyze_packet(ORDER, NOTE, llm=_llm(_complete()))
    assert len(_audit(log)) == 1


# --- unreadable text -----------------------------------------------------------

BAD = "EG?R ex@n 1# d%l%tion [unreadable]"
NOTE_WITH_BAD = NOTE + BAD + "\n"


def _llm_u(fields: list[ExtractedField], passages: list[str]):
    return lambda prompt: ExtractionResult(fields=fields, unreadable_passages=passages)


def _run_u(fields, passages, log: Path, note: str = NOTE_WITH_BAD) -> AnalysisRun:
    return analyze_packet(ORDER, note, llm=_llm_u(fields, passages), audit_path=log)


def test_verified_unreadable_text_sends_an_otherwise_ready_packet_to_review(tmp_path: Path) -> None:
    run = _run_u(_complete(), [BAD], tmp_path / "log.jsonl")
    assert run.report.status == "NEEDS_REVIEW"
    assert run.report.gaps == []
    assert run.report.unreadable_passages == [BAD]
    assert run.tools_called == PIPELINE + ["escalate_to_human"]
    assert "unreadable" in run.report.escalation_reason.lower()
    assert BAD not in run.report.escalation_reason  # the reason never repeats note text


def test_unreadable_text_overrides_not_ready(tmp_path: Path) -> None:
    """The garbled section hides the biomarker: the gap stays listed, but a person must look."""
    fields = _replace(_complete(), _f("biomarker", None))
    run = _run_u(fields, [BAD], tmp_path / "log.jsonl")
    assert run.report.status == "NEEDS_REVIEW"
    assert [(g.item, g.reason) for g in run.report.gaps] == [("biomarker", "missing")]


def test_a_reported_passage_that_is_not_in_the_note_is_ignored(tmp_path: Path) -> None:
    """The model cannot send a packet to review with text it made up."""
    run = _run_u(_complete(), ["totally invented gibberish"], tmp_path / "log.jsonl")
    assert run.report.status == "READY"
    assert run.report.unreadable_passages == []
    assert run.raw_unreadable == ["totally invented gibberish"]  # kept so the evaluation can count it


def test_unreadable_text_and_a_conflict_give_one_combined_reason(tmp_path: Path) -> None:
    stage = _f("stage", "Staging: Stage IIIB (cT3 N2 M0).", conflicting_quotes=["stage IV disease"])
    run = _run_u(_replace(_complete(), stage), [BAD], tmp_path / "log.jsonl")
    assert run.report.status == "NEEDS_REVIEW"
    reason = run.report.escalation_reason.lower()
    assert "stage" in reason and "unreadable" in reason
    assert run.tools_called.count("escalate_to_human") == 1


def test_audit_log_never_contains_the_unreadable_text(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    _run_u(_complete(), [BAD], log)
    assert "EG?R" not in log.read_text()
    assert _audit(log)[0]["status"] == "NEEDS_REVIEW"


# --- the Outreach draft ----------------------------------------------------------

from src.models import DraftText  # noqa: E402


class FakeOutreach:
    """Stands in for the Outreach model and records every prompt it is given."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> DraftText:
        self.prompts.append(prompt)
        return DraftText(subject="Items needed", body="Please send the listed items.")


def _not_ready_fields() -> list[ExtractedField]:
    return _replace(_complete(), _f("biomarker", None))


def test_no_draft_unless_asked(tmp_path: Path) -> None:
    """Drafting is opt-in, so a plain run makes no extra model call."""
    run = _run(_not_ready_fields(), tmp_path / "log.jsonl")
    assert run.report.status == "NOT_READY"
    assert run.draft is None and run.draft_error is None
    assert "draft_request" not in run.tools_called


def test_not_ready_packet_gets_a_draft_when_asked(tmp_path: Path) -> None:
    outreach = FakeOutreach()
    log = tmp_path / "log.jsonl"
    run = analyze_packet(ORDER, NOTE, llm=_llm(_not_ready_fields()), audit_path=log, draft=True, outreach_llm=outreach)
    assert run.report.status == "NOT_READY"
    assert run.draft is not None
    assert (run.draft.subject, run.draft.requested_items) == ("Items needed", ["biomarker"])
    assert run.tools_called == PIPELINE + ["draft_request"]
    assert _audit(log)[0]["tools_called"] == PIPELINE + ["draft_request"]


def test_ready_packet_gets_no_draft(tmp_path: Path) -> None:
    outreach = FakeOutreach()
    run = analyze_packet(ORDER, NOTE, llm=_llm(_complete()), audit_path=tmp_path / "log.jsonl", draft=True, outreach_llm=outreach)
    assert run.report.status == "READY" and run.draft is None
    assert outreach.prompts == []
    assert run.draft_error is None and run.tools_called == PIPELINE  # no attempt at all, not even a failed one


def test_needs_review_packet_goes_to_a_person_without_a_draft(tmp_path: Path) -> None:
    """Conflicts are read by a person first; Outreach is never called."""
    outreach = FakeOutreach()
    stage = _f("stage", "Staging: Stage IIIB (cT3 N2 M0).", conflicting_quotes=["stage IV disease"])
    run = analyze_packet(
        ORDER, NOTE, llm=_llm(_replace(_complete(), stage)), audit_path=tmp_path / "log.jsonl", draft=True, outreach_llm=outreach
    )
    assert run.report.status == "NEEDS_REVIEW" and run.draft is None
    assert outreach.prompts == []
    assert run.draft_error is None and "draft_request_failed" not in run.tools_called


def test_needs_review_with_missing_items_still_gets_no_draft(tmp_path: Path) -> None:
    """Unreadable text plus a missing biomarker: a person looks first, nothing is drafted."""
    outreach = FakeOutreach()
    run = analyze_packet(
        ORDER,
        NOTE_WITH_BAD,
        llm=_llm_u(_not_ready_fields(), [BAD]),
        audit_path=tmp_path / "log.jsonl",
        draft=True,
        outreach_llm=outreach,
    )
    assert run.report.status == "NEEDS_REVIEW" and run.draft is None
    assert outreach.prompts == []
    assert run.draft_error is None and "draft_request_failed" not in run.tools_called


def test_an_outreach_failure_leaves_the_packet_not_ready_and_is_audited(tmp_path: Path) -> None:
    """If the Outreach model fails, the analysis stands; only the draft is missing."""

    def broken(prompt: str) -> DraftText:
        raise RuntimeError("upstream said: Biomarkers: EGFR exon 19 deletion detected.")

    log = tmp_path / "log.jsonl"
    run = analyze_packet(ORDER, NOTE, llm=_llm(_not_ready_fields()), audit_path=log, draft=True, outreach_llm=broken)
    assert run.report.status == "NOT_READY"
    assert run.draft is None and run.draft_error == "RuntimeError"
    assert run.tools_called == PIPELINE + ["draft_request_failed"]
    (line,) = _audit(log)
    assert line["status"] == "NOT_READY"
    assert "EGFR" not in log.read_text()


def test_default_outreach_path_uses_the_sdk_and_a_blocked_call_is_a_draft_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, network_attempts: list[str]
) -> None:
    """With no fake, drafting goes through the real SDK (blocked in tests) and fails soft."""
    monkeypatch.setenv("MODEL_DEV", "test-model")
    run = analyze_packet(ORDER, NOTE, llm=_llm(_not_ready_fields()), audit_path=tmp_path / "log.jsonl", draft=True)
    assert run.draft is None and run.draft_error == "NetworkCallInTest"
    assert len(network_attempts) == 1
    network_attempts.clear()  # expected
