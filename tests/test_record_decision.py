"""Tests for record_decision: the person's decision is a second, linked audit line."""
import datetime
import json
from pathlib import Path

import pytest

from src.agents import AnalysisRun, analyze_packet, record_decision
from src.models import DraftText, ExtractedField, ExtractionResult, Order

NOTE = (
    "SYNTHETIC NOTE\n"
    "Diagnosis: Lung adenocarcinoma (ICD-10 C34.90).\n"
    "Staging: Stage IIIB (cT3 N2 M0).\n"
    "Labs: CBC drawn 2026-04-22.\n"
    "Plan: Start Examplumab 200 mg IV every 21 days.\n"
)
ORDER = Order(
    case_id="t_001", payer_id="payer_a", order_date=datetime.date(2026, 5, 4),
    drug="Examplumab", dose="200 mg", schedule="every 21 days",
)
DRAFT_BODY = "Zebra-striped distinctive draft body text."


def _f(name: str, quote: str | None, **kw: object) -> ExtractedField:
    return ExtractedField(name=name, value=quote, quote=quote, found=quote is not None, **kw)


def _fields(*, biomarker: bool = False) -> list[ExtractedField]:
    return [
        _f("diagnosis", "Diagnosis: Lung adenocarcinoma (ICD-10 C34.90)."),
        _f("stage", "Staging: Stage IIIB (cT3 N2 M0)."),
        _f("biomarker", "Staging: Stage IIIB (cT3 N2 M0)." if biomarker else None),
        _f("recent_lab", "Labs: CBC drawn 2026-04-22.", date=datetime.date(2026, 4, 22)),
        _f("treatment_plan", "Plan: Start Examplumab 200 mg IV every 21 days."),
    ]


def _run(log: Path, *, kind: str) -> AnalysisRun:
    """kind: 'draft' (NOT_READY with a draft), 'no_draft' (NOT_READY, no draft), 'ready', 'review'."""
    fields = _fields(biomarker=kind in ("ready", "review"))
    if kind == "review":
        fields[1] = _f("stage", "Staging: Stage IIIB (cT3 N2 M0).", conflicting_quotes=["Diagnosis: Lung adenocarcinoma (ICD-10 C34.90)."])
    return analyze_packet(
        ORDER, NOTE,
        llm=lambda p: ExtractionResult(fields=fields),
        audit_path=log,
        draft=(kind == "draft"),
        outreach_llm=lambda p: DraftText(subject="Subject", body=DRAFT_BODY),
    )


def _lines(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_the_decision_is_a_second_line_linked_by_run_id(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    run = _run(log, kind="draft")
    record_decision(run, "approved", audit_path=log)
    first, second = _lines(log)
    assert first["human_decision"] is None
    assert second["human_decision"] == "approved"
    assert second["run_id"] == first["run_id"] == run.run_id
    assert (second["case_id"], second["status"]) == ("t_001", "NOT_READY")
    assert second["tools_called"] == []


def test_recording_a_decision_leaves_the_run_line_untouched(tmp_path: Path) -> None:
    """Append-only: the original line is byte-for-byte the same afterwards."""
    log = tmp_path / "log.jsonl"
    run = _run(log, kind="draft")
    before = log.read_text()
    record_decision(run, "edited", audit_path=log)
    assert log.read_text().startswith(before)
    assert len(_lines(log)) == 2


def test_the_decision_line_is_later_than_the_run_line(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    run = _run(log, kind="draft")
    record_decision(run, "approved", audit_path=log)
    first, second = _lines(log)
    assert second["timestamp"] >= first["timestamp"]
    assert second["timestamp"].endswith("Z")  # UTC


@pytest.mark.parametrize("decision", ["approved", "edited", "discarded"])
def test_a_draft_can_be_approved_edited_or_discarded(tmp_path: Path, decision: str) -> None:
    log = tmp_path / "log.jsonl"
    record_decision(_run(log, kind="draft"), decision, audit_path=log)
    assert _lines(log)[-1]["human_decision"] == decision


@pytest.mark.parametrize("kind", ["no_draft", "ready", "review"])
def test_without_a_draft_the_only_decision_is_reviewed(tmp_path: Path, kind: str) -> None:
    """There is nothing to approve or discard without a draft; the person marks it reviewed."""
    log = tmp_path / "log.jsonl"
    run = _run(log, kind=kind)
    assert run.draft is None
    for refused in ["approved", "edited", "discarded"]:
        with pytest.raises(ValueError, match="reviewed"):
            record_decision(run, refused, audit_path=log)
    record_decision(run, "reviewed", audit_path=log)
    assert _lines(log)[-1]["human_decision"] == "reviewed"
    assert len(_lines(log)) == 2  # the run line and the one valid decision


def test_a_draft_cannot_simply_be_marked_reviewed(tmp_path: Path) -> None:
    """With a draft on screen the person must say what they did with it."""
    log = tmp_path / "log.jsonl"
    with pytest.raises(ValueError, match="approved"):
        record_decision(_run(log, kind="draft"), "reviewed", audit_path=log)


@pytest.mark.parametrize("bad", ["", "APPROVED", "maybe", "sent", None])
def test_unknown_decisions_are_refused_and_nothing_is_written(tmp_path: Path, bad: object) -> None:
    log = tmp_path / "log.jsonl"
    run = _run(log, kind="draft")
    before = log.read_text()
    with pytest.raises(ValueError):
        record_decision(run, bad, audit_path=log)  # type: ignore[arg-type]
    assert log.read_text() == before


def test_only_an_analysis_run_is_accepted(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        record_decision({"run_id": "x"}, "approved", audit_path=tmp_path / "log.jsonl")  # type: ignore[arg-type]


def test_a_person_may_decide_more_than_once_and_every_decision_is_kept(tmp_path: Path) -> None:
    """Edit first, approve later: two lines, in order. The last line for a run id is the final word."""
    log = tmp_path / "log.jsonl"
    run = _run(log, kind="draft")
    record_decision(run, "edited", audit_path=log)
    record_decision(run, "approved", audit_path=log)
    assert [l["human_decision"] for l in _lines(log)] == [None, "edited", "approved"]
    assert len({l["run_id"] for l in _lines(log)}) == 1


def test_decisions_for_different_runs_stay_separate(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    a, b = _run(log, kind="draft"), _run(log, kind="draft")
    record_decision(b, "discarded", audit_path=log)
    record_decision(a, "approved", audit_path=log)
    by_run = {l["run_id"]: l["human_decision"] for l in _lines(log) if l["human_decision"]}
    assert by_run == {b.run_id: "discarded", a.run_id: "approved"}


def test_no_note_or_draft_text_is_ever_logged(tmp_path: Path) -> None:
    """The decision line records what the person did, never the text of the note or the draft."""
    log = tmp_path / "log.jsonl"
    run = _run(log, kind="draft")
    assert run.draft and DRAFT_BODY in run.draft.body
    record_decision(run, "edited", audit_path=log)
    text = log.read_text()
    for fragment in [DRAFT_BODY, "Zebra", "Examplumab", "Stage IIIB", "C34.90", "Subject"]:
        assert fragment not in text


def test_the_default_path_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "env_log.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log))
    run = analyze_packet(ORDER, NOTE, llm=lambda p: ExtractionResult(fields=_fields(biomarker=True)))
    record_decision(run, "reviewed")
    assert [l["human_decision"] for l in _lines(log)] == [None, "reviewed"]
