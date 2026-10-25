"""Tests for write_audit_log, audit_log_path and escalate_to_human in src/tools.py."""
import datetime
import json
from pathlib import Path

import pytest

from src.models import AuditEntry, ExtractedField, Gap, GapReport
from src.tools import PROJECT_ROOT, audit_log_path, escalate_to_human, write_audit_log

ROOT = Path(__file__).parent.parent


def _entry(case_id: str = "gen_001", **changes: object) -> AuditEntry:
    base = AuditEntry(
        timestamp=datetime.datetime(2026, 10, 25, 12, 30, 5),
        case_id=case_id,
        tools_called=["lookup_payer_checklist", "validate_citations", "check_packet"],
        status="NOT_READY",
    )
    return base.model_copy(update=changes)


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


# --- write_audit_log --------------------------------------------------------


def test_writes_one_json_line_that_round_trips(tmp_path: Path) -> None:
    """One run is one JSON line, and it loads back into the same AuditEntry."""
    log = tmp_path / "runs.jsonl"
    entry = _entry(human_decision="approved")
    write_audit_log(entry, path=log)
    (line,) = _lines(log)
    assert AuditEntry.model_validate_json(line) == entry


def test_appends_and_never_rewrites_earlier_lines(tmp_path: Path) -> None:
    """Two writes make two lines, and the first line is untouched by the second."""
    log = tmp_path / "runs.jsonl"
    write_audit_log(_entry("gen_001"), path=log)
    first_line = _lines(log)[0]
    write_audit_log(_entry("gen_002"), path=log)
    lines = _lines(log)
    assert len(lines) == 2
    assert lines[0] == first_line
    assert json.loads(lines[1])["case_id"] == "gen_002"


def test_existing_log_content_is_preserved(tmp_path: Path) -> None:
    """Appending to a log that already has lines keeps them exactly as they were."""
    log = tmp_path / "runs.jsonl"
    log.write_text('{"earlier": "line"}\n')
    write_audit_log(_entry(), path=log)
    assert _lines(log)[0] == '{"earlier": "line"}'
    assert len(_lines(log)) == 2


def test_creates_missing_parent_folders(tmp_path: Path) -> None:
    """The audit folder does not have to exist beforehand."""
    log = tmp_path / "audit" / "nested" / "runs.jsonl"
    write_audit_log(_entry(), path=log)
    assert len(_lines(log)) == 1


def test_a_newline_inside_a_value_still_makes_one_line(tmp_path: Path) -> None:
    """Values are JSON-escaped, so one entry can never span or forge extra lines."""
    log = tmp_path / "runs.jsonl"
    write_audit_log(_entry(case_id='gen_001\n{"forged": true}'), path=log)
    (line,) = _lines(log)
    assert json.loads(line)["case_id"] == 'gen_001\n{"forged": true}'


def test_a_line_holds_only_the_audit_fields(tmp_path: Path) -> None:
    """The log has the five SPEC fields and nothing else, so there is nowhere for note text."""
    log = tmp_path / "runs.jsonl"
    write_audit_log(_entry(), path=log)
    assert set(json.loads(_lines(log)[0])) == {
        "timestamp",
        "case_id",
        "tools_called",
        "status",
        "human_decision",
    }


def test_human_decision_is_empty_until_a_person_decides(tmp_path: Path) -> None:
    """Before approval the decision is null; afterwards it records what the person did."""
    log = tmp_path / "runs.jsonl"
    write_audit_log(_entry(), path=log)
    write_audit_log(_entry(human_decision="discarded"), path=log)
    first, second = (json.loads(line) for line in _lines(log))
    assert first["human_decision"] is None
    assert second["human_decision"] == "discarded"


@pytest.mark.parametrize("not_an_entry", [{"case_id": "x", "note": "full note text"}, "a string", None])
def test_only_an_audit_entry_is_accepted(tmp_path: Path, not_an_entry: object) -> None:
    """A dict or string could smuggle in note text, so anything but AuditEntry is refused."""
    log = tmp_path / "runs.jsonl"
    with pytest.raises(TypeError):
        write_audit_log(not_an_entry, path=log)  # type: ignore[arg-type]
    assert not log.exists()


def test_a_write_failure_is_raised_not_swallowed(tmp_path: Path) -> None:
    """If the log cannot be written, the caller must know; a silent gap would break the audit."""
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    with pytest.raises(OSError):
        write_audit_log(_entry(), path=blocker / "runs.jsonl")


def test_default_write_uses_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no path given, AUDIT_LOG_PATH decides where the line goes."""
    log = tmp_path / "from_env.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log))
    # If the variable were ever ignored, the default must land in tmp, not the real audit/.
    monkeypatch.setattr("src.tools.PROJECT_ROOT", tmp_path / "fake_root")
    write_audit_log(_entry())
    assert len(_lines(log)) == 1
    assert not (tmp_path / "fake_root").exists()


# --- where the log goes -----------------------------------------------------


def test_explicit_path_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "env.jsonl"))
    explicit = tmp_path / "explicit.jsonl"
    assert audit_log_path(explicit) == explicit


def test_environment_variable_wins_over_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "env.jsonl"))
    assert audit_log_path() == tmp_path / "env.jsonl"


@pytest.mark.parametrize("value", [None, ""])
def test_default_is_audit_runs_jsonl_under_the_project(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """Unset or empty means audit/runs.jsonl at the project root, whatever the cwd."""
    if value is None:
        monkeypatch.delenv("AUDIT_LOG_PATH", raising=False)
    else:
        monkeypatch.setenv("AUDIT_LOG_PATH", value)
    assert audit_log_path() == PROJECT_ROOT / "audit" / "runs.jsonl"
    assert PROJECT_ROOT == ROOT


def test_relative_environment_path_is_anchored_at_the_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """env.example uses a relative path; it must not depend on where Python was started."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDIT_LOG_PATH", "audit/runs.jsonl")
    assert audit_log_path() == PROJECT_ROOT / "audit" / "runs.jsonl"


def test_audit_folder_is_git_ignored() -> None:
    """The log is never committed: audit/ must stay in .gitignore."""
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert "audit/" in lines


# --- escalate_to_human ------------------------------------------------------


def _report(status: str = "NOT_READY", reason: str | None = None) -> GapReport:
    return GapReport(
        case_id="gen_001",
        payer_id="payer_a",
        status=status,  # type: ignore[arg-type]
        fields=[ExtractedField(name="stage", value="II", quote="Stage II", found=True)],
        gaps=[Gap(item="recent_lab", reason="stale", detail="old")],
        escalation_reason=reason,
    )


@pytest.mark.parametrize("status", ["READY", "NOT_READY", "NEEDS_REVIEW"])
def test_escalation_sets_needs_review_and_the_reason(status: str) -> None:
    """Whatever the status was, an escalation ends as NEEDS_REVIEW with the stated reason."""
    result = escalate_to_human(_report(status), "Biomarker section is unreadable")
    assert result.status == "NEEDS_REVIEW"
    assert result.escalation_reason == "Biomarker section is unreadable"


def test_escalation_keeps_the_evidence_and_leaves_the_input_alone() -> None:
    """Fields and gaps carry over for the reviewer; the original report is not modified."""
    original = _report()
    result = escalate_to_human(original, "Conflicting dose")
    assert result is not original
    assert result.fields == original.fields
    assert result.gaps == original.gaps
    assert (result.case_id, result.payer_id) == ("gen_001", "payer_a")
    assert original.status == "NOT_READY" and original.escalation_reason is None


@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
def test_a_blank_reason_is_refused(reason: str) -> None:
    """The point of escalating is to tell a person why, so a blank reason is an error."""
    with pytest.raises(ValueError, match="reason"):
        escalate_to_human(_report(), reason)


def test_reason_is_stripped() -> None:
    assert escalate_to_human(_report(), "  needs a look \n").escalation_reason == "needs a look"


def test_a_second_escalation_keeps_the_first_reason() -> None:
    """Nothing a person might need is overwritten: later reasons are appended."""
    once = escalate_to_human(_report(), "Stage conflicts")
    twice = escalate_to_human(once, "Lab date unreadable")
    assert twice.escalation_reason == "Stage conflicts; Lab date unreadable"
    assert twice.status == "NEEDS_REVIEW"


def test_escalation_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It only returns a new report: no files are written and no audit line is added."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "runs.jsonl"))
    escalate_to_human(_report(), "reason")
    assert list(tmp_path.iterdir()) == []
