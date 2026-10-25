"""Tests for eval/run_eval.py (LLM always mocked, temp folders only)."""
import json
import shutil
from pathlib import Path

import pytest

from eval.run_eval import EVAL_AUDIT_PATH, case_ids_for, main, run_once
from src.models import ExtractionResult
from src.tools import PROJECT_ROOT

ROOT = Path(__file__).parent.parent
CASES = ["hand_01", "hand_07", "gen_001"]


def _nothing_found(prompt: str) -> ExtractionResult:
    """A fake model that finds nothing: every required item comes back missing."""
    return ExtractionResult(fields=[])


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A small copy of the real data: three cases and their labels, in a temp folder."""
    # Even if the code under test ignored audit_path, it must not reach the real log.
    monkeypatch.setattr("eval.run_eval.EVAL_AUDIT_PATH", tmp_path / "never_the_real_log.jsonl")
    cases = tmp_path / "cases"
    for case_id in CASES:
        shutil.copytree(ROOT / "data" / "cases" / case_id, cases / case_id)
    all_labels = json.loads((ROOT / "data" / "labels.json").read_text())
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({k: all_labels[k] for k in CASES}, indent=2) + "\n")
    return {
        "cases_dir": cases,
        "labels_path": labels,
        "audit_path": tmp_path / "eval_audit.jsonl",
        "out": tmp_path / "out" / "report.md",
    }


def _main(ws: dict, *args: str, llm=_nothing_found) -> int:
    argv = ["--out", str(ws["out"]), *args]
    return main(
        argv,
        llm=llm,
        env_file=None,
        labels_path=ws["labels_path"],
        cases_dir=ws["cases_dir"],
        audit_path=ws["audit_path"],
    )


def test_cases_are_ordered_hand_first_then_generated(workspace: dict) -> None:
    labels = json.loads(workspace["labels_path"].read_text())
    assert case_ids_for(labels, "all") == ["hand_01", "hand_07", "gen_001"]
    assert case_ids_for(labels, "hand") == ["hand_01", "hand_07"]
    assert case_ids_for(labels, "generated") == ["gen_001"]


def test_run_once_returns_one_result_per_case_in_order(workspace: dict) -> None:
    labels = json.loads(workspace["labels_path"].read_text())
    results = run_once(
        ["hand_01", "gen_001"],
        labels,
        llm=_nothing_found,
        audit_path=workspace["audit_path"],
        cases_dir=workspace["cases_dir"],
    )
    assert [r.case_id for r in results] == ["hand_01", "gen_001"]
    assert results[0].expected_status == "READY" and results[0].actual_status == "NOT_READY"


def test_main_writes_markdown_and_json_for_every_run(workspace: dict, capsys: pytest.CaptureFixture[str]) -> None:
    assert _main(workspace, "--runs", "2") == 0
    data = json.loads(workspace["out"].with_suffix(".json").read_text())
    assert len(data["runs"]) == 2
    assert [c["case_id"] for c in data["runs"][0]["cases"]] == ["hand_01", "hand_07", "gen_001"]
    text = workspace["out"].read_text()
    assert "# Evaluation results" in text and "Hand-labeled cases" in text
    out = capsys.readouterr().out
    assert "run 1/2" in out and "run 2/2" in out and "gen_001" in out


def test_evaluation_audit_lines_go_to_their_own_file(workspace: dict) -> None:
    """One line per case per run, in the evaluation log; the normal run log is untouched."""
    _main(workspace, "--runs", "2")
    lines = workspace["audit_path"].read_text().splitlines()
    assert len(lines) == 2 * len(CASES)
    assert EVAL_AUDIT_PATH == PROJECT_ROOT / "audit" / "eval_runs.jsonl"


def test_group_option_limits_the_cases(workspace: dict) -> None:
    _main(workspace, "--runs", "1", "--group", "hand")
    data = json.loads(workspace["out"].with_suffix(".json").read_text())
    assert [c["case_id"] for c in data["runs"][0]["cases"]] == ["hand_01", "hand_07"]


def test_labels_are_never_modified(workspace: dict) -> None:
    before = workspace["labels_path"].read_bytes()
    _main(workspace, "--runs", "1")
    assert workspace["labels_path"].read_bytes() == before


def test_a_change_to_the_labels_during_the_run_is_an_error(workspace: dict, capsys: pytest.CaptureFixture[str]) -> None:
    def tampering(prompt: str) -> ExtractionResult:
        workspace["labels_path"].write_text(workspace["labels_path"].read_text() + "\n")
        return ExtractionResult(fields=[])

    assert _main(workspace, "--runs", "1", llm=tampering) == 3
    assert "labels" in capsys.readouterr().err.lower()


def test_render_rebuilds_the_report_from_saved_json_without_calling_the_model(workspace: dict) -> None:
    _main(workspace, "--runs", "1")
    saved = workspace["out"].with_suffix(".json")
    new_out = workspace["out"].with_name("again.md")

    def must_not_run(prompt: str) -> ExtractionResult:
        raise AssertionError("the model was called during --render")

    code = main(
        ["--render", str(saved), "--out", str(new_out)],
        llm=must_not_run,
        env_file=None,
        labels_path=workspace["labels_path"],
        cases_dir=workspace["cases_dir"],
        audit_path=workspace["audit_path"],
    )
    assert code == 0
    assert new_out.read_text() == workspace["out"].read_text()


def test_baseline_and_changes_appear_in_the_final_report(workspace: dict, tmp_path: Path) -> None:
    _main(workspace, "--runs", "1")
    baseline = tmp_path / "baseline.json"
    shutil.copy(workspace["out"].with_suffix(".json"), baseline)
    changes = tmp_path / "changes.md"
    changes.write_text("Added a rule for unreadable text.\n")
    final = {**workspace, "out": tmp_path / "final.md"}
    _main(final, "--runs", "1", "--baseline", str(baseline), "--changes", str(changes))
    text = final["out"].read_text()
    assert "Baseline comparison" in text
    assert "Added a rule for unreadable text." in text


def test_missing_model_name_without_a_fake_is_a_clear_error(
    workspace: dict, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MODEL_DEV", raising=False)
    monkeypatch.delenv("USE_DEMO_MODEL", raising=False)
    code = main(
        ["--out", str(workspace["out"])],
        env_file=None,
        labels_path=workspace["labels_path"],
        cases_dir=workspace["cases_dir"],
        audit_path=workspace["audit_path"],
    )
    assert code == 2
    assert "MODEL_DEV" in capsys.readouterr().err
