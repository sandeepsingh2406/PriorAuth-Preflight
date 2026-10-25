"""The evaluation and the gate measure extraction only: they must never trigger Outreach drafts."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("path", ["eval/run_eval.py", "scripts/gate.py"])
def test_every_analyze_packet_call_turns_drafting_off(path: str) -> None:
    tree = ast.parse((ROOT / path).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "analyze_packet"
    ]
    assert calls, f"{path} no longer calls analyze_packet; update this test"
    for call in calls:
        draft = [k for k in call.keywords if k.arg == "draft"]
        assert draft and isinstance(draft[0].value, ast.Constant) and draft[0].value.value is False, path
