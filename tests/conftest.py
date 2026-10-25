"""Shared test setup: unit tests must never reach the network, the real model, or the real audit log."""
from pathlib import Path

import pytest


class NetworkCallInTest(RuntimeError):
    """Raised if a unit test reaches the real Agents SDK runner."""


def _refuse(*args: object, **kwargs: object) -> None:
    raise NetworkCallInTest("A unit test tried to call the real model. Inject a fake llm instead.")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the API key and make every SDK run entry point raise."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import agents

    monkeypatch.setattr(agents.Runner, "run_sync", _refuse)
    monkeypatch.setattr(agents.Runner, "run", _refuse)
    monkeypatch.setattr(agents.Runner, "run_streamed", _refuse)


@pytest.fixture(autouse=True)
def _isolated_audit_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Any audit write that does not pass a path lands in this test's temp folder."""
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit_from_test.jsonl"))
