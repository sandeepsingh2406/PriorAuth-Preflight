"""Shared test setup: unit tests must never reach the network, the real model, or the real audit log."""
from pathlib import Path

import pytest


class NetworkCallInTest(RuntimeError):
    """Raised if a unit test reaches the real Agents SDK runner."""


@pytest.fixture
def network_attempts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Block every SDK run entry point and record each attempt.

    A test that expects a blocked call must clear this list. The autouse fixture below
    fails any test that leaves an attempt behind, so a swallowed error cannot hide one.
    """
    import agents

    attempts: list[str] = []

    def refuse(*args: object, **kwargs: object) -> None:
        attempts.append("blocked")
        raise NetworkCallInTest("A unit test tried to call the real model. Inject a fake llm instead.")

    monkeypatch.setattr(agents.Runner, "run_sync", refuse)
    monkeypatch.setattr(agents.Runner, "run", refuse)
    monkeypatch.setattr(agents.Runner, "run_streamed", refuse)
    return attempts


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch, network_attempts: list[str]):
    """Remove the API key, and fail the test if anything tried to reach the real model."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield
    assert not network_attempts, "A test tried to call the real model; the call was blocked but not expected"


@pytest.fixture(autouse=True)
def _isolated_audit_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Any audit write that does not pass a path lands in this test's temp folder."""
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit_from_test.jsonl"))
