"""DEMO_MODE: run caps, queue limit, temp audit path, tracing off, and nothing private in logs."""
import json
import logging
import tempfile
from pathlib import Path

import pytest

from src.app import (
    CaseForm,
    _notice_html,
    build_app,
    load_case_into_form,
    make_limiter,
    resolve_audit_path,
    run_with_limits,
)
from src.guardrails import DEMO_CONCURRENCY, RunLimiter
from src.models import DraftText, ExtractionResult
from src.tools import PROJECT_ROOT


class CountingModel:
    """A fake extraction model that counts how often it is called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> ExtractionResult:
        self.calls += 1
        return ExtractionResult(fields=[])  # nothing found: a NOT_READY packet


def _outreach(prompt: str) -> DraftText:
    return DraftText(subject="Items needed", body="Please send the listed items.")


def _form(**changes: str) -> CaseForm:
    form = load_case_into_form("hand_01")
    return form.model_copy(update=changes) if changes else form


@pytest.fixture(autouse=True)
def _restore_tracing():
    """build_app(demo=True) switches tracing off globally; put it back after each test."""
    from agents.tracing import get_trace_provider

    provider = get_trace_provider()
    saved = (provider._manual_disabled, provider._disabled)
    yield
    provider._manual_disabled, provider._disabled = saved


def _go(form: CaseForm, usage: int, limiter: RunLimiter | None, model: CountingModel, tmp_path: Path):
    return run_with_limits(
        form, usage, limiter=limiter, llm=model, outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl"
    )


# --- the settings -------------------------------------------------------------------------------


def test_the_limiter_exists_only_in_demo_mode_and_reads_the_daily_cap() -> None:
    assert make_limiter(False, {"DAILY_RUN_CAP": "5"}) is None
    limiter = make_limiter(True, {"DAILY_RUN_CAP": "5"})
    assert limiter is not None and limiter.daily_cap == 5 and limiter.session_cap == 10
    assert make_limiter(True, {}).daily_cap == 200
    assert make_limiter(True, {"DAILY_RUN_CAP": "nonsense"}).daily_cap == 200


def test_in_demo_mode_the_audit_log_goes_to_a_temp_folder_never_the_project() -> None:
    path = resolve_audit_path(True, None)
    assert path is not None and Path(tempfile.gettempdir()).resolve() in path.resolve().parents
    assert PROJECT_ROOT.resolve() not in path.resolve().parents
    assert resolve_audit_path(True, None) != path  # each call gets its own fresh folder (not guessable)
    given = Path("/somewhere/else/audit.jsonl")
    assert resolve_audit_path(True, given) == given  # an explicit path always wins
    assert resolve_audit_path(False, None) is None  # locally: the normal environment-based default


def test_the_queue_allows_two_at_once_in_demo_mode_and_one_otherwise() -> None:
    assert build_app(demo=True, llm=CountingModel(), outreach_llm=_outreach)._queue.default_concurrency_limit == DEMO_CONCURRENCY == 2
    assert build_app(demo=False, llm=CountingModel(), outreach_llm=_outreach)._queue.default_concurrency_limit == 1


def test_tracing_is_switched_off_in_code_in_demo_mode() -> None:
    from agents.tracing import get_trace_provider

    build_app(demo=True, llm=CountingModel(), outreach_llm=_outreach)
    assert get_trace_provider()._manual_disabled is True


def test_local_runs_do_not_touch_the_tracing_setting() -> None:
    from agents.tracing import get_trace_provider

    before = get_trace_provider()._manual_disabled
    build_app(demo=False, llm=CountingModel(), outreach_llm=_outreach)
    assert get_trace_provider()._manual_disabled == before


# --- caps -----------------------------------------------------------------------------------------


def test_without_a_limiter_there_is_no_cap(tmp_path: Path) -> None:
    model, usage = CountingModel(), 0
    for _ in range(25):
        view, usage = _go(_form(), usage, None, model, tmp_path)
        assert view.error is None and view.notice is None
    assert model.calls == 25 and usage == 0


def test_a_session_gets_ten_runs_and_the_eleventh_is_blocked_before_any_model_call(tmp_path: Path) -> None:
    limiter, model, usage = RunLimiter(daily_cap=1000), CountingModel(), 0
    for i in range(10):
        view, usage = _go(_form(), usage, limiter, model, tmp_path)
        assert view.session is not None, i
    assert usage == 10 and model.calls == 10
    view, usage = _go(_form(), usage, limiter, model, tmp_path)
    assert view.session is None and view.error is None
    assert view.notice and "10 runs" in view.notice and "callout-warn" in _notice_html(view)
    assert model.calls == 10 and usage == 10  # nothing was called and nothing was counted


def test_the_daily_cap_is_shared_by_every_session(tmp_path: Path) -> None:
    limiter, model = RunLimiter(daily_cap=3), CountingModel()
    a = b = 0
    _, a = _go(_form(), a, limiter, model, tmp_path)
    _, b = _go(_form(), b, limiter, model, tmp_path)
    _, a = _go(_form(), a, limiter, model, tmp_path)
    view, b = _go(_form(), b, limiter, model, tmp_path)
    assert view.session is None and view.notice and "daily limit" in view.notice
    assert model.calls == 3 and (a, b) == (2, 1)


def test_invalid_input_never_uses_up_a_run(tmp_path: Path) -> None:
    limiter, model, usage = RunLimiter(daily_cap=5), CountingModel(), 0
    for bad in [{"order_date": "nope"}, {"note": "x" * 2001}, {"drug": "d" * 81}, {"payer_id": "payer_z"}, {"note": ""}] * 4:
        view, usage = _go(_form(**bad), usage, limiter, model, tmp_path)
        assert view.error and view.session is None
    assert usage == 0 and limiter.used_today == 0 and model.calls == 0
    view, usage = _go(_form(), usage, limiter, model, tmp_path)
    assert view.session is not None and usage == 1


def test_a_run_whose_model_call_fails_still_counts(tmp_path: Path) -> None:
    """Cost protection: a failing call may still have been billed, so it uses up a run."""

    def broken(prompt: str) -> ExtractionResult:
        raise RuntimeError("upstream error")

    limiter = RunLimiter(daily_cap=5)
    view, usage = run_with_limits(
        _form(), 0, limiter=limiter, llm=broken, outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl"
    )
    assert usage == 1 and limiter.used_today == 1
    assert view.session is not None  # the failure became a NEEDS_REVIEW result, not a crash


def test_a_cap_message_is_friendly_and_does_not_explain_how_to_dodge_it() -> None:
    limiter = RunLimiter(daily_cap=0)
    allowed, message = limiter.try_start(0)
    assert not allowed and "Please try again tomorrow" in message
    assert "reload" not in message.lower()


# --- the wired handler, end to end ---------------------------------------------------------------


def test_the_real_run_handler_counts_runs_in_session_state_and_blocks_the_eleventh(tmp_path: Path) -> None:
    model = CountingModel()
    demo = build_app(
        demo=True, limiter=RunLimiter(daily_cap=1000), llm=model, outreach_llm=_outreach, audit_path=tmp_path / "audit.jsonl"
    )
    on_run = next(f for f in demo.fns.values() if f.name == "on_run").fn
    form, usage = _form(), 0
    for _ in range(10):
        out = on_run(form.case_id, form.note, form.payer_id, form.order_date, form.drug, form.dose, form.schedule, usage)
        usage = out[-1]
    assert usage == 10 and model.calls == 10
    out = on_run(form.case_id, form.note, form.payer_id, form.order_date, form.drug, form.dose, form.schedule, usage)
    assert out[-1] == 10 and model.calls == 10
    assert "10 runs" in out[2]  # the friendly message is in the notice slot of the page


# --- nothing private in logs, output or the audit file --------------------------------------------


def test_note_text_order_fields_and_keys_never_reach_logs_output_or_the_audit_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]
) -> None:
    key = "not-a-real-key-" + "x" * 12  # a sentinel that must never appear anywhere; it does not look like a real key
    monkeypatch.setenv("OPENAI_API_KEY", key)
    secret_note = "SYNTHETIC NOTE\nDiagnosis: Zebra-stripe carcinoma of the quokka\nPlan: Start Quokkamab 77 mg"
    form = _form(note=secret_note, drug="Quokkamab", dose="77 mg", schedule="every 9 days")

    def failing(prompt: str) -> ExtractionResult:
        raise RuntimeError(f"upstream echoed the prompt: {prompt[:200]} and the key {key}")

    audit = tmp_path / "audit.jsonl"
    with caplog.at_level(logging.DEBUG):
        run_with_limits(form, 0, limiter=RunLimiter(), llm=failing, outreach_llm=_outreach, audit_path=audit)
        run_with_limits(form, 0, limiter=RunLimiter(), llm=CountingModel(), outreach_llm=_outreach, audit_path=audit)
    out, err = capfd.readouterr()
    everything = caplog.text + out + err + audit.read_text()
    for private in [key, "Zebra", "quokka", "Quokkamab", "77 mg", "every 9 days", secret_note]:
        assert private not in everything, private
    assert len(audit.read_text().splitlines()) == 2
    assert all("run_id" in json.loads(line) for line in audit.read_text().splitlines())


def test_a_crash_inside_validation_is_a_short_message_a_type_only_log_and_no_run_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from src.app import UNEXPECTED_MESSAGE

    def boom(form: CaseForm):
        raise ValueError("SECRET note text inside the exception")

    monkeypatch.setattr("src.app._to_order", boom)
    model = CountingModel()
    with caplog.at_level(logging.DEBUG):
        view, usage = _go(_form(), 0, RunLimiter(), model, tmp_path)
    assert view.error == UNEXPECTED_MESSAGE and view.session is None
    assert usage == 0 and model.calls == 0
    assert [r.getMessage() for r in caplog.records if r.name == "priorauth"] == ["run failed: ValueError"]
    assert "SECRET" not in caplog.text
