"""Tests for src/guardrails.py: input limits, run caps, environment settings and the counter."""
import datetime
import json
import shutil
import subprocess
import threading
import time

import pytest

from src.guardrails import (
    COUNTER_JS,
    DEFAULT_DAILY_CAP,
    MAX_DOSE_CHARS,
    MAX_DRUG_CHARS,
    MAX_NOTE_CHARS,
    MAX_SCHEDULE_CHARS,
    SESSION_RUN_CAP,
    RunLimiter,
    check_inputs,
    counter_html,
    daily_cap,
    demo_mode,
)

PAYERS = ["payer_a", "payer_b", "payer_c"]


def check(**changes: object) -> str | None:
    base = dict(
        note="Some note",
        payer_id="payer_a",
        order_date="2026-05-04",
        drug="Examplumab",
        dose="200 mg",
        schedule="every 21 days",
        payers=PAYERS,
    )
    base.update(changes)
    return check_inputs(**base)  # type: ignore[arg-type]


def test_the_limits_are_the_agreed_numbers() -> None:
    assert (MAX_NOTE_CHARS, MAX_DRUG_CHARS, MAX_DOSE_CHARS, MAX_SCHEDULE_CHARS) == (2000, 80, 40, 80)
    assert (SESSION_RUN_CAP, DEFAULT_DAILY_CAP) == (10, 200)


def test_valid_input_has_no_problem() -> None:
    assert check() is None


# --- the note -------------------------------------------------------------------------------------


def test_a_note_of_exactly_the_limit_is_accepted_and_one_more_is_not() -> None:
    assert check(note="x" * 2000) is None
    message = check(note="x" * 2001)
    assert message and "too long" in message and "2,001" in message and "2,000" in message


@pytest.mark.parametrize("char", ["é", "漢", "😀"])
def test_the_limit_counts_characters_not_bytes(char: str) -> None:
    """An accented letter, a CJK character and an emoji each count as one character."""
    assert check(note=char * 2000) is None
    assert check(note=char * 2001) is not None


def test_a_non_ascii_note_is_fine() -> None:
    assert check(note="Diagnóstico: carcinoma 肺癌 ✔ 😀\nStage IIIB") is None


def test_a_huge_note_is_rejected_at_once_without_doing_any_other_work() -> None:
    start = time.perf_counter()
    message = check(note="x" * 1_000_000, order_date="not a date", payer_id="nobody")
    assert time.perf_counter() - start < 0.5
    assert message and "too long" in message  # the note is judged first


@pytest.mark.parametrize("note", ["", " ", "\n\t  \n"])
def test_an_empty_or_blank_note_is_rejected(note: str) -> None:
    message = check(note=note)
    assert message and "note" in message.lower()


# --- the payer ------------------------------------------------------------------------------------


@pytest.mark.parametrize("payer", ["payer_z", "", None, "PAYER_A", "payer_a ", "payer_a; drop"])
def test_an_unknown_payer_is_rejected(payer: object) -> None:
    message = check(payer_id=payer)
    assert message and "payer" in message.lower()


# --- the order date -------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["2026-05-04", " 2026-05-04 ", "2024-02-29", "2026-12-31"])
def test_real_dates_in_the_agreed_format_are_accepted(value: str) -> None:
    assert check(order_date=value) is None


@pytest.mark.parametrize(
    "value",
    [
        "", " ", "05/04/2026", "2026-5-4", "2026-13-01", "2026-02-30", "2025-02-29", "0000-01-01",
        "20260504", "2026-W18-1", "2026-05-04T10:00", "2026-05-04 10:00", "２０２６-０５-０４", "2026/05/04", "May 4 2026", "2026-05-04; x",
    ],
)
def test_anything_but_a_real_yyyy_mm_dd_date_is_rejected(value: str) -> None:
    message = check(order_date=value)
    assert message and "YYYY-MM-DD" in message


# --- the order fields -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "limit"), [("drug", MAX_DRUG_CHARS), ("schedule", MAX_SCHEDULE_CHARS), ("dose", MAX_DOSE_CHARS)]
)
def test_each_order_field_stops_exactly_at_its_limit(field: str, limit: int) -> None:
    assert check(**{field: "x" * limit}) is None
    message = check(**{field: "x" * (limit + 1)})
    assert message and field in message and str(limit) in message and "too long" in message


@pytest.mark.parametrize("field", ["drug", "dose", "schedule"])
def test_empty_order_fields_are_rejected(field: str) -> None:
    for blank in ["", "   "]:
        message = check(**{field: blank})
        assert message and field in message


def test_spaces_around_a_field_do_not_count_toward_its_limit() -> None:
    assert check(drug="x" * 80 + "      ") is None


# --- run caps -------------------------------------------------------------------------------------


def make_limiter(daily: int = 200, days: list[datetime.date] | None = None) -> RunLimiter:
    days = days or [datetime.date(2026, 10, 25)]
    state = {"i": 0}

    def today() -> datetime.date:
        return days[min(state["i"], len(days) - 1)]

    limiter = RunLimiter(daily_cap=daily, today=today)
    limiter.next_day = lambda: state.__setitem__("i", state["i"] + 1)  # type: ignore[attr-defined]
    return limiter


def test_a_session_may_run_ten_times_and_the_eleventh_is_blocked() -> None:
    limiter = make_limiter()
    for runs_so_far in range(10):
        assert limiter.try_start(runs_so_far) == (True, None), runs_so_far
    allowed, message = limiter.try_start(10)
    assert allowed is False and message and "10 runs" in message
    assert limiter.used_today == 10


def test_blocked_attempts_do_not_use_up_the_daily_allowance() -> None:
    limiter = make_limiter()
    for _ in range(5):
        assert limiter.try_start(10)[0] is False
    assert limiter.used_today == 0


def test_the_daily_cap_blocks_with_a_friendly_message_and_stops_counting() -> None:
    limiter = make_limiter(daily=3)
    assert [limiter.try_start(0)[0] for _ in range(3)] == [True, True, True]
    allowed, message = limiter.try_start(0)
    assert allowed is False and message and "daily limit" in message and "3" in message
    assert limiter.try_start(0)[0] is False
    assert limiter.used_today == 3


def test_the_daily_count_starts_again_on_a_new_day() -> None:
    limiter = make_limiter(daily=2, days=[datetime.date(2026, 10, 25), datetime.date(2026, 10, 26)])
    assert limiter.try_start(0)[0] and limiter.try_start(0)[0]
    assert limiter.try_start(0)[0] is False
    limiter.next_day()  # type: ignore[attr-defined]
    assert limiter.try_start(0)[0] is True
    assert limiter.used_today == 1


def test_a_cap_of_zero_blocks_everything() -> None:
    assert make_limiter(daily=0).try_start(0)[0] is False


def test_many_threads_cannot_get_past_the_daily_cap() -> None:
    limiter = RunLimiter(daily_cap=50)
    results: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        allowed = limiter.try_start(0)[0]
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=attempt) for _ in range(300)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(results) == 50 and limiter.used_today == 50


def test_the_session_cap_message_does_not_say_how_to_get_around_it() -> None:
    message = make_limiter().try_start(10)[1].lower()
    assert "reload" not in message and "refresh" not in message and "new session" not in message


# --- environment settings -------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on", " 1 "])
def test_demo_mode_turns_on(value: str) -> None:
    assert demo_mode({"DEMO_MODE": value}) is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "2", "maybe"])
def test_demo_mode_is_off_by_default_and_for_anything_unclear(value: str | None) -> None:
    env = {} if value is None else {"DEMO_MODE": value}
    assert demo_mode(env) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 200), ("50", 50), (" 75 ", 75), ("0", 0), ("-5", 200), ("abc", 200), ("1.5", 200), ("", 200), ("１２", 200)],
)
def test_the_daily_cap_comes_from_the_environment_with_a_safe_fallback(value: str | None, expected: int) -> None:
    env = {} if value is None else {"DAILY_RUN_CAP": value}
    assert daily_cap(env) == expected


# --- the character counter ------------------------------------------------------------------------


def test_the_counter_shows_the_count_and_the_limit() -> None:
    assert counter_html(0) == '<div class="char-counter">0 / 2000 characters</div>'
    assert counter_html(2000) == '<div class="char-counter">2000 / 2000 characters</div>'


def test_the_counter_warns_in_words_when_over_the_limit() -> None:
    html = counter_html(2001)
    assert "char-over" in html and "too long" in html and "2001 / 2000" in html


def test_the_browser_script_counts_characters_like_python_and_uses_the_same_limit() -> None:
    """[...text] counts code points, so an emoji is 1, matching len() in Python."""
    assert "[...(text || \"\")]" in COUNTER_JS and "2000" in COUNTER_JS
    assert "char-over" in COUNTER_JS and "too long" in COUNTER_JS and "characters" in COUNTER_JS
    assert COUNTER_JS.count("2000") == 2  # the compare and the label, both from MAX_NOTE_CHARS


def _js_runner():
    """A way to run JavaScript here: node if it actually works, else macOS's built-in engine, else None."""
    node = shutil.which("node")
    if node:
        try:
            subprocess.run([node, "--version"], capture_output=True, check=True, timeout=10)
            return lambda expr: ["node", "-e", f"console.log({expr})"]
        except (subprocess.SubprocessError, OSError):
            pass  # installed but broken (for example a missing library)
    if shutil.which("osascript"):
        return lambda expr: ["osascript", "-l", "JavaScript", "-e", expr]
    return None


def test_the_browser_script_really_gives_the_same_html_as_python() -> None:
    """Run the script itself: ASCII, accents, CJK, emoji (one character each), empty, and over the limit."""
    runner = _js_runner()
    if runner is None:
        pytest.skip("no JavaScript engine available")
    samples = ["", "hello", "Diagnóstico ñ", "肺癌 stage III", "😀" * 7, "a😀b", "x" * 2000, "x" * 2001, "😀" * 2001]
    expr = f"JSON.stringify({json.dumps(samples)}.map(t => ({COUNTER_JS})(t)))"
    result = subprocess.run(runner(expr), capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(result.stdout) == [counter_html(len(s)) for s in samples]


class _SlowCapLimiter(RunLimiter):
    """Reading the cap takes a moment, so unlocked threads would interleave between check and count."""

    @property
    def daily_cap(self) -> int:  # type: ignore[override]
        time.sleep(0.001)
        return self._cap

    @daily_cap.setter
    def daily_cap(self, value: int) -> None:
        self._cap = value


def test_the_daily_cap_holds_even_when_threads_interleave_inside_the_check() -> None:
    """The earlier thread test can pass by luck; this one only passes if check and count share a lock."""
    limiter = _SlowCapLimiter(daily_cap=20)
    results: list[bool] = []
    results_lock = threading.Lock()

    def attempt() -> None:
        allowed = limiter.try_start(0)[0]
        with results_lock:
            results.append(allowed)

    threads = [threading.Thread(target=attempt) for _ in range(120)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(results) == 20 and limiter.used_today == 20
