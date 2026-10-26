"""Input limits and run caps for the public demo. Pure functions: no model calls, no logging of input."""
import datetime
import os
import re
import threading
from collections.abc import Callable, Mapping

MAX_NOTE_CHARS = 2000
MAX_DRUG_CHARS = 80
MAX_DOSE_CHARS = 40
MAX_SCHEDULE_CHARS = 80
SESSION_RUN_CAP = 10
DEFAULT_DAILY_CAP = 200
DEMO_CONCURRENCY = 2

ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")  # ASCII digits only
DATE_MESSAGE = "Enter the order date as YYYY-MM-DD."


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def check_inputs(
    note: str,
    payer_id: str,
    order_date: str,
    drug: str,
    dose: str,
    schedule: str,
    payers: list[str],
) -> str | None:
    """Return a plain-words message for the first problem, or None if the input is acceptable.

    The note is judged first and by length alone, so a huge paste is refused at once.
    Lengths are counted in characters (an emoji is one), after trimming for the order fields.
    """
    note = _text(note)
    if len(note) > MAX_NOTE_CHARS:
        return f"The note is too long: {len(note):,} characters (limit {MAX_NOTE_CHARS:,})."
    if not note.strip():
        return "Paste or choose a note first."
    if payer_id not in payers:
        return "Choose a payer from the list."
    date_text = _text(order_date).strip()
    if not ISO_DATE.fullmatch(date_text):
        return DATE_MESSAGE
    try:
        datetime.date.fromisoformat(date_text)
    except ValueError:
        return DATE_MESSAGE
    for label, value, limit in [
        ("drug", drug, MAX_DRUG_CHARS),
        ("dose", dose, MAX_DOSE_CHARS),
        ("schedule", schedule, MAX_SCHEDULE_CHARS),
    ]:
        trimmed = _text(value).strip()
        if not trimmed:
            return f"Enter the {label} from the order."
        if len(trimmed) > limit:
            return f"The {label} is too long (limit {limit} characters)."
    return None


# --- settings from the environment ----------------------------------------------------------------


def demo_mode(env: Mapping[str, str] | None = None) -> bool:
    """DEMO_MODE turns the public-demo protections on. Off unless clearly switched on."""
    source = os.environ if env is None else env
    return source.get("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def daily_cap(env: Mapping[str, str] | None = None) -> int:
    """DAILY_RUN_CAP as a whole number of runs; anything unclear falls back to the default."""
    source = os.environ if env is None else env
    raw = source.get("DAILY_RUN_CAP", "").strip()
    return int(raw) if re.fullmatch(r"[0-9]+", raw) else DEFAULT_DAILY_CAP


# --- run caps -------------------------------------------------------------------------------------


def _utc_today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


class RunLimiter:
    """A per-session cap and a global daily cap on runs, kept in memory and safe across threads.

    The daily count starts again each UTC day and whenever the process restarts.
    """

    def __init__(
        self,
        daily_cap: int = DEFAULT_DAILY_CAP,
        session_cap: int = SESSION_RUN_CAP,
        today: Callable[[], datetime.date] | None = None,
    ) -> None:
        self.daily_cap = daily_cap
        self.session_cap = session_cap
        self._today = today or _utc_today
        self._lock = threading.Lock()
        self._day: datetime.date | None = None
        self._count = 0

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._count = today, 0

    @property
    def used_today(self) -> int:
        with self._lock:
            self._roll()
            return self._count

    def try_start(self, session_runs: int) -> tuple[bool, str | None]:
        """Reserve one run if both caps allow it. Returns (allowed, friendly message if not)."""
        with self._lock:
            self._roll()
            if session_runs >= self.session_cap:
                return False, (
                    f"This demo allows {self.session_cap} runs per session, and you have used them. "
                    "Thank you for trying it."
                )
            if self._count >= self.daily_cap:
                return False, (
                    f"The demo has reached its daily limit of {self.daily_cap} runs. Please try again tomorrow."
                )
            self._count += 1
            return True, None


# --- the live character counter -------------------------------------------------------------------


def counter_html(count: int) -> str:
    """The text under the note box. Red and worded ("too long") when over the limit."""
    over = count > MAX_NOTE_CHARS
    css = "char-counter char-over" if over else "char-counter"
    tail = " (too long)" if over else ""
    return f'<div class="{css}">{count} / {MAX_NOTE_CHARS} characters{tail}</div>'


# The same text built in the browser, so typing needs no server round trip. [...text] counts
# characters the way Python's len() does (an emoji is one), unlike text.length.
COUNTER_JS = (
    "(text) => {\n"
    '  const n = [...(text || "")].length;\n'
    f"  const over = n > {MAX_NOTE_CHARS};\n"
    "  return '<div class=\"char-counter' + (over ? ' char-over' : '') + '\">' + n + ' / '"
    f" + '{MAX_NOTE_CHARS} characters' + (over ? ' (too long)' : '') + '</div>';\n"
    "}"
)
