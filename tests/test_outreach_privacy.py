"""Hard rule 6: the Outreach agent never sees the clinical note or quotes from it."""
import datetime
from pathlib import Path

import pytest

from src.agents import analyze_packet
from src.models import DraftText, ExtractedField, ExtractionResult, Order
from src.run import load_case

WINDOW = 20  # a run of this many note characters appearing in the prompt counts as a leak

NOTE = (
    "SYNTHETIC NOTE\n"
    "Diagnosis: Zebra-stripe carcinoma of the quokka (ICD-10 C99.99).\n"
    "Staging: Stage Q9 per the purple-elephant protocol.\n"
    "Labs: CBC drawn 2026-04-22 at Marigold Annex.\n"
    "Plan: Start Examplumab 200 mg IV every 21 days.\n"
)


def _runs(text: str) -> set[str]:
    """Every WINDOW-character stretch of the text that is not just whitespace."""
    return {text[i : i + WINDOW] for i in range(len(text) - WINDOW + 1) if text[i : i + WINDOW].strip()}


class Recorder:
    """A fake Outreach model that records every prompt it is given."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> DraftText:
        self.prompts.append(prompt)
        return DraftText(subject="s", body="b")


def _f(name: str, quote: str | None, **kw: object) -> ExtractedField:
    return ExtractedField(name=name, value=quote, quote=quote, found=quote is not None, **kw)


def test_the_test_would_notice_a_leak() -> None:
    """Self-check: the run-of-characters comparison does flag a prompt that quotes the note."""
    leaky_prompt = "Please confirm: Staging: Stage Q9 per the purple-elephant protocol."
    assert _runs(NOTE) & _runs(leaky_prompt)


def test_no_quote_value_or_run_of_the_note_reaches_the_outreach_prompt(tmp_path: Path) -> None:
    """A note full of distinctive text, with one missing item: the Outreach prompt has none of it."""
    order = Order(
        case_id="t_001",
        payer_id="payer_a",
        order_date=datetime.date(2026, 5, 4),
        drug="Examplumab",
        dose="200 mg",
        schedule="every 21 days",
    )
    fields = [
        _f("diagnosis", "Diagnosis: Zebra-stripe carcinoma of the quokka (ICD-10 C99.99)."),
        _f("stage", "Staging: Stage Q9 per the purple-elephant protocol."),
        _f("biomarker", None),  # the gap Outreach will be asked about
        _f("recent_lab", "Labs: CBC drawn 2026-04-22 at Marigold Annex.", date=datetime.date(2026, 4, 22)),
        _f("treatment_plan", "Plan: Start Examplumab 200 mg IV every 21 days."),
    ]
    outreach = Recorder()
    run = analyze_packet(
        order,
        NOTE,
        llm=lambda prompt: ExtractionResult(fields=fields),
        audit_path=tmp_path / "log.jsonl",
        draft=True,
        outreach_llm=outreach,
    )
    assert run.report.status == "NOT_READY"
    (prompt,) = outreach.prompts
    for secret in ["Zebra", "quokka", "purple-elephant", "Marigold", "Q9", "Examplumab", "C99.99"]:
        assert secret not in prompt
    assert not (_runs(NOTE) & _runs(prompt))


@pytest.mark.parametrize("case_id", ["hand_03", "hand_07", "hand_10"])
def test_real_not_ready_cases_leak_nothing_to_outreach(case_id: str, tmp_path: Path) -> None:
    """On real hand-labeled notes, with the fake extractor quoting the lines it can find."""
    order, note = load_case(case_id)
    lines = [line for line in note.splitlines() if line.strip()]
    wanted = [("diagnosis", "Diagnosis:"), ("prior_therapies", "Prior therapy:"), ("treatment_plan", "Plan:")]
    fields = [_f(name, next((l for l in lines if l.startswith(prefix)), None)) for name, prefix in wanted]
    outreach = Recorder()
    run = analyze_packet(
        order,
        note,
        llm=lambda prompt: ExtractionResult(fields=fields),
        audit_path=tmp_path / "log.jsonl",
        draft=True,
        outreach_llm=outreach,
    )
    assert run.report.status == "NOT_READY"
    (prompt,) = outreach.prompts
    assert not (_runs(note) & _runs(prompt))
    assert all(f.quote not in prompt for f in fields if f.quote)
