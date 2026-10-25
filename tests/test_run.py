"""Tests for the command line runner (LLM always mocked, .env never loaded)."""
import datetime

import pytest

from src.agents import analyze_packet
from src.models import ExtractedField, ExtractionResult, Order
from src.run import format_report, load_case, main

NOTE = (
    "SYNTHETIC NOTE\n"
    "Staging: Stage II (cT2 N0 M0).\n"
    "Assessment: stage IV disease.\n"
    "Labs: CBC drawn 2026-04-22.\n"
    "Plan: Start Examplumab 200 mg IV every 21 days.\n"
)
ORDER = Order(
    case_id="t_001",
    payer_id="payer_a",
    order_date=datetime.date(2026, 5, 4),
    drug="Examplumab",
    dose="200 mg",
    schedule="every 21 days",
)


def _f(name: str, quote: str | None, **kw: object) -> ExtractedField:
    return ExtractedField(name=name, value=quote, quote=quote, found=quote is not None, **kw)


def _llm(*fields: ExtractedField):
    return lambda prompt: ExtractionResult(fields=list(fields))


def test_report_shows_status_each_gap_and_its_evidence() -> None:
    """Conflicting stage: both quotes shown; missing items say no evidence; reason printed."""
    run = analyze_packet(
        ORDER,
        NOTE,
        llm=_llm(
            _f("diagnosis", None),
            _f("stage", "Staging: Stage II (cT2 N0 M0).", conflicting_quotes=["stage IV disease"]),
            _f("biomarker", None),
            _f("recent_lab", "Labs: CBC drawn 2026-04-22.", date=datetime.date(2026, 4, 22)),
            _f("treatment_plan", "Plan: Start Examplumab 200 mg IV every 21 days."),
        ),
    )
    text = format_report(run)
    assert "NEEDS_REVIEW" in text
    assert "stage [conflicting]" in text
    assert '"Staging: Stage II (cT2 N0 M0)."' in text
    assert '"stage IV disease"' in text
    assert "biomarker [missing]" in text
    assert "no evidence found in the note" in text
    assert "Escalation reason:" in text


def test_ready_report_says_no_gaps_and_has_no_escalation_line() -> None:
    run = analyze_packet(
        ORDER,
        NOTE,
        llm=_llm(
            _f("diagnosis", "SYNTHETIC NOTE"),
            _f("stage", "Staging: Stage II (cT2 N0 M0)."),
            _f("biomarker", "Assessment: stage IV disease."),
            _f("recent_lab", "Labs: CBC drawn 2026-04-22.", date=datetime.date(2026, 4, 22)),
            _f("treatment_plan", "Plan: Start Examplumab 200 mg IV every 21 days."),
        ),
    )
    text = format_report(run)
    assert "READY" in text and "NOT_READY" not in text
    assert "Gaps: none" in text
    assert "Escalation reason" not in text


def test_stale_lab_evidence_shows_the_quote() -> None:
    run = analyze_packet(
        ORDER,
        NOTE,
        llm=_llm(
            _f("diagnosis", "SYNTHETIC NOTE"),
            _f("stage", "Staging: Stage II (cT2 N0 M0)."),
            _f("biomarker", "Assessment: stage IV disease."),
            _f("recent_lab", "Labs: CBC drawn 2026-04-22.", date=datetime.date(2026, 1, 2)),
            _f("treatment_plan", "Plan: Start Examplumab 200 mg IV every 21 days."),
        ),
    )
    text = format_report(run)
    assert "recent_lab [stale]" in text
    assert '"Labs: CBC drawn 2026-04-22."' in text


def test_load_case_reads_order_and_note() -> None:
    order, note = load_case("hand_01")
    assert order.case_id == "hand_01"
    assert note.startswith("SYNTHETIC NOTE")


@pytest.mark.parametrize("bad", ["../hand_01", "hand_01/../../x", "", "hand 01", "/etc"])
def test_case_ids_outside_the_pattern_are_refused(bad: str) -> None:
    """A case id can only name a folder inside data/cases."""
    with pytest.raises(ValueError):
        load_case(bad)


def test_unknown_case_is_a_clear_error() -> None:
    with pytest.raises(FileNotFoundError, match="nope_99"):
        load_case("nope_99")


def test_main_prints_the_report(capsys: pytest.CaptureFixture[str]) -> None:
    """End to end through main with a fake model: prints the case and its status, exit 0."""
    code = main(["--case", "hand_01"], llm=_llm(), env_file=None)
    out = capsys.readouterr().out
    assert code == 0
    assert "hand_01" in out
    assert "NOT_READY" in out  # the fake model found nothing, so every item is missing


def test_main_without_a_model_name_explains_the_problem(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No MODEL_DEV and no fake model: a clear message and exit code 2, no model call."""
    monkeypatch.delenv("MODEL_DEV", raising=False)
    monkeypatch.delenv("USE_DEMO_MODEL", raising=False)
    code = main(["--case", "hand_01"], env_file=None)
    assert code == 2
    assert "MODEL_DEV" in capsys.readouterr().err


def test_main_with_a_bad_case_id_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--case", "../hand_01"], llm=_llm(), env_file=None)
    assert code == 2
    assert "case" in capsys.readouterr().err.lower()


def test_report_shows_unreadable_passages_for_the_reviewer() -> None:
    """A person has to read the garbled text themselves, so the report prints it."""
    bad = "EG?R ex@n 1# [unreadable]"
    note = NOTE + bad + "\n"
    run = analyze_packet(
        ORDER,
        note,
        llm=lambda prompt: ExtractionResult(
            fields=[
                _f("diagnosis", "SYNTHETIC NOTE"),
                _f("stage", "Staging: Stage II (cT2 N0 M0)."),
                _f("biomarker", None),
                _f("recent_lab", "Labs: CBC drawn 2026-04-22.", date=datetime.date(2026, 4, 22)),
                _f("treatment_plan", "Plan: Start Examplumab 200 mg IV every 21 days."),
            ],
            unreadable_passages=[bad],
        ),
    )
    text = format_report(run)
    assert "NEEDS_REVIEW" in text
    assert f'Unreadable text: "{bad}"' in text
    assert "Escalation reason:" in text
