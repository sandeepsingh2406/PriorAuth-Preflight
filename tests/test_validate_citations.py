"""Tests for validate_citations in src/tools.py.

Rule under test: a quote counts only if it is a verbatim substring of the note.
Anything else is "not found", whatever the model claimed.
"""
import datetime
import random
from pathlib import Path

from src.models import ExtractedField
from src.tools import validate_citations

NOTE = (
    "SYNTHETIC NOTE\n"
    "Diagnosis: Non-small cell lung cancer (ICD-10 C34.90).\n"
    "Staging: Stage IIIB (cT3 N2 M0).\n"
    "Labs: CBC drawn 2026-04-22. WBC 6.1, hemoglobin 12.4.\n"
    "Plan: Start Examplumab 200 mg IV every 21 days.\n"
)
ROOT = Path(__file__).parent.parent


def _field(
    quote: str | None,
    found: bool = True,
    value: str | None = "some value",
    name: str = "stage",
    date: datetime.date | None = None,
    conflicting: list[str] | None = None,
) -> ExtractedField:
    return ExtractedField(
        name=name,
        value=value,
        quote=quote,
        found=found,
        date=date,
        conflicting_quotes=conflicting or [],
    )


def _validate_one(field: ExtractedField, note: str = NOTE) -> ExtractedField:
    (result,) = validate_citations(note, [field])
    return result


def _assert_not_found(field: ExtractedField) -> None:
    assert field.found is False
    assert field.value is None
    assert field.quote is None
    assert field.date is None


def test_verbatim_quote_stays_found_with_its_value_and_date() -> None:
    """A quote copied exactly from the note is kept, along with value and lab date."""
    lab_date = datetime.date(2026, 4, 22)
    field = _field("CBC drawn 2026-04-22", name="recent_lab", value="CBC", date=lab_date)
    result = _validate_one(field)
    assert result.found is True
    assert result.quote == "CBC drawn 2026-04-22"
    assert result.value == "CBC"
    assert result.date == lab_date


def test_one_changed_character_is_not_found() -> None:
    """A single wrong character (IIIB vs IIIA) means the quote is not in the note."""
    _assert_not_found(_validate_one(_field("Stage IIIA (cT3 N2 M0)", date=datetime.date(2026, 1, 1))))


def test_different_case_is_not_found() -> None:
    """Matching is exact: no case folding."""
    _assert_not_found(_validate_one(_field("stage iiib")))


def test_different_whitespace_is_not_found() -> None:
    """No whitespace normalisation: a double space or a newline in place of a space fails."""
    _assert_not_found(_validate_one(_field("Stage  IIIB")))
    _assert_not_found(_validate_one(_field("Staging:\nStage IIIB")))


def test_empty_quote_is_not_found() -> None:
    """An empty string is a substring of everything, so it must be refused explicitly."""
    _assert_not_found(_validate_one(_field("")))


def test_whitespace_only_quote_is_not_found() -> None:
    """Spaces and newlines exist in every note but prove nothing."""
    _assert_not_found(_validate_one(_field(" ")))
    _assert_not_found(_validate_one(_field("\n")))


def test_missing_quote_is_not_found() -> None:
    """found=True with no quote at all is not evidence."""
    _assert_not_found(_validate_one(_field(None)))


def test_multi_line_quote_must_match_across_the_line_break() -> None:
    """A quote spanning lines is fine when it matches exactly, newline included."""
    quote = "Staging: Stage IIIB (cT3 N2 M0).\nLabs: CBC drawn"
    assert _validate_one(_field(quote)).found is True


def test_a_field_marked_not_found_is_never_upgraded() -> None:
    """Validation only downgrades: found=False stays False even if its quote is in the note."""
    result = _validate_one(_field("Stage IIIB", found=False))
    _assert_not_found(result)


def test_empty_note_finds_nothing() -> None:
    """With no note text, no quote can be verified."""
    _assert_not_found(_validate_one(_field("Stage IIIB"), note=""))


def test_conflicting_quotes_keep_only_verbatim_ones() -> None:
    """Each conflicting quote is checked too; invented, empty and blank ones are dropped."""
    field = _field(
        "Stage IIIB",
        conflicting=["Staging: Stage IIIB (cT3 N2 M0).", "Stage IV", "", "  "],
    )
    assert _validate_one(field).conflicting_quotes == ["Staging: Stage IIIB (cT3 N2 M0)."]


def test_verified_conflicting_quotes_survive_when_the_main_quote_fails() -> None:
    """The main quote can fail while a real contradicting quote is still worth reporting."""
    result = _validate_one(_field("Stage IX", conflicting=["Stage IIIB"]))
    _assert_not_found(result)
    assert result.conflicting_quotes == ["Stage IIIB"]


def test_inputs_are_not_modified_and_results_are_new_objects() -> None:
    """The function is pure: the caller's fields are untouched."""
    bad = _field("Stage IX", conflicting=["Stage IV", "Stage IIIB"])
    good = _field("Stage IIIB")
    fields = [bad, good]
    results = validate_citations(NOTE, fields)
    assert fields == [bad, good] and len(fields) == 2
    assert bad.found is True and bad.quote == "Stage IX"
    assert bad.conflicting_quotes == ["Stage IV", "Stage IIIB"]
    assert all(r is not f for r, f in zip(results, fields))


def test_order_and_length_are_preserved() -> None:
    """One output per input, in the same order, with names carried over."""
    fields = [
        _field("Stage IIIB", name="stage"),
        _field("nope", name="biomarker"),
        _field("Start Examplumab 200 mg IV every 21 days.", name="treatment_plan"),
    ]
    results = validate_citations(NOTE, fields)
    assert [r.name for r in results] == ["stage", "biomarker", "treatment_plan"]
    assert [r.found for r in results] == [True, False, True]
    assert validate_citations(NOTE, []) == []


def test_quote_copied_from_a_real_case_note_is_found() -> None:
    """Works on a real file from data/cases, not just the inline sample."""
    note = (ROOT / "data" / "cases" / "hand_01" / "note.txt").read_text()
    line = next(l for l in note.splitlines() if l.startswith("Staging:"))
    assert _validate_one(_field(line), note=note).found is True


def test_fuzz_found_means_verbatim_substring() -> None:
    """Seeded fuzz: after validation, found is True exactly when the quote is in the note."""
    rng = random.Random(1)
    for _ in range(300):
        start = rng.randrange(len(NOTE))
        quote = NOTE[start : start + rng.randint(1, 40)]
        if rng.random() < 0.5:  # corrupt one character
            i = rng.randrange(len(quote))
            quote = quote[:i] + rng.choice("xyz#") + quote[i + 1 :]
        result = _validate_one(_field(quote))
        assert result.found == (quote in NOTE and bool(quote.strip())), repr(quote)
        if result.found:
            assert result.quote in NOTE
