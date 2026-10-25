"""Tests for validate_passages in src/tools.py: unreadable text the model reports must be real."""
from src.tools import validate_passages

NOTE = "SYNTHETIC NOTE\nBIOMARKERS\n[scan illegible] EG?R ex@n 1# d%l%tion\nPLAN\nStart X.\n"


def test_verbatim_passage_is_kept() -> None:
    passage = "[scan illegible] EG?R ex@n 1# d%l%tion"
    assert validate_passages(NOTE, [passage]) == [passage]


def test_invented_passage_is_dropped() -> None:
    """A passage that is not in the note cannot send a packet to review."""
    assert validate_passages(NOTE, ["[scan illegible] ALK ??"]) == []


def test_matching_is_exact() -> None:
    assert validate_passages(NOTE, ["[SCAN ILLEGIBLE] EG?R"]) == []
    assert validate_passages(NOTE, ["EG?R  ex@n"]) == []


def test_blank_passages_are_dropped() -> None:
    """Spaces and newlines are in every note and prove nothing."""
    assert validate_passages(NOTE, ["", " ", "\n"]) == []


def test_duplicates_are_removed_and_order_is_kept() -> None:
    result = validate_passages(NOTE, ["Start X.", "EG?R ex@n", "Start X."])
    assert result == ["Start X.", "EG?R ex@n"]


def test_empty_inputs() -> None:
    assert validate_passages(NOTE, []) == []
    assert validate_passages("", ["anything"]) == []


def test_input_list_is_not_modified() -> None:
    passages = ["EG?R ex@n", "invented"]
    validate_passages(NOTE, passages)
    assert passages == ["EG?R ex@n", "invented"]
