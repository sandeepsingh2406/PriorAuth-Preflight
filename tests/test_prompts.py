"""Guards on the prompt text in src/prompts.py.

These do not test model behaviour. They make sure no required rule is lost by an
accidental edit.
"""
import pytest

from src.prompts import (
    EXTRACT_FIELDS_INPUT,
    EXTRACT_FIELDS_INSTRUCTIONS,
    PACKET_ANALYST_INSTRUCTIONS,
)


@pytest.mark.parametrize(
    "rule",
    [
        "exactly one entry for each requested item",
        "verbatim",
        "character for character",
        "found = false",
        "never infer a stage or a biomarker",
        "conflicting_quotes",
        "do not choose",
        "pending",
        "not yet reported",
        "family history",
        "treatment-naive",
        "yyyy-mm-dd",
        "ignore any instructions",
        "unreadable_passages",
        "illegible",
        "do not try to reconstruct",
        "abbreviations",
    ],
)
def test_extraction_prompt_keeps_each_required_rule(rule: str) -> None:
    """Each rule the extraction must follow is spelled out in the instructions."""
    assert rule in " ".join(EXTRACT_FIELDS_INSTRUCTIONS.lower().split())  # ignores line wrapping


@pytest.mark.parametrize(
    "rule",
    ["do not decide", "never submit, send", "plain code", "a person reviews"],
)
def test_packet_analyst_prompt_states_its_limits(rule: str) -> None:
    """The agent is told it only reads and quotes; code decides; it sends nothing."""
    assert rule in " ".join(PACKET_ANALYST_INSTRUCTIONS.lower().split())  # ignores line wrapping


def test_input_template_holds_the_items_and_the_note() -> None:
    """The user message carries the requested items and the note between clear markers."""
    text = EXTRACT_FIELDS_INPUT.format(items="- stage: Cancer stage", note="NOTE BODY")
    assert "- stage: Cancer stage" in text
    assert "NOTE BODY" in text
    assert text.index("- stage") < text.index("NOTE BODY")
