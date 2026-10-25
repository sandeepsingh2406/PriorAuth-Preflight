"""Tests for extract_fields and the Packet Analyst agent definition (LLM always mocked)."""
import datetime

import pytest

from conftest import NetworkCallInTest
from src.agents import build_packet_analyst, extract_fields, model_name
from src.models import ExtractedField, ExtractionResult
from src.prompts import EXTRACT_FIELDS_INSTRUCTIONS, PACKET_ANALYST_INSTRUCTIONS
from src.tools import lookup_payer_checklist

NOTE = (
    "SYNTHETIC NOTE\n"
    "Staging: Stage II (cT2 N0 M0).\n"
    "Assessment: patient with stage IV disease.\n"
    "Labs: CBC drawn 2026-04-22.\n"
)


class FakeLLM:
    """Stands in for the model: records each prompt and returns a prepared result."""

    def __init__(self, fields: list[ExtractedField], unreadable: list[str] | None = None) -> None:
        self.result = ExtractionResult(fields=fields, unreadable_passages=unreadable or [])
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> ExtractionResult:
        self.prompts.append(prompt)
        return self.result


def _field(name: str, quote: str | None, **kw: object) -> ExtractedField:
    return ExtractedField(name=name, value=quote, quote=quote, found=quote is not None, **kw)


def test_found_field_comes_back_with_its_quote() -> None:
    """A field the model found is returned with its quote, value and lab date intact."""
    llm = FakeLLM([_field("recent_lab", "CBC drawn 2026-04-22.", date=datetime.date(2026, 4, 22))])
    (field,) = extract_fields(NOTE, lookup_payer_checklist("payer_a"), llm=llm).fields
    assert isinstance(field, ExtractedField)
    assert field.found is True
    assert field.quote == "CBC drawn 2026-04-22."
    assert field.date == datetime.date(2026, 4, 22)


def test_missing_field_comes_back_not_found() -> None:
    """When the model reports an item as absent, it stays found=False with no quote."""
    llm = FakeLLM([_field("biomarker", None)])
    (field,) = extract_fields(NOTE, lookup_payer_checklist("payer_a"), llm=llm).fields
    assert field.found is False
    assert field.quote is None and field.value is None


def test_conflicting_values_are_passed_through_not_resolved() -> None:
    """extract_fields does not pick a winner: the conflicting quotes reach the code checks."""
    llm = FakeLLM(
        [_field("stage", "Stage II (cT2 N0 M0).", conflicting_quotes=["stage IV disease"])]
    )
    (field,) = extract_fields(NOTE, lookup_payer_checklist("payer_a"), llm=llm).fields
    assert field.quote == "Stage II (cT2 N0 M0)."
    assert field.conflicting_quotes == ["stage IV disease"]


def test_order_of_fields_is_kept() -> None:
    llm = FakeLLM([_field("stage", "Stage II (cT2 N0 M0)."), _field("biomarker", None)])
    result = extract_fields(NOTE, lookup_payer_checklist("payer_a"), llm=llm)
    assert [f.name for f in result.fields] == ["stage", "biomarker"]


def test_prompt_lists_the_payer_items_and_carries_the_note() -> None:
    """The model is asked for exactly the payer's items, with descriptions, plus the note."""
    llm = FakeLLM([])
    checklist = lookup_payer_checklist("payer_c")
    extract_fields(NOTE, checklist, llm=llm)
    (prompt,) = llm.prompts
    for item in checklist.items:
        assert f"- {item.name}: {item.description}" in prompt
    assert "- stage:" not in prompt  # payer_c does not require stage
    assert NOTE in prompt


def test_braces_in_the_note_do_not_break_the_prompt() -> None:
    """Notes are free text; a stray {placeholder} must reach the model unchanged."""
    llm = FakeLLM([])
    note = "SYNTHETIC NOTE\nDose {see pharmacy}.\n"
    extract_fields(note, lookup_payer_checklist("payer_a"), llm=llm)
    assert note in llm.prompts[0]


def test_wrong_result_type_from_the_model_is_refused() -> None:
    """Anything other than an ExtractionResult is an error, not silently used."""
    with pytest.raises(TypeError):
        extract_fields(NOTE, lookup_payer_checklist("payer_a"), llm=lambda prompt: {"fields": []})


def test_default_path_goes_through_the_sdk_and_is_blocked_in_tests(
    monkeypatch: pytest.MonkeyPatch, network_attempts: list[str]
) -> None:
    """Without an injected llm, the real SDK runner is used; the test guard stops it."""
    monkeypatch.setenv("MODEL_DEV", "test-model")
    with pytest.raises(NetworkCallInTest):
        extract_fields(NOTE, lookup_payer_checklist("payer_a"))
    assert len(network_attempts) == 1
    network_attempts.clear()  # expected, so the autouse guard does not fail this test


# --- the agent definition ---------------------------------------------------


def test_packet_analyst_has_no_tools_and_no_handoffs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent can only read and answer in the structured format: nothing to call or hand off to."""
    monkeypatch.setenv("MODEL_DEV", "test-model")
    agent = build_packet_analyst()
    assert agent.tools == []
    assert agent.handoffs == []
    assert agent.output_type is ExtractionResult
    assert PACKET_ANALYST_INSTRUCTIONS in agent.instructions
    assert EXTRACT_FIELDS_INSTRUCTIONS in agent.instructions
    assert agent.model == "test-model"


def test_model_name_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """MODEL_DEV by default, MODEL_DEMO when USE_DEMO_MODEL is true; never hard-coded."""
    monkeypatch.setenv("MODEL_DEV", "dev-model")
    monkeypatch.setenv("MODEL_DEMO", "demo-model")
    monkeypatch.delenv("USE_DEMO_MODEL", raising=False)
    assert model_name() == "dev-model"
    monkeypatch.setenv("USE_DEMO_MODEL", "false")
    assert model_name() == "dev-model"
    monkeypatch.setenv("USE_DEMO_MODEL", "true")
    assert model_name() == "demo-model"


def test_missing_model_name_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_DEV", raising=False)
    monkeypatch.delenv("USE_DEMO_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_DEV"):
        model_name()


def test_unreadable_passages_are_passed_through() -> None:
    """Text the model could not read comes back for the code to validate."""
    llm = FakeLLM([_field("biomarker", None)], unreadable=["EG?R ex@n 1# [unreadable]"])
    result = extract_fields(NOTE, lookup_payer_checklist("payer_a"), llm=llm)
    assert result.unreadable_passages == ["EG?R ex@n 1# [unreadable]"]


def test_extract_fields_returns_an_extraction_result() -> None:
    llm = FakeLLM([_field("stage", "Stage II (cT2 N0 M0).")])
    result = extract_fields(NOTE, lookup_payer_checklist("payer_a"), llm=llm)
    assert isinstance(result, ExtractionResult)
    assert result.unreadable_passages == []
