"""The Packet Analyst agent. It only reads the note and returns quoted evidence.

Plain code in src/tools.py checks the quotes and makes every decision.
"""
import os
from collections.abc import Callable

from agents import Agent, Runner

from src.models import ExtractedField, ExtractionResult, PayerChecklist
from src.prompts import (
    EXTRACT_FIELDS_INPUT,
    EXTRACT_FIELDS_INSTRUCTIONS,
    PACKET_ANALYST_INSTRUCTIONS,
)

# Takes the user message, returns the structured extraction. Tests inject a fake.
ExtractionLLM = Callable[[str], ExtractionResult]


def model_name() -> str:
    """MODEL_DEMO when USE_DEMO_MODEL is true, otherwise MODEL_DEV. Never hard-coded."""
    use_demo = os.environ.get("USE_DEMO_MODEL", "false").strip().lower() in ("true", "1")
    var = "MODEL_DEMO" if use_demo else "MODEL_DEV"
    name = os.environ.get(var, "").strip()
    if not name:
        raise RuntimeError(f"{var} is not set; add it to .env (see env.example)")
    return name


def build_packet_analyst() -> Agent:
    """The Packet Analyst: no tools, no handoffs, structured output only."""
    return Agent(
        name="Packet Analyst",
        instructions=PACKET_ANALYST_INSTRUCTIONS + "\n" + EXTRACT_FIELDS_INSTRUCTIONS,
        model=model_name(),
        output_type=ExtractionResult,
    )


def _call_model(prompt: str) -> ExtractionResult:
    """Run the Packet Analyst once through the Agents SDK."""
    return Runner.run_sync(build_packet_analyst(), prompt).final_output


def extract_fields(
    note: str, checklist: PayerChecklist, *, llm: ExtractionLLM | None = None
) -> list[ExtractedField]:
    """Ask the model for each of the payer's required items, with verbatim quotes.

    Returns the model's fields as they are. Nothing here is trusted yet:
    validate_citations and check_packet run next.
    """
    items = "\n".join(f"- {item.name}: {item.description}" for item in checklist.items)
    prompt = EXTRACT_FIELDS_INPUT.format(items=items, note=note)
    result = (llm or _call_model)(prompt)
    if not isinstance(result, ExtractionResult):
        raise TypeError(f"Expected ExtractionResult from the model, got {type(result).__name__}")
    return list(result.fields)
