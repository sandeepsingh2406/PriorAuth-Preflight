"""Code-only tools. Plain deterministic functions: no LLM calls, nothing is sent."""
from pathlib import Path

import yaml

from src.models import ExtractedField, PayerChecklist

CHECKLISTS_PATH = Path(__file__).resolve().parent.parent / "data" / "checklists.yaml"


def lookup_payer_checklist(payer_id: str, path: Path = CHECKLISTS_PATH) -> PayerChecklist:
    """Return the checklist for one payer. Raises ValueError if it is not in the file."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("payers"), list):
        raise ValueError(f"{path} must contain a 'payers' list")

    checklists = [PayerChecklist.model_validate(entry) for entry in raw["payers"]]
    ids = [c.payer_id for c in checklists]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} has a duplicate payer_id")
    for checklist in checklists:
        if checklist.payer_id == payer_id:
            return checklist
    raise ValueError(f"Unknown payer_id {payer_id!r}; known payers: {sorted(ids)}")


def _is_verbatim(note: str, quote: str | None) -> bool:
    """True if the quote is non-blank and appears exactly as written in the note."""
    return quote is not None and quote.strip() != "" and quote in note


def validate_citations(note: str, fields: list[ExtractedField]) -> list[ExtractedField]:
    """Drop any evidence whose quote is not a verbatim substring of the note.

    Matching is exact: no case folding, no whitespace changes. A field that fails
    becomes not found, with value, quote and date cleared, because they came from
    evidence that cannot be verified. Fields are never upgraded to found. Inputs
    are not modified.
    """
    results = []
    for field in fields:
        conflicting = [q for q in field.conflicting_quotes if _is_verbatim(note, q)]
        if field.found and _is_verbatim(note, field.quote):
            results.append(field.model_copy(update={"conflicting_quotes": conflicting}))
        else:
            results.append(
                field.model_copy(
                    update={
                        "found": False,
                        "value": None,
                        "quote": None,
                        "date": None,
                        "conflicting_quotes": conflicting,
                    }
                )
            )
    return results
