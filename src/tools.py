"""Code-only tools. Plain deterministic functions: no LLM calls, nothing is sent."""
from pathlib import Path

import yaml

from src.models import PayerChecklist

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
