"""Code-only tools. Plain deterministic functions: no LLM calls, nothing is sent."""
import re
from pathlib import Path

import yaml

from src.models import ChecklistItem, ExtractedField, Gap, Order, PayerChecklist, Status

CHECKLISTS_PATH = Path(__file__).resolve().parent.parent / "data" / "checklists.yaml"
PLAN_ITEM = "treatment_plan"  # the one item that is also compared with the order


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


def _plan_conflict(quote: str, order: Order) -> str | None:
    """Why the quoted plan disagrees with the order, or None if drug and dose both appear.

    Case and spacing are ignored. The dose must appear as a whole number, so
    "200 mg" is not satisfied by "1200 mg" or "0.200 mg". Schedule is not compared.
    """
    text = " ".join(quote.lower().split())
    if " ".join(order.drug.lower().split()) not in text:
        return "The treatment plan does not mention the drug on the order."
    dose = "".join(order.dose.lower().split())
    pattern = r"(?<![\w.])" + r"\s*".join(re.escape(ch) for ch in dose)
    if not re.search(pattern, text):
        return "The treatment plan does not show the dose on the order."
    return None


def _lab_problem(field: ExtractedField, window: int, order: Order) -> Gap | None:
    """Recency check for the lab item: stale, or a missing or impossible date."""
    if field.date is None:
        return Gap(item=field.name, reason="conflicting", detail="The lab date is missing or unreadable.")
    age = (order.order_date - field.date).days
    if age < 0:
        return Gap(
            item=field.name,
            reason="conflicting",
            detail=f"The lab is dated {field.date}, after the order date {order.order_date}.",
        )
    if age > window:
        return Gap(
            item=field.name,
            reason="stale",
            detail=(
                f"The lab is dated {field.date}, {age} days before the order date; "
                f"this payer allows {window} days."
            ),
        )
    return None


def _check_item(
    item: ChecklistItem, entries: list[ExtractedField], window: int, order: Order
) -> Gap | None:
    """Decide the single gap, if any, for one checklist item. First matching rule wins."""
    if len(entries) > 1:
        return Gap(
            item=item.name,
            reason="conflicting",
            detail=f"There are {len(entries)} entries for {item.name}; cannot tell which is right.",
        )
    field = entries[0] if entries else None
    if field is not None and field.conflicting_quotes:
        return Gap(
            item=item.name,
            reason="conflicting",
            detail=f"The note has {len(field.conflicting_quotes)} conflicting passage(s) about this item.",
        )
    if field is None or not field.found or field.quote is None or not field.quote.strip():
        return Gap(item=item.name, reason="missing", detail=f"{item.description} is not documented in the note.")
    if item.kind == "lab":
        return _lab_problem(field, window, order)
    if item.name == PLAN_ITEM:
        problem = _plan_conflict(field.quote, order)
        if problem:
            return Gap(item=item.name, reason="conflicting", detail=problem)
    return None


def check_packet(
    fields: list[ExtractedField], checklist: PayerChecklist, order: Order
) -> list[Gap]:
    """Compare validated fields with the payer checklist and the order.

    Returns at most one gap per checklist item, in checklist order. Expects fields
    that already went through validate_citations. Inputs are not modified.
    """
    if order.payer_id != checklist.payer_id:
        raise ValueError(
            f"Order is for payer {order.payer_id!r} but the checklist is for {checklist.payer_id!r}"
        )
    gaps = []
    for item in checklist.items:
        entries = [f for f in fields if f.name == item.name]
        gap = _check_item(item, entries, checklist.lab_window_days, order)
        if gap is not None:
            gaps.append(gap)
    return gaps


def route(gaps: list[Gap]) -> Status:
    """Any conflict needs a person; otherwise any gap means not ready; otherwise ready."""
    if any(g.reason == "conflicting" for g in gaps):
        return "NEEDS_REVIEW"
    return "NOT_READY" if gaps else "READY"
