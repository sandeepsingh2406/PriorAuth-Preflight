"""Code-only tools. Plain deterministic functions: no LLM calls, nothing is sent."""
import os
import re
from pathlib import Path

import yaml

from src.models import (
    AuditEntry,
    ChecklistItem,
    ExtractedField,
    Gap,
    GapReport,
    Order,
    OutreachRequest,
    PayerChecklist,
    Status,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKLISTS_PATH = PROJECT_ROOT / "data" / "checklists.yaml"
AUDIT_ENV_VAR = "AUDIT_LOG_PATH"
DEFAULT_AUDIT_LOG = Path("audit") / "runs.jsonl"
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


def validate_passages(note: str, passages: list[str]) -> list[str]:
    """Keep only reported unreadable passages that appear verbatim in the note.

    Same rule as validate_citations: exact match, blanks refused. Duplicates are
    removed and order is kept. Text the model made up cannot send a packet to review.
    """
    kept: list[str] = []
    for passage in passages:
        if _is_verbatim(note, passage) and passage not in kept:
            kept.append(passage)
    return kept


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
        return Gap(item=item.name, reason="missing", detail=f"Not documented in the note: {item.description}.")
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


def audit_log_path(path: Path | None = None) -> Path:
    """Where the audit log goes: the given path, else AUDIT_LOG_PATH, else audit/runs.jsonl.

    A path passed in is used as given. A relative path from the environment or the
    default is anchored at the project root, so it does not depend on the working folder.
    """
    if path is not None:
        return path
    chosen = Path(os.environ.get(AUDIT_ENV_VAR) or DEFAULT_AUDIT_LOG)
    return chosen if chosen.is_absolute() else PROJECT_ROOT / chosen


def write_audit_log(entry: AuditEntry, path: Path | None = None) -> None:
    """Append one JSON line for a run. Only an AuditEntry is accepted, so no note text can slip in.

    Write errors are raised: a missing audit line must never go unnoticed.
    """
    if not isinstance(entry, AuditEntry):
        raise TypeError("write_audit_log only accepts an AuditEntry")
    target = audit_log_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as log:
        log.write(entry.model_dump_json() + "\n")


def escalate_to_human(report: GapReport, reason: str) -> GapReport:
    """Return a copy of the report set to NEEDS_REVIEW with the reason, keeping fields and gaps.

    A blank reason is refused. If the report already has a reason, the new one is
    appended so nothing is lost. This only builds a report; it sends nothing.
    """
    reason = reason.strip()
    if not reason:
        raise ValueError("escalate_to_human needs a non-blank reason")
    if report.escalation_reason:
        reason = f"{report.escalation_reason}; {reason}"
    return report.model_copy(update={"status": "NEEDS_REVIEW", "escalation_reason": reason})


def build_outreach_request(report: GapReport) -> OutreachRequest:
    """Reduce a report to the gap list for the Outreach agent: no fields, quotes or passages.

    Only the report's identifiers, status, gaps and escalation reason are copied. Gap
    text is written by code and never repeats quoted note text.
    """
    if not isinstance(report, GapReport):
        raise TypeError("build_outreach_request only accepts a GapReport")
    return OutreachRequest(
        case_id=report.case_id,
        payer_id=report.payer_id,
        status=report.status,
        gaps=[gap.model_copy() for gap in report.gaps],
        escalation_reason=report.escalation_reason,
    )
