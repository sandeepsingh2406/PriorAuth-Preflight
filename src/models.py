"""Pydantic models for PriorAuth Preflight (SPEC section 6)."""
import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["READY", "NOT_READY", "NEEDS_REVIEW"]


class ExtractedField(BaseModel):
    """One required item pulled from the note, with the exact quoted span."""

    name: str
    value: str | None
    quote: str | None
    found: bool
    date: datetime.date | None = None  # used by the lab item for recency checks
    conflicting_quotes: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """What the Packet Analyst returns: one ExtractedField per requested item."""

    fields: list[ExtractedField]


class Gap(BaseModel):
    """A required item that is missing, stale, or backed by conflicting evidence."""

    item: str
    reason: Literal["missing", "stale", "conflicting"]
    detail: str


class GapReport(BaseModel):
    """Result of checking one packet. The Outreach agent sees only this."""

    case_id: str
    payer_id: str
    status: Status
    fields: list[ExtractedField]
    gaps: list[Gap]
    escalation_reason: str | None = None


class Draft(BaseModel):
    """Internal request to the ordering provider's care team. Never sent."""

    subject: str
    body: str
    requested_items: list[str]


class ChecklistItem(BaseModel):
    """One item a payer requires. `lab` items are subject to the recency window."""

    name: str
    description: str
    kind: Literal["field", "lab"]


class PayerChecklist(BaseModel):
    """Required items and lab recency window for one invented payer."""

    payer_id: str
    lab_window_days: int = Field(gt=0)
    items: list[ChecklistItem]


class Order(BaseModel):
    """The synthetic order that accompanies a clinical note."""

    case_id: str
    payer_id: str
    order_date: datetime.date
    drug: str
    dose: str
    schedule: str


class AuditEntry(BaseModel):
    """One line of the audit log. No field can hold note text or keys."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime.datetime
    case_id: str
    tools_called: list[str]
    status: Status
    human_decision: Literal["approved", "edited", "discarded"] | None = None
