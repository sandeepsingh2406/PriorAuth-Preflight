"""Tests for the Pydantic models in src/models.py."""
import datetime

import pytest
from pydantic import ValidationError

from src.models import (
    AuditEntry,
    ChecklistItem,
    Draft,
    ExtractedField,
    Gap,
    GapReport,
    Order,
    PayerChecklist,
)


def test_extracted_field_defaults() -> None:
    """Optional fields default to "no lab date" and "no conflicting quotes"."""
    field = ExtractedField(name="stage", value=None, quote=None, found=False)
    assert field.date is None
    assert field.conflicting_quotes == []


def test_extracted_field_conflicting_quotes_not_shared() -> None:
    """Each field gets its own list, so adding a quote to one leaves others empty."""
    a = ExtractedField(name="stage", value=None, quote=None, found=False)
    b = ExtractedField(name="stage", value=None, quote=None, found=False)
    a.conflicting_quotes.append("Stage II")
    assert b.conflicting_quotes == []


def test_extracted_field_parses_iso_date() -> None:
    """An ISO date string such as "2026-03-01" becomes a real date object."""
    field = ExtractedField(
        name="lab", value="ANC 1.2", quote="ANC 1.2", found=True, date="2026-03-01"
    )
    assert field.date == datetime.date(2026, 3, 1)


def test_gap_rejects_unknown_reason() -> None:
    """A gap reason outside missing/stale/conflicting raises a validation error."""
    with pytest.raises(ValidationError):
        Gap(item="stage", reason="absent", detail="not in note")


@pytest.mark.parametrize("reason", ["missing", "stale", "conflicting"])
def test_gap_accepts_spec_reasons(reason: str) -> None:
    """Runs once per reason; each of the three SPEC values is accepted."""
    assert Gap(item="stage", reason=reason, detail="x").reason == reason


def test_gap_report_rejects_unknown_status() -> None:
    """A status outside READY/NOT_READY/NEEDS_REVIEW raises a validation error."""
    with pytest.raises(ValidationError):
        GapReport(
            case_id="gen_001", payer_id="payer_a", status="DONE", fields=[], gaps=[]
        )


def test_gap_report_defaults_and_roundtrip() -> None:
    """escalation_reason defaults to None, and a report survives JSON in and out unchanged."""
    report = GapReport(
        case_id="gen_001",
        payer_id="payer_a",
        status="NOT_READY",
        fields=[ExtractedField(name="stage", value=None, quote=None, found=False)],
        gaps=[Gap(item="stage", reason="missing", detail="no stage in note")],
    )
    assert report.escalation_reason is None
    assert GapReport.model_validate_json(report.model_dump_json()) == report


def test_draft_fields() -> None:
    """A draft stores its subject, body, and the list of requested items."""
    draft = Draft(subject="Missing items", body="Please send stage.", requested_items=["stage"])
    assert draft.requested_items == ["stage"]


def test_checklist_item_rejects_unknown_kind() -> None:
    """A checklist item kind other than field/lab raises a validation error."""
    with pytest.raises(ValidationError):
        ChecklistItem(name="stage", description="Cancer stage", kind="other")


def test_payer_checklist_rejects_non_positive_window() -> None:
    """A lab recency window of 0 days is invalid; it must be at least 1."""
    with pytest.raises(ValidationError):
        PayerChecklist(payer_id="payer_a", lab_window_days=0, items=[])


def test_order_parses_iso_date() -> None:
    """An order's ISO date string becomes a real date, ready for recency math."""
    order = Order(
        case_id="gen_001",
        payer_id="payer_a",
        order_date="2026-04-01",
        drug="Examplestat",
        dose="100 mg",
        schedule="daily",
    )
    assert order.order_date == datetime.date(2026, 4, 1)


def test_audit_entry_rejects_extra_keys() -> None:
    """The audit log must have no field where note text could land."""
    with pytest.raises(ValidationError):
        AuditEntry(
            timestamp=datetime.datetime(2026, 10, 24, 12, 0),
            case_id="gen_001",
            tools_called=["check_packet"],
            status="READY",
            note="full clinical note text",
        )


def test_audit_entry_human_decision_defaults_none_and_validates() -> None:
    """human_decision is optional, but when given it must be approved/edited/discarded."""
    entry = AuditEntry(
        timestamp=datetime.datetime(2026, 10, 24, 12, 0),
        case_id="gen_001",
        tools_called=[],
        status="READY",
    )
    assert entry.human_decision is None
    with pytest.raises(ValidationError):
        AuditEntry(
            timestamp=datetime.datetime(2026, 10, 24, 12, 0),
            case_id="gen_001",
            tools_called=[],
            status="READY",
            human_decision="maybe",
        )


def test_extraction_result_and_gap_report_default_to_no_unreadable_passages() -> None:
    """Unreadable text is optional: nothing unreadable is the normal case."""
    from src.models import ExtractionResult

    assert ExtractionResult(fields=[]).unreadable_passages == []
    report = GapReport(case_id="c", payer_id="payer_a", status="READY", fields=[], gaps=[])
    assert report.unreadable_passages == []
