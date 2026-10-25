"""Tests for OutreachRequest and build_outreach_request: the only thing the Outreach agent sees."""
import pytest
from pydantic import ValidationError

from src.models import ExtractedField, Gap, GapReport, OutreachRequest
from src.tools import build_outreach_request

SECRETS = ["SECRET QUOTE ALPHA", "SECRET CONFLICT BRAVO", "SECRET UNREADABLE CHARLIE", "SECRET VALUE DELTA"]


def _report(status: str = "NOT_READY") -> GapReport:
    return GapReport(
        case_id="gen_001",
        payer_id="payer_b",
        status=status,  # type: ignore[arg-type]
        fields=[
            ExtractedField(
                name="stage",
                value=SECRETS[3],
                quote=SECRETS[0],
                found=True,
                conflicting_quotes=[SECRETS[1]],
            )
        ],
        gaps=[
            Gap(item="prior_therapies", reason="missing", detail="Not documented in the note: Prior therapies."),
            Gap(item="recent_lab", reason="stale", detail="The lab is dated 2026-01-02, 90 days before the order date."),
        ],
        escalation_reason=None,
        unreadable_passages=[SECRETS[2]],
    )


def test_the_request_has_exactly_the_agreed_fields() -> None:
    """SPEC section 6: case id, payer id, status, gaps, escalation reason; nothing from the note."""
    assert set(OutreachRequest.model_fields) == {
        "case_id",
        "payer_id",
        "status",
        "gaps",
        "escalation_reason",
    }


def test_builder_copies_the_gap_list_and_identifiers() -> None:
    request = build_outreach_request(_report())
    assert (request.case_id, request.payer_id, request.status) == ("gen_001", "payer_b", "NOT_READY")
    assert [(g.item, g.reason) for g in request.gaps] == [("prior_therapies", "missing"), ("recent_lab", "stale")]
    assert request.escalation_reason is None


def test_no_quote_value_or_unreadable_text_reaches_the_request() -> None:
    """Fields, quotes, conflicting quotes and unreadable passages are all left behind."""
    text = build_outreach_request(_report()).model_dump_json()
    for secret in SECRETS:
        assert secret not in text


def test_the_request_refuses_extra_fields() -> None:
    """Even by mistake, nobody can attach fields or passages to it."""
    with pytest.raises(ValidationError):
        OutreachRequest(
            case_id="c", payer_id="p", status="NOT_READY", gaps=[], fields=[ExtractedField(name="a", value=None, quote=None, found=False)]
        )
    with pytest.raises(ValidationError):
        OutreachRequest(case_id="c", payer_id="p", status="NOT_READY", gaps=[], unreadable_passages=["x"])


def test_the_request_is_independent_of_the_report() -> None:
    """Changing the request's gaps cannot change the report."""
    report = _report()
    request = build_outreach_request(report)
    request.gaps.clear()
    assert len(report.gaps) == 2


def test_escalation_reason_is_carried_over() -> None:
    report = _report("NEEDS_REVIEW").model_copy(update={"escalation_reason": "Unreadable text in the note (1 passage(s))."})
    assert build_outreach_request(report).escalation_reason == "Unreadable text in the note (1 passage(s))."


@pytest.mark.parametrize("not_a_report", [{"case_id": "x"}, "text", None])
def test_only_a_gap_report_is_accepted(not_a_report: object) -> None:
    with pytest.raises(TypeError):
        build_outreach_request(not_a_report)  # type: ignore[arg-type]
