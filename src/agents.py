"""The Packet Analyst agent. It only reads the note and returns quoted evidence.

Plain code in src/tools.py checks the quotes and makes every decision.
"""
import datetime
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agents import Agent, Runner

from src.models import (
    AuditEntry,
    Draft,
    DraftText,
    ExtractedField,
    ExtractionResult,
    Gap,
    GapReport,
    Order,
    OutreachRequest,
    PayerChecklist,
)
from src.prompts import (
    EXTRACT_FIELDS_INPUT,
    EXTRACT_FIELDS_INSTRUCTIONS,
    OUTREACH_INPUT,
    OUTREACH_INSTRUCTIONS,
    PACKET_ANALYST_INSTRUCTIONS,
)
from src.tools import (
    check_packet,
    escalate_to_human,
    lookup_payer_checklist,
    build_outreach_request,
    route,
    validate_citations,
    validate_passages,
    write_audit_log,
)

# Takes the user message, returns the structured extraction. Tests inject a fake.
ExtractionLLM = Callable[[str], ExtractionResult]
OutreachLLM = Callable[[str], DraftText]


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
) -> ExtractionResult:
    """Ask the model for each of the payer's required items, with verbatim quotes.

    Returns the model's result as it is. Nothing here is trusted yet:
    validate_citations, validate_passages and check_packet run next.
    """
    items = "\n".join(f"- {item.name}: {item.description}" for item in checklist.items)
    prompt = EXTRACT_FIELDS_INPUT.format(items=items, note=note)
    result = (llm or _call_model)(prompt)
    if not isinstance(result, ExtractionResult):
        raise TypeError(f"Expected ExtractionResult from the model, got {type(result).__name__}")
    return result


def build_outreach_agent() -> Agent:
    """The Outreach agent: no tools, no handoffs. It only writes a subject and a body."""
    return Agent(
        name="Outreach",
        instructions=OUTREACH_INSTRUCTIONS,
        model=model_name(),
        output_type=DraftText,
    )


def _call_outreach_model(prompt: str) -> DraftText:
    """Run the Outreach agent once through the Agents SDK."""
    return Runner.run_sync(build_outreach_agent(), prompt).final_output


def draft_request(request: OutreachRequest, *, llm: OutreachLLM | None = None) -> Draft:
    """Draft the internal request to the care team from the gap list alone.

    Only an OutreachRequest is accepted, and only for a NOT_READY packet. The model
    writes the subject and body; the requested items come from the gaps, not the model.
    The draft is never sent.
    """
    if not isinstance(request, OutreachRequest):
        raise TypeError("draft_request only accepts an OutreachRequest")
    if request.status != "NOT_READY":
        raise ValueError(f"Drafts are only written for NOT_READY packets, not {request.status}")
    gaps = "\n".join(f"- {gap.item} ({gap.reason}): {gap.detail}" for gap in request.gaps)
    prompt = OUTREACH_INPUT.format(case_id=request.case_id, payer_id=request.payer_id, gaps=gaps)
    text = (llm or _call_outreach_model)(prompt)
    if not isinstance(text, DraftText):
        raise TypeError(f"Expected DraftText from the model, got {type(text).__name__}")
    subject, body = text.subject.strip(), text.body.strip()
    if not subject or not body:
        raise ValueError("The model returned an empty draft")
    return Draft(subject=subject, body=body, requested_items=[gap.item for gap in request.gaps])


@dataclass
class AnalysisRun:
    """One run of the Packet Analyst pipeline."""

    report: GapReport
    raw_fields: list[ExtractedField]  # what the model returned, before validation
    tools_called: list[str]
    raw_unreadable: list[str] = field(default_factory=list)  # unreadable passages as claimed
    draft: Draft | None = None  # the internal request, only for NOT_READY and only when asked
    draft_error: str | None = None  # error type if the Outreach step failed


def _review_reason(gaps: list[Gap]) -> str:
    """Escalation reason built by code from the conflicting gaps (no quoted note text)."""
    return "; ".join(f"{g.item}: {g.detail}" for g in gaps if g.reason == "conflicting")


def analyze_packet(
    order: Order,
    note: str,
    *,
    llm: ExtractionLLM | None = None,
    audit_path: Path | None = None,
    draft: bool = False,
    outreach_llm: OutreachLLM | None = None,
) -> AnalysisRun:
    """Run lookup, extraction, citation check and packet check in that fixed order.

    Code decides the status: route() on the gaps, and escalate_to_human for
    NEEDS_REVIEW. If the model call fails, the packet goes to a person. Every run
    writes exactly one audit line (case id, tools called, status, timestamp).

    With draft=True, a NOT_READY packet also gets an internal request drafted by the
    Outreach agent from the gap list alone. NEEDS_REVIEW and READY packets never do. If
    the Outreach step fails the analysis stands and only the draft is missing.
    """
    tools_called = ["lookup_payer_checklist"]
    checklist = lookup_payer_checklist(order.payer_id)

    tools_called.append("extract_fields")
    try:
        extraction = extract_fields(note, checklist, llm=llm)
    except Exception as error:  # any model or parsing failure: fail toward a person
        report = GapReport(
            case_id=order.case_id, payer_id=order.payer_id, status="NOT_READY", fields=[], gaps=[]
        )
        report = escalate_to_human(
            report, f"Extraction failed ({type(error).__name__}); review the note by hand."
        )
        tools_called.append("escalate_to_human")
        _audit(order, tools_called, report, audit_path)
        return AnalysisRun(report=report, raw_fields=[], tools_called=tools_called)

    raw_fields = extraction.fields
    tools_called.append("validate_citations")
    fields = validate_citations(note, raw_fields)
    unreadable = validate_passages(note, extraction.unreadable_passages)
    tools_called.append("check_packet")
    gaps = check_packet(fields, checklist, order)

    status = route(gaps)
    reasons = []
    if status == "NEEDS_REVIEW":
        reasons.append(_review_reason(gaps))
    if unreadable:
        reasons.append(f"Unreadable text in the note ({len(unreadable)} passage(s)); review it by hand.")
    report = GapReport(
        case_id=order.case_id,
        payer_id=order.payer_id,
        status=status,
        fields=fields,
        gaps=gaps,
        unreadable_passages=unreadable,
    )
    if reasons:
        report = escalate_to_human(report, "; ".join(reasons))
        tools_called.append("escalate_to_human")

    drafted, draft_error = None, None
    if draft and report.status == "NOT_READY":
        try:
            drafted = draft_request(build_outreach_request(report), llm=outreach_llm)
            tools_called.append("draft_request")
        except Exception as error:  # the analysis stands; only the draft is missing
            draft_error = type(error).__name__
            tools_called.append("draft_request_failed")

    _audit(order, tools_called, report, audit_path)
    return AnalysisRun(
        report=report,
        raw_fields=raw_fields,
        tools_called=tools_called,
        raw_unreadable=list(extraction.unreadable_passages),
        draft=drafted,
        draft_error=draft_error,
    )


def _audit(order: Order, tools_called: list[str], report: GapReport, path: Path | None) -> None:
    write_audit_log(
        AuditEntry(
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            case_id=order.case_id,
            tools_called=list(tools_called),
            status=report.status,
        ),
        path=path,
    )
