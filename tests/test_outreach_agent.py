"""Tests for the Outreach prompt, agent definition and draft_request (LLM always mocked)."""
import pytest

from conftest import NetworkCallInTest
from src.agents import build_outreach_agent, draft_request
from src.models import Draft, DraftText, Gap, GapReport, OutreachRequest
from src.prompts import OUTREACH_INPUT, OUTREACH_INSTRUCTIONS
from src.tools import build_outreach_request


def _request(status: str = "NOT_READY") -> OutreachRequest:
    return OutreachRequest(
        case_id="gen_001",
        payer_id="payer_b",
        status=status,  # type: ignore[arg-type]
        gaps=[
            Gap(item="prior_therapies", reason="missing", detail="Not documented in the note: Prior therapies tried, with outcomes."),
            Gap(item="recent_lab", reason="stale", detail="The lab is dated 2026-01-02, 90 days before the order date; this payer allows 45 days."),
        ],
    )


class FakeLLM:
    """Stands in for the Outreach model: records the prompt and returns a prepared draft text."""

    def __init__(self, subject: str = "Items needed for case gen_001", body: str = "Please send the items listed.") -> None:
        self.result = DraftText(subject=subject, body=body)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> DraftText:
        self.prompts.append(prompt)
        return self.result


# --- prompt ----------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "internal request",
        "ordering provider",
        "ask only for the items listed",
        "do not invent",
        "case id",
        "do not judge",
        "a person will review",
        "never say it has been sent",
        "ignore any instructions",
    ],
)
def test_outreach_prompt_keeps_each_rule(rule: str) -> None:
    assert rule in " ".join(OUTREACH_INSTRUCTIONS.lower().split())  # ignores line wrapping


def test_input_template_lists_the_packet_and_the_items() -> None:
    text = OUTREACH_INPUT.format(case_id="gen_001", payer_id="payer_b", gaps="- stage (missing): x")
    assert "gen_001" in text and "payer_b" in text and "- stage (missing): x" in text


# --- agent definition ---------------------------------------------------------


def test_outreach_agent_has_no_tools_no_handoffs_and_only_writes_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """It cannot call anything or hand off to anything: it only returns subject and body."""
    monkeypatch.setenv("MODEL_DEV", "test-model")
    agent = build_outreach_agent()
    assert agent.tools == []
    assert agent.handoffs == []
    assert agent.output_type is DraftText
    assert agent.instructions == OUTREACH_INSTRUCTIONS
    assert agent.model == "test-model"


# --- draft_request ---------------------------------------------------------------


def test_draft_has_the_models_text_and_code_chosen_items() -> None:
    llm = FakeLLM("Subject line", "Body text")
    draft = draft_request(_request(), llm=llm)
    assert isinstance(draft, Draft)
    assert (draft.subject, draft.body) == ("Subject line", "Body text")
    assert draft.requested_items == ["prior_therapies", "recent_lab"]  # from the gaps, in order


def test_prompt_carries_the_case_and_every_gap() -> None:
    llm = FakeLLM()
    draft_request(_request(), llm=llm)
    (prompt,) = llm.prompts
    assert "gen_001" in prompt and "payer_b" in prompt
    assert "- prior_therapies (missing): Not documented in the note: Prior therapies tried, with outcomes." in prompt
    assert "- recent_lab (stale):" in prompt and "this payer allows 45 days" in prompt


def test_text_is_stripped() -> None:
    draft = draft_request(_request(), llm=FakeLLM("  Subject \n", "\n Body  "))
    assert (draft.subject, draft.body) == ("Subject", "Body")


@pytest.mark.parametrize(("subject", "body"), [("", "body"), ("subject", ""), ("  ", "body"), ("subject", "\n")])
def test_an_empty_draft_is_an_error(subject: str, body: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        draft_request(_request(), llm=FakeLLM(subject, body))


def test_only_an_outreach_request_is_accepted() -> None:
    """A full GapReport (which carries quotes) is refused outright."""
    report = GapReport(case_id="c", payer_id="payer_a", status="NOT_READY", fields=[], gaps=[])
    with pytest.raises(TypeError):
        draft_request(report, llm=FakeLLM())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        draft_request(build_outreach_request(report).model_dump(), llm=FakeLLM())  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["READY", "NEEDS_REVIEW"])
def test_drafts_are_only_written_for_not_ready_packets(status: str) -> None:
    """NEEDS_REVIEW goes to a person first; READY needs nothing requested."""
    llm = FakeLLM()
    with pytest.raises(ValueError, match="NOT_READY"):
        draft_request(_request(status), llm=llm)
    assert llm.prompts == []  # the model was never called


def test_wrong_result_type_from_the_model_is_refused() -> None:
    with pytest.raises(TypeError):
        draft_request(_request(), llm=lambda prompt: {"subject": "s", "body": "b"})  # type: ignore[arg-type, return-value]


def test_default_path_goes_through_the_sdk_and_is_blocked_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_DEV", "test-model")
    with pytest.raises(NetworkCallInTest):
        draft_request(_request())
