"""Tests for check_packet and route in src/tools.py.

check_packet is plain code: it compares already-validated fields with the payer
checklist and the order. It never sees the note.
"""
import datetime
import json
import re
from pathlib import Path

import pytest

from src.models import ChecklistItem, ExtractedField, Gap, Order, PayerChecklist
from src.tools import check_packet, lookup_payer_checklist, route, validate_citations

ROOT = Path(__file__).parent.parent
ORDER_DATE = datetime.date(2026, 5, 4)
PLAN_QUOTE = "Start Examplumab 200 mg IV every 21 days."


def _order(payer: str = "payer_a", **changes: object) -> Order:
    base = Order(
        case_id="t_001",
        payer_id=payer,
        order_date=ORDER_DATE,
        drug="Examplumab",
        dose="200 mg",
        schedule="every 21 days",
    )
    return base.model_copy(update=changes)


def _complete(payer: str = "payer_a", lab_age: int = 10) -> list[ExtractedField]:
    """A fully documented packet: every required item found, lab `lab_age` days old."""
    fields = []
    for item in lookup_payer_checklist(payer).items:
        quote = f"{item.name} evidence"
        date = None
        if item.kind == "lab":
            quote, date = "CBC drawn", ORDER_DATE - datetime.timedelta(days=lab_age)
        elif item.name == "treatment_plan":
            quote = PLAN_QUOTE
        fields.append(
            ExtractedField(name=item.name, value=item.name, quote=quote, found=True, date=date)
        )
    return fields


def _change(fields: list[ExtractedField], name: str, **changes: object) -> list[ExtractedField]:
    return [f.model_copy(update=changes) if f.name == name else f for f in fields]


def _drop(fields: list[ExtractedField], name: str) -> list[ExtractedField]:
    return [f for f in fields if f.name != name]


def _gaps(
    fields: list[ExtractedField], payer: str = "payer_a", order: Order | None = None
) -> list[tuple[str, str]]:
    found = check_packet(fields, lookup_payer_checklist(payer), order or _order(payer))
    return [(g.item, g.reason) for g in found]


# --- which items are checked ------------------------------------------------


@pytest.mark.parametrize("payer", ["payer_a", "payer_b", "payer_c"])
def test_complete_packet_has_no_gaps(payer: str) -> None:
    """A packet with every required item documented and a recent lab has no gaps."""
    assert _gaps(_complete(payer), payer) == []


def test_item_absent_from_the_fields_is_missing() -> None:
    """A required item the extractor never returned is a missing gap."""
    assert _gaps(_drop(_complete(), "stage")) == [("stage", "missing")]


def test_field_marked_not_found_is_missing() -> None:
    """found=False means the evidence was not in the note (or failed validation)."""
    fields = _change(_complete(), "biomarker", found=False, value=None, quote=None)
    assert _gaps(fields) == [("biomarker", "missing")]


@pytest.mark.parametrize("quote", [None, "", "   "])
def test_found_without_a_real_quote_counts_as_missing(quote: str | None) -> None:
    """Unvalidated evidence is not evidence: found=True with no usable quote is missing."""
    assert _gaps(_change(_complete(), "stage", quote=quote)) == [("stage", "missing")]


def test_only_items_the_payer_requires_are_checked() -> None:
    """Payer C does not require stage; payer A does not require prior_therapies."""
    assert _gaps(_drop(_complete("payer_c"), "biomarker"), "payer_c") == [("biomarker", "missing")]
    c_extra = _complete("payer_c") + [
        ExtractedField(name="stage", value=None, quote=None, found=False)
    ]
    assert _gaps(c_extra, "payer_c") == []
    a_extra = _complete("payer_a") + [
        ExtractedField(name="prior_therapies", value=None, quote=None, found=False)
    ]
    assert _gaps(a_extra, "payer_a") == []


def test_fields_not_on_the_checklist_are_ignored() -> None:
    """An unknown field name neither creates a gap nor hides one."""
    extra = ExtractedField(name="favourite_colour", value="blue", quote="blue", found=True)
    assert _gaps(_complete() + [extra]) == []


def test_gaps_follow_checklist_order_with_one_gap_per_item() -> None:
    """Output order is the checklist order, so results are stable run to run."""
    fields = _drop(_drop(_drop(_complete("payer_b"), "treatment_plan"), "stage"), "prior_therapies")
    assert _gaps(fields, "payer_b") == [
        ("stage", "missing"),
        ("prior_therapies", "missing"),
        ("treatment_plan", "missing"),
    ]


# --- conflicting evidence ----------------------------------------------------


def test_surviving_conflicting_quotes_make_a_conflicting_gap() -> None:
    """A verified contradicting quote means a person must look at the item."""
    fields = _change(_complete(), "stage", conflicting_quotes=["Stage IV disease"])
    assert _gaps(fields) == [("stage", "conflicting")]


def test_conflicting_wins_over_missing_and_gives_one_gap() -> None:
    """Even when the main quote failed, real contradicting evidence means review."""
    fields = _change(
        _complete(), "stage", found=False, value=None, quote=None, conflicting_quotes=["Stage IV"]
    )
    assert _gaps(fields) == [("stage", "conflicting")]


def test_duplicate_field_names_are_conflicting_not_silently_resolved() -> None:
    """Two entries for one item: do not guess which one wins, send it to review."""
    dup = ExtractedField(name="stage", value="Stage I", quote="Stage I", found=True)
    assert _gaps(_complete() + [dup]) == [("stage", "conflicting")]


# --- lab recency ------------------------------------------------------------


@pytest.mark.parametrize(("payer", "window"), [("payer_a", 30), ("payer_b", 45), ("payer_c", 60)])
def test_lab_exactly_at_the_window_is_fine(payer: str, window: int) -> None:
    """Stale means strictly older than the window: exactly the window is accepted."""
    assert _gaps(_complete(payer, lab_age=window), payer) == []


@pytest.mark.parametrize(("payer", "window"), [("payer_a", 30), ("payer_b", 45), ("payer_c", 60)])
def test_lab_one_day_past_the_window_is_stale(payer: str, window: int) -> None:
    """One day over the payer's window is stale, and each payer uses its own window."""
    assert _gaps(_complete(payer, lab_age=window + 1), payer) == [("recent_lab", "stale")]


def test_stale_detail_states_the_age_and_the_window() -> None:
    """The reason text is concrete enough to act on."""
    (gap,) = check_packet(
        _complete(lab_age=31), lookup_payer_checklist("payer_a"), _order("payer_a")
    )
    assert "31" in gap.detail and "30" in gap.detail


def test_lab_on_the_order_date_is_fine() -> None:
    """A lab drawn the same day as the order is age 0, which is recent."""
    assert _gaps(_complete(lab_age=0)) == []


def test_lab_dated_after_the_order_is_conflicting() -> None:
    """A lab from the future relative to the order does not add up."""
    assert _gaps(_complete(lab_age=-1)) == [("recent_lab", "conflicting")]


def test_found_lab_without_a_date_is_conflicting() -> None:
    """Recency cannot be judged without a date, so a person has to look."""
    assert _gaps(_change(_complete(), "recent_lab", date=None)) == [("recent_lab", "conflicting")]


def test_lab_not_found_is_missing_not_stale() -> None:
    """No lab evidence at all is missing, and there is no date to call stale."""
    fields = _change(_complete(), "recent_lab", found=False, value=None, quote=None, date=None)
    assert _gaps(fields) == [("recent_lab", "missing")]


def test_conflicting_lab_evidence_wins_over_stale() -> None:
    """When labs contradict each other, review beats a stale verdict."""
    fields = _change(_complete(lab_age=90), "recent_lab", conflicting_quotes=["CBC repeat"])
    assert _gaps(fields) == [("recent_lab", "conflicting")]


def test_lab_item_is_found_by_kind_not_by_name() -> None:
    """The recency rule applies to whatever item is kind=lab, whatever it is called."""
    checklist = PayerChecklist(
        payer_id="payer_a",
        lab_window_days=7,
        items=[ChecklistItem(name="cbc", description="Blood count", kind="lab")],
    )
    old = ExtractedField(
        name="cbc",
        value="CBC",
        quote="CBC drawn",
        found=True,
        date=ORDER_DATE - datetime.timedelta(days=8),
    )
    assert [(g.item, g.reason) for g in check_packet([old], checklist, _order())] == [
        ("cbc", "stale")
    ]


# --- treatment plan versus the order ---------------------------------------


def test_plan_with_a_different_drug_is_conflicting() -> None:
    """The note's plan names a drug other than the one on the order."""
    fields = _change(_complete(), "treatment_plan", quote="Start Samplimab 200 mg IV every 21 days.")
    assert _gaps(fields) == [("treatment_plan", "conflicting")]


def test_plan_with_a_different_dose_is_conflicting() -> None:
    """Same drug, different dose (the hand_09 shape): conflicting."""
    fields = _change(_complete(), "treatment_plan", quote="Start Examplumab 400 mg IV every 21 days.")
    assert _gaps(fields) == [("treatment_plan", "conflicting")]


@pytest.mark.parametrize(
    "quote",
    [
        "Start Examplumab 200mg IV every 21 days.",
        "start EXAMPLUMAB 200 MG IV every 21 days.",
        "Start Examplumab 200  mg IV every 21 days.",
    ],
)
def test_spacing_and_case_differences_are_not_conflicts(quote: str) -> None:
    """200mg and 200 mg are the same dose; drug case does not matter."""
    assert _gaps(_change(_complete(), "treatment_plan", quote=quote)) == []


@pytest.mark.parametrize("dose_in_note", ["1200 mg", "2200 mg", "0.200 mg"])
def test_dose_must_match_as_a_whole_number(dose_in_note: str) -> None:
    """200 mg must not be satisfied by 1200 mg, 2200 mg or 0.200 mg."""
    fields = _change(
        _complete(), "treatment_plan", quote=f"Start Examplumab {dose_in_note} IV every 21 days."
    )
    assert _gaps(fields) == [("treatment_plan", "conflicting")]


def test_schedule_is_not_compared() -> None:
    """Free-text schedules are too brittle to compare, so a different one is not flagged."""
    fields = _change(_complete(), "treatment_plan", quote="Start Examplumab 200 mg IV every 28 days.")
    assert _gaps(fields) == []


def test_plan_not_found_is_only_missing() -> None:
    """With no plan evidence there is nothing to compare, so no conflict is added."""
    fields = _change(_complete(), "treatment_plan", found=False, value=None, quote=None)
    assert _gaps(fields) == [("treatment_plan", "missing")]


def test_checklist_without_a_plan_item_skips_the_comparison() -> None:
    """The order comparison only applies when the payer requires a treatment plan."""
    checklist = PayerChecklist(
        payer_id="payer_a",
        lab_window_days=30,
        items=[ChecklistItem(name="stage", description="Stage", kind="field")],
    )
    stage = ExtractedField(name="stage", value="II", quote="Stage II", found=True)
    wrong_plan = ExtractedField(name="treatment_plan", value="x", quote="Other drug 1 mg", found=True)
    assert check_packet([stage, wrong_plan], checklist, _order()) == []


# --- guards and purity ------------------------------------------------------


def test_order_and_checklist_for_different_payers_are_refused() -> None:
    """Checking an order against another payer's rules is a bug, not a result."""
    with pytest.raises(ValueError, match="payer"):
        check_packet(_complete("payer_a"), lookup_payer_checklist("payer_a"), _order("payer_b"))


def test_inputs_are_not_modified() -> None:
    """check_packet is pure."""
    fields = _change(_complete(), "stage", conflicting_quotes=["Stage IV"])
    snapshot = [f.model_copy(deep=True) for f in fields]
    check_packet(fields, lookup_payer_checklist("payer_a"), _order())
    assert fields == snapshot


def test_gap_details_never_contain_note_quotes() -> None:
    """Gap text goes toward the Outreach agent, so it must not repeat quoted note text."""
    secret_quotes = ["SECRET PLAN TEXT", "SECRET CONFLICT TEXT"]
    fields = _change(_complete(), "treatment_plan", quote=secret_quotes[0])
    fields = _change(fields, "stage", conflicting_quotes=[secret_quotes[1]])
    gaps = check_packet(fields, lookup_payer_checklist("payer_a"), _order())
    assert len(gaps) == 2
    assert all(isinstance(g, Gap) for g in gaps)
    assert not any(q in g.detail for g in gaps for q in secret_quotes)


# --- route ------------------------------------------------------------------


def _gap(reason: str) -> Gap:
    return Gap(item="x", reason=reason, detail="d")  # type: ignore[arg-type]


def test_route_no_gaps_is_ready() -> None:
    assert route([]) == "READY"


@pytest.mark.parametrize("reasons", [["missing"], ["stale"], ["missing", "stale"]])
def test_route_missing_or_stale_is_not_ready(reasons: list[str]) -> None:
    assert route([_gap(r) for r in reasons]) == "NOT_READY"


@pytest.mark.parametrize("reasons", [["conflicting"], ["missing", "conflicting", "stale"]])
def test_route_any_conflict_needs_review(reasons: list[str]) -> None:
    """One conflicting gap sends the whole packet to a person, even alongside missing items."""
    assert route([_gap(r) for r in reasons]) == "NEEDS_REVIEW"


# --- the two tools together on real hand-labeled cases ----------------------

LINE_PREFIXES = {
    "diagnosis": ("Diagnosis:",),
    "stage": ("Stage:", "Staging:"),
    "biomarker": ("Biomarkers",),  # hand_03 writes "Biomarkers (recurrence biopsy):"
    "prior_therapies": ("Prior therapy:",),
    "recent_lab": ("Labs:",),
    "treatment_plan": ("Plan:",),
}
# What a careful extractor would decide that a line-prefix rule cannot.
NOT_FOUND = {"hand_10": {"stage"}}  # "stage has not yet been assigned"
CONFLICTS = {"hand_08": {"stage": ["stage IV non-small cell lung cancer"]}}


def _fields_from_note(case_id: str, note: str) -> list[ExtractedField]:
    """Quote whole lines from a hand-written note, as a perfect extractor would."""
    fields = []
    for name, prefixes in LINE_PREFIXES.items():
        line = next((l for l in note.splitlines() if l.startswith(prefixes)), None)
        if line is None or name in NOT_FOUND.get(case_id, set()):
            fields.append(ExtractedField(name=name, value=None, quote=None, found=False))
            continue
        date = None
        if name == "recent_lab":
            date = datetime.date.fromisoformat(re.search(r"\d{4}-\d{2}-\d{2}", line).group())
        fields.append(
            ExtractedField(
                name=name,
                value=line,
                quote=line,
                found=True,
                date=date,
                conflicting_quotes=CONFLICTS.get(case_id, {}).get(name, []),
            )
        )
    return fields


@pytest.mark.parametrize(
    "case_id",
    ["hand_01", "hand_03", "hand_04", "hand_05", "hand_06", "hand_07", "hand_08", "hand_09", "hand_10"],
)
def test_validate_then_check_matches_the_hand_labels(case_id: str) -> None:
    """With a perfect extractor, the code-only half reproduces the human labels."""
    folder = ROOT / "data" / "cases" / case_id
    note = (folder / "note.txt").read_text()
    order = Order.model_validate_json((folder / "order.json").read_text())
    label = json.loads((ROOT / "data" / "labels.json").read_text())[case_id]

    fields = validate_citations(note, _fields_from_note(case_id, note))
    gaps = check_packet(fields, lookup_payer_checklist(order.payer_id), order)

    assert {(g.item, g.reason) for g in gaps} == {
        (g["item"], g["reason"]) for g in label["expected_gaps"]
    }
    assert route(gaps) == label["expected_status"]


def test_missing_gap_text_names_the_item_in_plain_words() -> None:
    """The reader (and later the Outreach agent) sees which item is missing, not an awkward sentence."""
    (gap,) = check_packet(
        _drop(_complete("payer_b"), "prior_therapies"), lookup_payer_checklist("payer_b"), _order("payer_b")
    )
    assert gap.detail == "Not documented in the note: Prior therapies tried, with outcomes."
