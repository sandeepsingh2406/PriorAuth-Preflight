"""Tests for the seeded synthetic case generator (scripts/gen_cases.py)."""
import datetime
import json
import re
from pathlib import Path

import pytest
import yaml

from scripts.gen_cases import SEED, GeneratedCase, generate_cases, write_cases
from src.models import Order, PayerChecklist

ROOT = Path(__file__).parent.parent
LAB_DATE = re.compile(r"CBC with differential (\d{4}-\d{2}-\d{2})")
AMBIGUITY_KINDS = {
    "conflicting_stage",
    "conflicting_dose",
    "conflicting_lab",
    "unreadable_biomarker",
    "unreadable_lab_date",
}
HAND_ENTRY = {
    "payer_id": "payer_b",
    "expected_status": "NOT_READY",
    "expected_gaps": [{"item": "stage", "reason": "missing"}],
    "ambiguous": False,
    "hand_labeled": True,
}


@pytest.fixture(scope="module")
def cases() -> list[GeneratedCase]:
    return generate_cases()


@pytest.fixture(scope="module")
def checklists() -> dict[str, PayerChecklist]:
    raw = yaml.safe_load((ROOT / "data" / "checklists.yaml").read_text())
    return {p["payer_id"]: PayerChecklist.model_validate(p) for p in raw["payers"]}


def _write(tmp_path: Path, labels: dict | None = None) -> tuple[Path, Path]:
    """Run the writer into a temp folder, optionally starting from existing labels."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(exist_ok=True)
    labels_path = tmp_path / "labels.json"
    if labels is not None:
        labels_path.write_text(json.dumps(labels))
    write_cases(generate_cases(), cases_dir, labels_path)
    return cases_dir, labels_path


def _snapshot(folder: Path) -> dict[str, bytes]:
    """Map every file under a folder to its bytes, to compare two runs."""
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_generates_thirty_cases_with_sequential_ids(cases: list[GeneratedCase]) -> None:
    """About 30 cases, named gen_001 to gen_030 so they never clash with hand_ cases."""
    assert [c.case_id for c in cases] == [f"gen_{i:03d}" for i in range(1, 31)]


def test_same_seed_gives_identical_output(cases: list[GeneratedCase]) -> None:
    """A fixed seed means regenerating always produces the same cases."""
    assert generate_cases(SEED) == cases


def test_different_seed_gives_different_output(cases: list[GeneratedCase]) -> None:
    """The seed actually drives the content (guards against an ignored seed)."""
    assert generate_cases(SEED + 1) != cases


def test_orders_are_valid_and_match_their_case(cases: list[GeneratedCase]) -> None:
    """Every order.json parses as an Order with the same case id and payer as its label."""
    for case in cases:
        order = Order.model_validate(case.order)
        assert order.case_id == case.case_id
        assert order.payer_id == case.label["payer_id"]


def test_five_ambiguous_cases_of_distinct_kinds(cases: list[GeneratedCase]) -> None:
    """Exactly five ambiguous cases, one per kind, all routed to NEEDS_REVIEW."""
    ambiguous = [c for c in cases if c.label["ambiguous"]]
    assert len(ambiguous) == 5
    assert {c.label["ambiguity_kind"] for c in ambiguous} == AMBIGUITY_KINDS
    assert all(c.label["expected_status"] == "NEEDS_REVIEW" for c in ambiguous)


def test_expected_gaps_only_name_items_the_payer_requires(
    cases: list[GeneratedCase], checklists: dict[str, PayerChecklist]
) -> None:
    """A label never expects a gap for an item the payer does not require."""
    for case in cases:
        required = {i.name for i in checklists[case.label["payer_id"]].items}
        for gap in case.label["expected_gaps"]:
            assert gap["item"] in required, case.case_id
            assert gap["reason"] in {"missing", "stale", "conflicting"}


def test_status_matches_expected_gaps(cases: list[GeneratedCase]) -> None:
    """Non-ambiguous cases are READY with no gaps and NOT_READY with at least one."""
    for case in cases:
        if case.label["ambiguous"]:
            continue
        gaps = case.label["expected_gaps"]
        assert case.label["expected_status"] == ("NOT_READY" if gaps else "READY")


def test_every_seeded_gap_type_and_some_clean_cases_appear(
    cases: list[GeneratedCase],
) -> None:
    """Coverage: each seeded gap type, a two-gap case, and complete cases all exist."""
    seen = {(g["item"], g["reason"]) for c in cases for g in c.label["expected_gaps"]}
    assert {
        ("stage", "missing"),
        ("recent_lab", "stale"),
        ("prior_therapies", "missing"),
        ("treatment_plan", "missing"),
    } <= seen
    assert any(len(c.label["expected_gaps"]) >= 2 for c in cases)
    assert sum(c.label["expected_status"] == "READY" for c in cases) >= 4


def test_stale_labels_agree_with_the_note_dates(
    cases: list[GeneratedCase], checklists: dict[str, PayerChecklist]
) -> None:
    """The lab date in the note is stale for the payer window exactly when the label says so."""
    for case in cases:
        dates = LAB_DATE.findall(case.note)
        if not dates:  # the unreadable-date case has no parseable date
            continue
        order_date = datetime.date.fromisoformat(case.order["order_date"])
        window = checklists[case.label["payer_id"]].lab_window_days
        ages = [(order_date - datetime.date.fromisoformat(d)).days for d in dates]
        stale = {"item": "recent_lab", "reason": "stale"} in case.label["expected_gaps"]
        if stale:
            assert all(a > window for a in ages), case.case_id
        else:
            assert all(0 <= a <= window for a in ages), case.case_id


def test_missing_sections_are_really_absent_from_the_note(
    cases: list[GeneratedCase],
) -> None:
    """A label says an item is missing exactly when the note has no section for it."""
    for case in cases:
        missing = {g["item"] for g in case.label["expected_gaps"] if g["reason"] == "missing"}
        assert ("\nSTAGE\n" not in case.note) == ("stage" in missing), case.case_id
        assert ("\nPRIOR THERAPIES\n" not in case.note) == ("prior_therapies" in missing)
        assert ("Start " not in case.note) == ("treatment_plan" in missing), case.case_id


def test_notes_are_marked_synthetic(cases: list[GeneratedCase]) -> None:
    """Every note says it is fictional and uses an obviously fake SYN- record number."""
    for case in cases:
        assert case.note.startswith("SYNTHETIC NOTE")
        assert re.search(r"MRN: SYN-\d{5}", case.note)


def test_write_creates_case_folders_and_labels(tmp_path: Path) -> None:
    """Each case gets order.json and note.txt, and labels.json lists all 30 cases."""
    cases_dir, labels_path = _write(tmp_path)
    assert sorted(p.name for p in cases_dir.iterdir()) == [f"gen_{i:03d}" for i in range(1, 31)]
    assert (cases_dir / "gen_001" / "order.json").is_file()
    assert (cases_dir / "gen_001" / "note.txt").is_file()
    assert len(json.loads(labels_path.read_text())) == 30


def test_write_is_idempotent(tmp_path: Path) -> None:
    """Running twice leaves every file byte-for-byte identical."""
    cases_dir, labels_path = _write(tmp_path)
    first = _snapshot(tmp_path)
    _write(tmp_path)
    assert _snapshot(tmp_path) == first


def test_hand_labeled_entries_and_folders_are_untouched(tmp_path: Path) -> None:
    """A hand_labeled entry and its hand_ folder survive a run unchanged."""
    hand_dir = tmp_path / "cases" / "hand_01"
    hand_dir.mkdir(parents=True)
    (hand_dir / "note.txt").write_text("written by hand")
    cases_dir, labels_path = _write(tmp_path, {"hand_01": HAND_ENTRY})
    labels = json.loads(labels_path.read_text())
    assert labels["hand_01"] == HAND_ENTRY
    assert len(labels) == 31
    assert (hand_dir / "note.txt").read_text() == "written by hand"
    assert sorted(p.name for p in hand_dir.iterdir()) == ["note.txt"]


def test_other_non_generated_entries_are_kept(tmp_path: Path) -> None:
    """An in-progress hand_ entry (not yet hand_labeled) is not deleted either."""
    draft = {**HAND_ENTRY, "hand_labeled": False}
    _, labels_path = _write(tmp_path, {"hand_02": draft})
    assert json.loads(labels_path.read_text())["hand_02"] == draft


def test_stale_generated_entries_are_replaced(tmp_path: Path, cases: list[GeneratedCase]) -> None:
    """An old gen_ entry that is not hand_labeled is overwritten by the fresh label."""
    stale = {**HAND_ENTRY, "hand_labeled": False, "expected_status": "READY"}
    _, labels_path = _write(tmp_path, {"gen_001": stale})
    assert json.loads(labels_path.read_text())["gen_001"] == cases[0].label


def test_hand_labeled_gen_id_aborts_before_writing(tmp_path: Path) -> None:
    """If a hand_labeled entry collides with a generated id, abort and write nothing."""
    labels = {"gen_001": HAND_ENTRY}
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="hand_labeled"):
        write_cases(generate_cases(), cases_dir, labels_path)
    assert list(cases_dir.iterdir()) == []
    assert json.loads(labels_path.read_text()) == labels


def test_committed_data_matches_the_generator(cases: list[GeneratedCase]) -> None:
    """data/cases and the gen_ part of data/labels.json are exactly what the generator makes."""
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    for case in cases:
        folder = ROOT / "data" / "cases" / case.case_id
        assert json.loads((folder / "order.json").read_text()) == case.order
        assert (folder / "note.txt").read_text() == case.note
        assert labels[case.case_id] == case.label
