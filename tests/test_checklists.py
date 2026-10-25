"""Tests for the invented payer checklists in data/checklists.yaml."""
from pathlib import Path

import pytest
import yaml

from src.models import PayerChecklist

CHECKLISTS_PATH = Path(__file__).parent.parent / "data" / "checklists.yaml"


@pytest.fixture(scope="module")
def checklists() -> dict[str, PayerChecklist]:
    """Load the YAML and validate every payer, keyed by payer_id."""
    raw = yaml.safe_load(CHECKLISTS_PATH.read_text())
    payers = [PayerChecklist.model_validate(p) for p in raw["payers"]]
    return {p.payer_id: p for p in payers}


def test_file_has_three_invented_payers(checklists: dict[str, PayerChecklist]) -> None:
    """There are exactly three payers, with generic ids (no real payer names)."""
    assert sorted(checklists) == ["payer_a", "payer_b", "payer_c"]


def test_payer_ids_are_unique() -> None:
    """No payer_id appears twice, which would silently drop one in a dict."""
    raw = yaml.safe_load(CHECKLISTS_PATH.read_text())
    ids = [p["payer_id"] for p in raw["payers"]]
    assert len(ids) == len(set(ids))


def test_lab_windows_differ_per_payer(checklists: dict[str, PayerChecklist]) -> None:
    """Payers A, B and C use 30, 45 and 60 day lab recency windows."""
    windows = {pid: c.lab_window_days for pid, c in checklists.items()}
    assert windows == {"payer_a": 30, "payer_b": 45, "payer_c": 60}


def test_each_payer_has_exactly_one_lab_item(
    checklists: dict[str, PayerChecklist],
) -> None:
    """The recency rule needs one unambiguous lab item per payer to check."""
    for checklist in checklists.values():
        labs = [i for i in checklist.items if i.kind == "lab"]
        assert len(labs) == 1, checklist.payer_id


def test_item_names_unique_within_payer(checklists: dict[str, PayerChecklist]) -> None:
    """Item names identify gaps, so they cannot repeat inside one payer."""
    for checklist in checklists.values():
        names = [i.name for i in checklist.items]
        assert len(names) == len(set(names)), checklist.payer_id


def test_required_items_differ_between_payers(
    checklists: dict[str, PayerChecklist],
) -> None:
    """Payer B also wants prior therapies; Payer C does not require stage."""
    names = {pid: {i.name for i in c.items} for pid, c in checklists.items()}
    assert "prior_therapies" not in names["payer_a"]
    assert "prior_therapies" in names["payer_b"]
    assert "stage" in names["payer_a"] and "stage" in names["payer_b"]
    assert "stage" not in names["payer_c"]


def test_every_payer_requires_core_items(
    checklists: dict[str, PayerChecklist],
) -> None:
    """All payers require a diagnosis, a biomarker, a recent lab and a treatment plan."""
    core = {"diagnosis", "biomarker", "recent_lab", "treatment_plan"}
    for checklist in checklists.values():
        assert core <= {i.name for i in checklist.items}, checklist.payer_id
