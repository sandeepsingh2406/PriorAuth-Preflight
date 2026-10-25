"""Tests for lookup_payer_checklist in src/tools.py."""
from pathlib import Path

import pytest

from src.models import PayerChecklist
from src.tools import lookup_payer_checklist


def _yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "checklists.yaml"
    path.write_text(text)
    return path


ONE_PAYER = """
payers:
  - payer_id: payer_x
    lab_window_days: 10
    items:
      - {name: stage, description: Cancer stage, kind: field}
      - {name: recent_lab, description: Recent lab, kind: lab}
"""


@pytest.mark.parametrize(
    ("payer_id", "window"), [("payer_a", 30), ("payer_b", 45), ("payer_c", 60)]
)
def test_returns_validated_checklist_with_payer_window(payer_id: str, window: int) -> None:
    """Each real payer comes back as a PayerChecklist with its own lab window."""
    checklist = lookup_payer_checklist(payer_id)
    assert isinstance(checklist, PayerChecklist)
    assert checklist.payer_id == payer_id
    assert checklist.lab_window_days == window
    assert checklist.items


def test_unknown_payer_raises_value_error_naming_the_id() -> None:
    """An unknown payer fails loudly and says which id was asked for."""
    with pytest.raises(ValueError, match="payer_z"):
        lookup_payer_checklist("payer_z")


@pytest.mark.parametrize("payer_id", ["PAYER_A", "Payer A", " payer_a", ""])
def test_payer_id_must_match_exactly(payer_id: str) -> None:
    """No guessing: wrong case, spaces, or an empty id are unknown payers."""
    with pytest.raises(ValueError):
        lookup_payer_checklist(payer_id)


def test_default_path_works_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default file is found relative to the code, not the current directory."""
    monkeypatch.chdir(tmp_path)
    assert lookup_payer_checklist("payer_a").payer_id == "payer_a"


def test_reads_from_a_custom_path(tmp_path: Path) -> None:
    """A checklist file can be passed in, which keeps the tests independent of real data."""
    checklist = lookup_payer_checklist("payer_x", path=_yaml(tmp_path, ONE_PAYER))
    assert checklist.lab_window_days == 10
    assert [i.name for i in checklist.items] == ["stage", "recent_lab"]


def test_each_lookup_returns_an_independent_object() -> None:
    """Changing one result cannot leak into the next lookup."""
    first = lookup_payer_checklist("payer_a")
    first.items.clear()
    assert lookup_payer_checklist("payer_a").items


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """A wrong path is a clear error, not an empty checklist."""
    with pytest.raises(FileNotFoundError):
        lookup_payer_checklist("payer_a", path=tmp_path / "nope.yaml")


@pytest.mark.parametrize("text", ["", "payers: not-a-list", "- just\n- a list\n", "other: 1"])
def test_file_without_a_payers_list_is_rejected(tmp_path: Path, text: str) -> None:
    """A file that is not shaped like the checklists is an error, not a silent miss."""
    with pytest.raises(ValueError, match="payers"):
        lookup_payer_checklist("payer_a", path=_yaml(tmp_path, text))


def test_duplicate_payer_ids_are_rejected(tmp_path: Path) -> None:
    """Two entries with one id would be ambiguous, so the file is refused."""
    twice = ONE_PAYER + ONE_PAYER.replace("payers:\n", "")
    with pytest.raises(ValueError, match="duplicate"):
        lookup_payer_checklist("payer_x", path=_yaml(tmp_path, twice))


def test_invalid_entry_is_rejected(tmp_path: Path) -> None:
    """An entry that breaks the model (a 0 day window) fails validation."""
    bad = ONE_PAYER.replace("lab_window_days: 10", "lab_window_days: 0")
    with pytest.raises(ValueError):
        lookup_payer_checklist("payer_x", path=_yaml(tmp_path, bad))
