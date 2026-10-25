"""Seeded generator for synthetic prior authorization cases.

Writes data/cases/gen_NNN/{order.json,note.txt} and the gen_ entries of
data/labels.json. Everything is fictional. It never touches hand_ cases or any
label entry it does not own.

Run: uv run python scripts/gen_cases.py
"""
import datetime
import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

SEED = 20261024
ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "data" / "cases"
LABELS_PATH = ROOT / "data" / "labels.json"
CHECKLISTS_PATH = ROOT / "data" / "checklists.yaml"
ID_PREFIX = "gen_"

# Seeded gap codes -> the (item, reason) a label expects for each.
GAP_LABELS = {
    "stage": ("stage", "missing"),
    "prior": ("prior_therapies", "missing"),
    "plan": ("treatment_plan", "missing"),
    "stale_lab": ("recent_lab", "stale"),
}

# (payer, seeded gaps, ambiguity kind). Each payer must require the item it is seeded to miss.
SCENARIOS: list[tuple[str, tuple[str, ...], str | None]] = [
    ("payer_a", (), None), ("payer_b", (), None), ("payer_c", (), None),
    ("payer_a", (), None), ("payer_b", (), None), ("payer_c", (), None),
    ("payer_a", ("stage",), None), ("payer_b", ("stage",), None),
    ("payer_a", ("stage",), None), ("payer_b", ("stage",), None),
    ("payer_a", ("stale_lab",), None), ("payer_b", ("stale_lab",), None),
    ("payer_c", ("stale_lab",), None), ("payer_a", ("stale_lab",), None),
    ("payer_b", ("prior",), None), ("payer_c", ("prior",), None),
    ("payer_b", ("prior",), None), ("payer_c", ("prior",), None),
    ("payer_a", ("plan",), None), ("payer_b", ("plan",), None), ("payer_c", ("plan",), None),
    ("payer_a", ("stage", "stale_lab"), None), ("payer_b", ("stale_lab", "plan"), None),
    ("payer_c", ("prior", "plan"), None), ("payer_b", ("stage", "prior"), None),
    ("payer_a", (), "conflicting_stage"), ("payer_b", (), "conflicting_dose"),
    ("payer_c", (), "conflicting_lab"), ("payer_a", (), "unreadable_biomarker"),
    ("payer_b", (), "unreadable_lab_date"),
]

# Items an ambiguous case is expected to flag. Only the status is scored for these.
AMBIGUITY_GAPS = {
    "conflicting_stage": [("stage", "conflicting")],
    "conflicting_dose": [("treatment_plan", "conflicting")],
    "conflicting_lab": [("recent_lab", "conflicting")],
    "unreadable_biomarker": [],
    "unreadable_lab_date": [("recent_lab", "conflicting")],
}

DIAGNOSES = [
    ("Non-small cell lung cancer", "C34.90", "EGFR exon 19 deletion: detected"),
    ("Breast cancer", "C50.919", "HER2 IHC 3+, positive"),
    ("Colorectal cancer", "C18.9", "KRAS G12C mutation: detected"),
    ("Melanoma", "C43.9", "BRAF V600E mutation: detected"),
]
STAGES = ["Stage IIA", "Stage IIB", "Stage IIIA", "Stage IIIB", "Stage IV"]
PRIOR_THERAPIES = [
    "Fictoplatin plus Examplexed, 4 cycles, best response partial response.",
    "Mockorubicin, 6 cycles, best response stable disease.",
    "Samplemab, 8 cycles, best response progressive disease.",
]
DRUGS = [
    ("Examplumab", "200 mg", "every 21 days"),
    ("Fictinib", "80 mg", "once daily"),
    ("Mocktuzumab", "6 mg/kg", "every 3 weeks"),
    ("Samplestat", "150 mg", "twice daily"),
]
FIRST_NAMES = ["Avery", "Jordan", "Riley", "Casey", "Morgan", "Quinn", "Rowan", "Sage"]
LAST_NAMES = ["Testcase", "Placeholder", "Sampleton", "Mockwell", "Fakeman", "Demoson"]


@dataclass
class GeneratedCase:
    """One synthetic case: the order, the note, and its expected label."""

    case_id: str
    order: dict
    note: str
    label: dict


def _lab_windows() -> dict[str, int]:
    """Lab recency window per payer, read from the checklists so they cannot drift."""
    raw = yaml.safe_load(CHECKLISTS_PATH.read_text())
    return {p["payer_id"]: p["lab_window_days"] for p in raw["payers"]}


def _cbc_line(date_text: str, rng: random.Random) -> str:
    anc = round(rng.uniform(1.2, 4.5), 1)
    hgb = round(rng.uniform(9.5, 14.0), 1)
    plt = rng.randint(150, 400)
    return f"CBC with differential {date_text}: ANC {anc}, hemoglobin {hgb}, platelets {plt}"


def _build_case(
    index: int,
    payer: str,
    gaps: tuple[str, ...],
    ambiguity: str | None,
    windows: dict[str, int],
    rng: random.Random,
) -> GeneratedCase:
    """Build the order, note and label for one scenario row."""
    case_id = f"{ID_PREFIX}{index:03d}"
    window = windows[payer]
    order_date = datetime.date(2026, 2, 1) + datetime.timedelta(days=rng.randint(0, 240))
    visit_date = order_date - datetime.timedelta(days=rng.randint(0, 3))
    cancer, icd10, biomarker = rng.choice(DIAGNOSES)
    stage = rng.choice(STAGES)
    drug, dose, schedule = rng.choice(DRUGS)
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

    age_days = window + rng.randint(5, 40) if "stale_lab" in gaps else rng.randint(3, window - 2)
    lab_date = (order_date - datetime.timedelta(days=age_days)).isoformat()
    lab_lines = [_cbc_line(lab_date, rng)]
    stage_lines = [stage]
    biomarker_lines = [biomarker]
    plan_lines = [f"Start {drug} {dose} {schedule}."]

    if ambiguity == "conflicting_stage":
        other = rng.choice([s for s in STAGES if s != stage])
        stage_lines = [f"{stage} per initial staging.", f"Updated imaging report lists {other}."]
    elif ambiguity == "conflicting_dose":
        other_dose = "120 mg" if dose != "120 mg" else "250 mg"
        plan_lines.append(f"Pharmacy call: dose clarified as {other_dose} {schedule}.")
    elif ambiguity == "conflicting_lab":
        second = (order_date - datetime.timedelta(days=rng.randint(1, 2))).isoformat()
        lab_lines = [_cbc_line(lab_date, rng), _cbc_line(second, rng) + " (repeat, different lab)"]
    elif ambiguity == "unreadable_biomarker":
        biomarker_lines = ["[scan illegible] EG?R ex@n 1# d%l%tion ?? res_lt: [unreadable]"]
    elif ambiguity == "unreadable_lab_date":
        lab_lines = [_cbc_line("2026-0?-?? (date smudged)", rng)]

    if "plan" in gaps:
        plan_lines = ["Treatment options to be discussed at the next visit."]

    sections = [
        "SYNTHETIC NOTE (fictional patient, not real)",
        f"Patient: {name}  MRN: SYN-{index:05d}",
        f"Visit date: {visit_date.isoformat()}",
        "Ordering provider: Dr. Example Provider",
        "",
        "DIAGNOSIS",
        f"{cancer}, ICD-10 {icd10}. Diagnosis confirmed by biopsy.",
    ]
    if "stage" not in gaps:
        sections += ["", "STAGE", *stage_lines]
    sections += ["", "BIOMARKERS", *biomarker_lines]
    if "prior" not in gaps:
        sections += ["", "PRIOR THERAPIES", rng.choice(PRIOR_THERAPIES)]
    sections += ["", "LABS", *lab_lines, "", "PLAN", *plan_lines]
    note = "\n".join(sections) + "\n"

    if ambiguity:
        expected = AMBIGUITY_GAPS[ambiguity]
        status = "NEEDS_REVIEW"
    else:
        expected = [GAP_LABELS[g] for g in gaps]
        status = "NOT_READY" if expected else "READY"
    order = {
        "case_id": case_id,
        "payer_id": payer,
        "order_date": order_date.isoformat(),
        "drug": drug,
        "dose": dose,
        "schedule": schedule,
    }
    label = {
        "payer_id": payer,
        "expected_status": status,
        "expected_gaps": [{"item": item, "reason": reason} for item, reason in expected],
        "ambiguous": ambiguity is not None,
        "ambiguity_kind": ambiguity,
        "hand_labeled": False,
    }
    return GeneratedCase(case_id, order, note, label)


def generate_cases(seed: int = SEED) -> list[GeneratedCase]:
    """Build all synthetic cases in memory. The same seed always gives the same cases."""
    rng = random.Random(seed)
    windows = _lab_windows()
    rows = list(SCENARIOS)
    rng.shuffle(rows)
    return [
        _build_case(i, payer, gaps, ambiguity, windows, rng)
        for i, (payer, gaps, ambiguity) in enumerate(rows, start=1)
    ]


def write_cases(cases: list[GeneratedCase], cases_dir: Path, labels_path: Path) -> None:
    """Write case folders and merge labels, leaving everything this script does not own alone."""
    existing = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    new_ids = {c.case_id for c in cases}
    for case_id, entry in existing.items():
        if case_id in new_ids and entry.get("hand_labeled") is True:
            raise ValueError(f"{case_id} is hand_labeled; refusing to overwrite it")

    kept = {k: v for k, v in existing.items() if not k.startswith(ID_PREFIX)}
    for case in cases:
        folder = cases_dir / case.case_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "order.json").write_text(json.dumps(case.order, indent=2) + "\n")
        (folder / "note.txt").write_text(case.note)
        kept[case.case_id] = case.label
    labels_path.write_text(json.dumps(dict(sorted(kept.items())), indent=2) + "\n")


def main() -> None:
    cases = generate_cases()
    write_cases(cases, CASES_DIR, LABELS_PATH)
    print(f"Wrote {len(cases)} cases to {CASES_DIR} and labels to {LABELS_PATH}")


if __name__ == "__main__":
    main()
