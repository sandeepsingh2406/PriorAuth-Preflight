"""The README must agree with the evaluation it quotes, show the real data flow, and stay clean."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
RESULTS = json.loads((ROOT / "eval" / "results.json").read_text())
RESULTS_MD = (ROOT / "eval" / "results.md").read_text()

# README row label -> (key in results.json measures, target text)
ROWS = {
    "Status accuracy": ("status_accuracy", "-"),
    "Gap recall": ("gap_recall", ">= 0.90"),
    "False alarm rate": ("false_alarm_rate", "<= 0.10"),
    "Escalation correctness": ("escalation_correctness", ">= 0.80"),
    "Citation validity (quotes in reports)": ("reported_citation_validity", "1.00"),
}


def section(text: str, heading: str) -> str:
    """The text under `## heading` up to the next `## ` heading."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"README has no '## {heading}' section"
    return match.group(1)


def flat(text: str) -> str:
    """The text with line wraps turned into single spaces, so a phrase can be found across a wrap."""
    return re.sub(r"\s+", " ", text)


def across_runs(group: str, key: str) -> str:
    """The value over all runs, as 1.00 when every run agrees and 0.60-0.80 when they differ."""
    values = [run["measures"][group][key] for run in RESULTS["runs"]]
    low, high = min(values), max(values)
    return f"{low:.2f}" if f"{low:.2f}" == f"{high:.2f}" else f"{low:.2f}-{high:.2f}"


def table_rows(text: str) -> dict[str, list[str]]:
    rows = {}
    for line in text.splitlines():
        if line.startswith("|") and not set(line) <= set("|- "):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


# --- results ----------------------------------------------------------------------------------------------


def test_the_results_table_matches_results_json_for_every_measure() -> None:
    rows = table_rows(section(README, "Results"))
    for label, (key, target) in ROWS.items():
        assert label in rows, label
        assert rows[label] == [target, across_runs("hand", key), across_runs("generated", key)], label


def test_the_results_table_headers_show_the_real_case_and_run_counts() -> None:
    text = section(README, "Results")
    runs = len(RESULTS["runs"])
    hand = RESULTS["runs"][0]["measures"]["hand"]["cases"]
    generated = RESULTS["runs"][0]["measures"]["generated"]["cases"]
    ambiguous = sum(1 for c in RESULTS["runs"][0]["cases"] if c["ambiguous"])
    assert f"Hand-labeled ({hand} cases)" in text and f"Generated ({generated} cases)" in text
    assert f"{hand + generated} synthetic cases" in text
    assert f"{ambiguous} ambiguous" in text
    assert f"{runs} runs" in text


def test_the_targets_in_the_readme_are_the_ones_the_evaluation_report_uses() -> None:
    report = table_rows(RESULTS_MD.split("## Baseline comparison")[0])
    for label, (_, target) in ROWS.items():
        if target != "-":
            assert report[label][0] == target, label


def test_the_cases_that_failed_in_every_run_are_named_in_the_readme() -> None:
    """Whatever the evaluation got wrong in all runs must be stated, so a new miss cannot hide."""
    wrong_in = {}
    for run in RESULTS["runs"]:
        for case in run["cases"]:
            if case["expected_status"] != case["actual_status"]:
                wrong_in.setdefault(case["case_id"], []).append(case["actual_status"])
    always_wrong = {case_id for case_id, seen in wrong_in.items() if len(seen) == len(RESULTS["runs"])}
    text = section(README, "Results")
    for case_id in always_wrong:
        assert case_id in text, case_id
    for case_id in wrong_in:  # and nothing is named that did not fail
        assert case_id in text, case_id
    named = set(re.findall(r"\b(?:gen|hand)_\d{2,3}\b", text))
    assert named == set(wrong_in)


def test_a_miss_in_the_unsafe_direction_is_called_out_as_such() -> None:
    text = section(README, "Results")
    unsafe = [
        c["case_id"]
        for c in RESULTS["runs"][0]["cases"]
        if c["expected_status"] == "NEEDS_REVIEW" and c["actual_status"] == "READY"
    ]
    assert unsafe, "this test expects the evaluation to contain a miss of that kind; update it if it was fixed"
    assert "unsafe direction" in flat(text)


def test_the_readme_says_the_set_is_small_and_synthetic_and_links_the_full_report() -> None:
    text = section(README, "Results").lower()
    assert "small" in text and "synthetic" in text
    assert "(eval/results.md)" in text
    assert (ROOT / "eval" / "results.md").exists()


def test_the_nuance_the_readme_repeats_is_in_the_evaluation_report() -> None:
    """The README says the model escalates the missed case about 2 times in 5; the report must say so too."""
    assert "2 of 5" in RESULTS_MD
    assert "2 of 5" in flat(section(README, "Results"))


def test_zero_automatic_sends_is_stated() -> None:
    assert "0 automatic" in section(README, "Results").lower() or "automatic sends: 0" in section(README, "Results").lower()


# --- the architecture diagram ------------------------------------------------------------------------------


def diagram() -> str:
    match = re.search(r"```mermaid\n(.*?)```", README, re.S)
    assert match, "README has no mermaid diagram"
    return match.group(1)


def edges() -> list[tuple[str, str]]:
    found = []
    for line in diagram().splitlines():
        if "-->" in line:
            ids = re.findall(r"\b[A-Z]{3,}\b", re.sub(r'"[^"]*"', "", line))  # node ids only, not label words
            found.append((ids[0], ids[-1]))
    return found


def test_the_diagram_shows_every_step_in_order() -> None:
    text = diagram().lower()
    for word in ["note", "extraction model", "quotes", "checklist", "recency", "plan", "status", "gap list",
                 "outreach model", "human", "audit log"]:
        assert word in text, word
    assert "flowchart" in text.splitlines()[0] or "graph" in text.splitlines()[0]


def test_in_the_diagram_the_outreach_model_gets_only_the_gap_list() -> None:
    """The note, the extraction step and the order must have no arrow into the Outreach model."""
    into_outreach = [src for src, dst in edges() if dst == "OUTREACH"]
    assert into_outreach == ["GAPS"]
    assert not any(dst == "GAPS" and src in {"NOTE", "EXTRACT", "ORDER"} for src, dst in edges())


def test_in_the_diagram_only_code_decides_the_status() -> None:
    into_status = [src for src, dst in edges() if dst == "STATUS"]
    assert into_status == ["CHECKS"]


def test_in_the_diagram_the_person_reviews_before_anything_ends_up_in_the_log() -> None:
    assert ("HUMAN", "AUDIT") in edges()
    assert not any(src == "OUTREACH" and dst not in {"HUMAN"} for src, dst in edges())  # nothing leaves the draft but a person


def test_the_headline_sentences_are_there() -> None:
    assert "model reads and drafts; code decides and verifies" in README
    assert "only the gap list" in README or "gap list only" in README


# --- the quick start ----------------------------------------------------------------------------------------


def quick_start() -> str:
    return section(README, "Quick start")


def test_the_quick_start_begins_with_a_clone_and_names_a_python_and_uv_requirement() -> None:
    text = quick_start()
    assert "git clone https://github.com/sandeepsingh2406/PriorAuth-Preflight.git" in text
    assert "cd PriorAuth-Preflight" in text
    assert "Python 3.12" in text and "uv" in text


def test_every_file_the_quick_start_runs_exists_in_the_repo() -> None:
    text = quick_start()
    for path in re.findall(r"(?:python|cp) ((?:[\w./-]+/)?[\w.-]+\.(?:py|example))", text):
        assert (ROOT / path).exists(), path
    assert (ROOT / "env.example").exists()


def test_the_quick_start_case_exists_and_the_modules_exist() -> None:
    text = quick_start()
    case = re.search(r"--case (\w+)", text).group(1)
    assert (ROOT / "data" / "cases" / f"{case}.note.txt").exists() or any((ROOT / "data" / "cases").glob(f"{case}*"))
    for module in re.findall(r"-m (src\.\w+)", text):
        assert (ROOT / (module.replace(".", "/") + ".py")).exists(), module


def test_the_variables_the_quick_start_tells_you_to_set_are_in_env_example() -> None:
    example = (ROOT / "env.example").read_text()
    for name in ["OPENAI_API_KEY", "MODEL_DEV", "MODEL_DEMO"]:
        assert name in quick_start() and name in example, name


def test_the_quick_start_never_puts_a_key_on_a_command_line() -> None:
    assert not re.search(r"OPENAI_API_KEY=\S", quick_start())
    assert "sk-" not in README


# --- what must not be in the public docs ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["README.md", "SPEC.md"])
def test_public_docs_do_not_mention_the_dropped_hosting_experiment(name: str) -> None:
    text = (ROOT / name).read_text().lower()
    for word in ["hugging face", "huggingface", "hf.co"]:
        assert word not in text, (name, word)


def test_the_limits_section_is_still_there() -> None:
    text = section(README, "Limits")
    for phrase in ["Synthetic data only", "Illustrative checklists", "Documentation completeness only"]:
        assert phrase in text, phrase
