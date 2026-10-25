"""Evaluation measures (SPEC section 2) and the results report. Plain code, no model calls."""
from dataclasses import asdict, dataclass

from src.agents import AnalysisRun

GapKey = tuple[str, str]
GROUPS = {"hand": "Hand-labeled cases", "generated": "Generated cases"}

# name -> (label, target text, comparison, target value). Targets are from SPEC section 2.
MEASURES = {
    "status_accuracy": ("Status accuracy", "-", None, None),
    "gap_recall": ("Gap recall", ">= 0.90", ">=", 0.90),
    "false_alarm_rate": ("False alarm rate", "<= 0.10", "<=", 0.10),
    "escalation_correctness": ("Escalation correctness", ">= 0.80", ">=", 0.80),
    "false_escalation_rate": ("False escalation rate", "-", None, None),
    "reported_citation_validity": ("Citation validity (quotes in reports)", "1.00", ">=", 1.0),
    "raw_citation_validity": ("Raw model quotes verbatim", "-", None, None),
}


@dataclass
class CaseResult:
    """How one case compared with its label."""

    case_id: str
    expected_status: str
    actual_status: str
    hits: set[GapKey]
    missed: set[GapKey]
    extra: set[GapKey]
    raw_quote_failures: list[str]
    raw_quotes_total: int
    final_quotes_verbatim: bool
    ambiguous: bool = False
    final_quotes_total: int = 0
    final_quotes_invalid: int = 0

    @property
    def status_match(self) -> bool:
        return self.expected_status == self.actual_status


@dataclass
class Measures:
    """SPEC section 2 measures for one group of cases in one run. None means undefined."""

    cases: int = 0
    status_accuracy: float | None = None
    gap_recall: float | None = None
    false_alarm_rate: float | None = None
    escalation_correctness: float | None = None
    false_escalation_rate: float | None = None
    reported_citation_validity: float | None = None
    raw_citation_validity: float | None = None


def _verbatim(quote: str, note: str) -> bool:
    return quote.strip() != "" and quote in note


def evaluate_case(case_id: str, label: dict, note: str, run: AnalysisRun) -> CaseResult:
    """Compare one run with its label, and check the model's quotes against the note."""
    expected = {(g["item"], g["reason"]) for g in label["expected_gaps"]}
    actual = {(g.item, g.reason) for g in run.report.gaps}

    claimed = []
    for f in run.raw_fields:
        if f.found and f.quote is not None:
            claimed.append((f.name, f.quote))
        claimed += [(f.name, q) for q in f.conflicting_quotes]
    claimed += [("unreadable", p) for p in run.raw_unreadable]
    failures = [f"{name}: {quote!r}" for name, quote in claimed if not _verbatim(quote, note)]

    final = []
    for f in run.report.fields:
        if f.quote is not None:
            final.append(f.quote)
        final += f.conflicting_quotes
    final += run.report.unreadable_passages
    invalid = sum(not _verbatim(q, note) for q in final)

    return CaseResult(
        case_id=case_id,
        expected_status=label["expected_status"],
        actual_status=run.report.status,
        hits=expected & actual,
        missed=expected - actual,
        extra=actual - expected,
        raw_quote_failures=failures,
        raw_quotes_total=len(claimed),
        final_quotes_verbatim=invalid == 0,
        ambiguous=bool(label.get("ambiguous", False)),
        final_quotes_total=len(final),
        final_quotes_invalid=invalid,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_measures(results: list[CaseResult]) -> Measures:
    """Gap recall and false alarms use non-ambiguous cases only (SPEC section 7)."""
    clear = [r for r in results if not r.ambiguous]
    ambiguous = [r for r in results if r.ambiguous]
    hits = sum(len(r.hits) for r in clear)
    missed = sum(len(r.missed) for r in clear)
    extra = sum(len(r.extra) for r in clear)
    raw_total = sum(r.raw_quotes_total for r in results)
    raw_bad = sum(len(r.raw_quote_failures) for r in results)
    final_total = sum(r.final_quotes_total for r in results)
    final_bad = sum(r.final_quotes_invalid for r in results)
    return Measures(
        cases=len(results),
        status_accuracy=_ratio(sum(r.status_match for r in results), len(results)),
        gap_recall=_ratio(hits, hits + missed),
        false_alarm_rate=_ratio(extra, hits + extra),
        escalation_correctness=_ratio(
            sum(r.actual_status == "NEEDS_REVIEW" for r in ambiguous), len(ambiguous)
        ),
        false_escalation_rate=_ratio(
            sum(r.actual_status == "NEEDS_REVIEW" for r in clear), len(clear)
        ),
        reported_citation_validity=_ratio(final_total - final_bad, final_total),
        raw_citation_validity=_ratio(raw_total - raw_bad, raw_total),
    )


def group_of(case_id: str) -> str:
    """hand_ cases are the trusted core; gen_ cases are generated."""
    if case_id.startswith("hand_"):
        return "hand"
    if case_id.startswith("gen_"):
        return "generated"
    raise ValueError(f"Cannot tell which group {case_id!r} belongs to")


def measure_range(values: list[float | None]) -> tuple[float, float] | None:
    """Min and max across runs, skipping undefined values."""
    present = [v for v in values if v is not None]
    return (min(present), max(present)) if present else None


def meets_target(name: str, value: float | None) -> bool | None:
    """True/False against the SPEC target, or None when there is no target or no value."""
    _, _, op, target = MEASURES[name]
    if op is None or value is None:
        return None
    return value >= target - 1e-9 if op == ">=" else value <= target + 1e-9


def _case_dict(r: CaseResult) -> dict:
    return {
        "case_id": r.case_id,
        "group": group_of(r.case_id),
        "ambiguous": r.ambiguous,
        "expected_status": r.expected_status,
        "actual_status": r.actual_status,
        "hits": sorted([list(k) for k in r.hits]),
        "missed": sorted([list(k) for k in r.missed]),
        "extra": sorted([list(k) for k in r.extra]),
        "raw_quote_failures": r.raw_quote_failures,
        "raw_quotes_total": r.raw_quotes_total,
        "final_quotes_total": r.final_quotes_total,
        "final_quotes_invalid": r.final_quotes_invalid,
    }


def results_to_json(model: str, runs: list[list[CaseResult]]) -> dict:
    """Plain-data form of an evaluation: every case, plus measures per group, for each run."""
    out_runs = []
    for results in runs:
        measures = {}
        for group in GROUPS:
            measures[group] = asdict(compute_measures([r for r in results if group_of(r.case_id) == group]))
        out_runs.append({"cases": [_case_dict(r) for r in results], "measures": measures})
    return {"model": model, "runs": out_runs}


def _f2(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _range_text(values: list[float | None]) -> str:
    r = measure_range(values)
    return "n/a" if r is None else f"{r[0]:.2f}-{r[1]:.2f}"


def _series(data: dict, group: str, name: str) -> list[float | None]:
    return [run["measures"][group][name] for run in data["runs"]]


def _group_cases(data: dict, group: str) -> int:
    return data["runs"][0]["measures"][group]["cases"] if data["runs"] else 0


def _failing(case: dict) -> list[str]:
    problems = []
    if case["expected_status"] != case["actual_status"]:
        problems.append(f"expected {case['expected_status']}, got {case['actual_status']}")
    if not case["ambiguous"]:
        if case["missed"]:
            problems.append("missed " + ", ".join(f"{i}:{r}" for i, r in case["missed"]))
        if case["extra"]:
            problems.append("extra " + ", ".join(f"{i}:{r}" for i, r in case["extra"]))
    if case["raw_quote_failures"]:
        problems.append(f"{len(case['raw_quote_failures'])} raw quote(s) not verbatim")
    if case["final_quotes_invalid"]:
        problems.append(f"{case['final_quotes_invalid']} quote(s) in the report not verbatim")
    return problems


def _failure_section(data: dict) -> list[str]:
    by_case: dict[str, dict] = {}
    for number, run in enumerate(data["runs"], start=1):
        for case in run["cases"]:
            problems = _failing(case)
            if problems:
                entry = by_case.setdefault(
                    case["case_id"], {"runs": [], "ambiguous": case["ambiguous"], "why": "; ".join(problems)}
                )
                entry["runs"].append(number)
    if not by_case:
        return ["No failures in any run."]
    lines = []
    for case_id, entry in by_case.items():
        label = "runs" if len(entry["runs"]) > 1 else "run"
        kind = " (ambiguous)" if entry["ambiguous"] else ""
        lines.append(f"- **{case_id}**{kind}, {label} {', '.join(map(str, entry['runs']))}: {entry['why']}")
    return lines


def render_report(data: dict, baseline: dict | None = None, changes: str | None = None) -> str:
    """Markdown report with the measures, ranges across runs, failures and baseline comparison."""
    n = len(data["runs"])
    lines = [
        "# Evaluation results",
        "",
        f"Model: `{data['model']}`. Runs: {n}. "
        f"Cases per run: {_group_cases(data, 'hand')} hand-labeled, {_group_cases(data, 'generated')} generated.",
        "",
        "Targets come from SPEC section 2. Numbers are as measured; none were adjusted. "
        "Gap recall and false alarm rate use non-ambiguous cases only; ambiguous cases are "
        "scored on whether they were routed to review.",
    ]
    for group, title in GROUPS.items():
        if _group_cases(data, group) == 0:
            continue
        header = "| Measure | Target | " + " | ".join(f"Run {i}" for i in range(1, n + 1)) + " | Range | Target met |"
        lines += ["", f"## {title}", "", header, "|" + " --- |" * (n + 4)]
        for name, (label, target, _, _) in MEASURES.items():
            series = _series(data, group, name)
            met = [meets_target(name, v) for v in series]
            verdict = "-" if any(m is None for m in met) else ("yes" if all(met) else "no")
            cells = " | ".join(_f2(v) for v in series)
            lines.append(f"| {label} | {target} | {cells} | {_range_text(series)} | {verdict} |")

    lines += [
        "",
        "## Safety invariant",
        "",
        "| Measure | Target | Result |",
        "| --- | --- | --- |",
        "| Automatic submissions or sends | 0 | 0 |",
        "",
        "No tool exists that submits, emails or sends anything; `tests/test_no_send_tools.py` enforces it.",
        "",
        "## Failures",
        "",
        *_failure_section(data),
    ]

    if baseline is not None:
        lines += [
            "",
            f"## Baseline comparison (model `{baseline['model']}`, {len(baseline['runs'])} run(s))",
        ]
        for group, title in GROUPS.items():
            if _group_cases(data, group) == 0:
                continue
            lines += ["", f"### {title}", "", "| Measure | Baseline | Final |", "| --- | --- | --- |"]
            for name, (label, _, _, _) in MEASURES.items():
                lines.append(
                    f"| {label} | {_range_text(_series(baseline, group, name))} | {_range_text(_series(data, group, name))} |"
                )
    if changes:
        lines += ["", "## What changed since the baseline", "", changes.strip()]
    return "\n".join(lines) + "\n"
