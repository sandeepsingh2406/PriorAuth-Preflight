"""Sunday gate: run the 10 hand-labeled cases through the single Packet Analyst.

Prints a per-case table, then a pass/fail gate report. It reads data/labels.json
and never writes to it.

Run: uv run python -m scripts.gate
"""
import json
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

from eval.metrics import CaseResult, GapKey, evaluate_case
from src.agents import analyze_packet, model_name
from src.run import ENV_FILE, format_report, load_case
from src.tools import PROJECT_ROOT, audit_log_path

HAND_CASES = [f"hand_{i:02d}" for i in range(1, 11)]
DEMO_CASES = {"hand_01", "hand_07", "hand_08"}
MIN_STATUS_MATCHES = 8
LABELS_PATH = PROJECT_ROOT / "data" / "labels.json"


@dataclass
class GateVerdict:
    passed: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def gate_verdict(results: list[CaseResult], *, audit_ok: bool) -> GateVerdict:
    """Pass if the demo cases are right, >= 8/10 statuses match, and every citation validates."""
    failures, notes = [], []
    by_id = {r.case_id: r for r in results}
    for case_id in sorted(DEMO_CASES):
        r = by_id.get(case_id)
        if r is None:
            failures.append(f"demo case {case_id} was not run")
        elif not r.status_match or r.missed or r.extra:
            failures.append(
                f"demo case {case_id}: expected {r.expected_status}, got {r.actual_status}"
                f"; missed {sorted(r.missed) or '-'}, extra {sorted(r.extra) or '-'}"
            )
    matches = sum(r.status_match for r in results)
    if matches < MIN_STATUS_MATCHES:
        failures.append(f"status matched {matches}/{len(results)} (need {MIN_STATUS_MATCHES})")
    bad = sum(len(r.raw_quote_failures) for r in results)
    if bad:
        failures.append(f"{bad} raw citation(s) failed validate_citations")
    if not all(r.final_quotes_verbatim for r in results):
        failures.append("a quote in a final report is not verbatim (validate_citations bug)")
    if not audit_ok:
        notes.append("audit log: lines added do not match the runs (reported, not a gate condition)")
    return GateVerdict(passed=not failures, failures=failures, notes=notes)


def _gaps(keys: set[GapKey]) -> str:
    return ", ".join(f"{item}:{reason}" for item, reason in sorted(keys)) or "-"


def _audit_lines() -> list[dict]:
    path = audit_log_path()
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main() -> int:
    load_dotenv(ENV_FILE)
    try:
        model = model_name()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    labels = json.loads(LABELS_PATH.read_text())
    before = len(_audit_lines())

    print(f"Model: {model}")
    results, runs = [], {}
    for case_id in HAND_CASES:
        order, note = load_case(case_id)
        run = analyze_packet(order, note)
        runs[case_id] = (note, run)
        results.append(evaluate_case(case_id, labels[case_id], note, run))
        print(f"\n################ {case_id}")
        print(format_report(run))

    new_lines = _audit_lines()[before:]
    audit_ok = [line["case_id"] for line in new_lines] == HAND_CASES

    print()
    print(f"{'case':8} {'expected':13} {'actual':13} {'match':5}  expected gaps  |  actual gaps")
    for r in results:
        print(
            f"{r.case_id:8} {r.expected_status:13} {r.actual_status:13} "
            f"{'yes' if r.status_match else 'NO':5}  "
            f"{_gaps(r.hits | r.missed)}  |  {_gaps(r.hits | r.extra)}"
        )

    for r in results:
        if r.status_match and not r.missed and not r.extra:
            continue
        note, run = runs[r.case_id]
        raw = {f.name: f for f in run.raw_fields}
        print(f"\nDisagreement on {r.case_id}: missed {_gaps(r.missed)}; extra {_gaps(r.extra)}")
        for item in sorted({i for i, _ in r.missed | r.extra}):
            f = raw.get(item)
            if f is None:
                print(f"  {item}: model returned no entry")
            else:
                print(f"  {item}: found={f.found} quote={f.quote!r} conflicting={f.conflicting_quotes}")
        if run.report.escalation_reason:
            print(f"  escalation reason: {run.report.escalation_reason}")

    hits = sum(len(r.hits) for r in results)
    expected = sum(len(r.hits | r.missed) for r in results)
    extra = sum(len(r.extra) for r in results)
    claimed = sum(r.raw_quotes_total for r in results)
    failed = [f"{r.case_id} {q}" for r in results for q in r.raw_quote_failures]
    verdict = gate_verdict(results, audit_ok=audit_ok)

    print("\nGate report")
    print(f"  statuses matched: {sum(r.status_match for r in results)}/{len(results)}")
    print(f"  gaps: {hits} of {expected} expected found, {extra} extra")
    print(f"  raw model quotes passing validate_citations: {claimed - len(failed)}/{claimed}")
    for f in failed:
        print(f"    failed: {f}")
    print(f"  every quote in final reports verbatim: {all(r.final_quotes_verbatim for r in results)}")
    print(f"  audit log: {len(new_lines)} lines added for {len(HAND_CASES)} runs, case ids in order: {audit_ok}")
    print(f"\nGATE: {'PASS' if verdict.passed else 'FAIL'}")
    for f in verdict.failures:
        print(f"  - {f}")
    for n in verdict.notes:
        print(f"  note: {n}")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
