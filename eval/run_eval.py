"""Run the Packet Analyst on every labeled case and write the evaluation report.

Usage:
    uv run python eval/run_eval.py --runs 3 --out eval/results.md
    uv run python eval/run_eval.py --render eval/results.json --out eval/results.md

It reads data/labels.json and never writes to it. Evaluation runs write their audit
lines to audit/eval_runs.jsonl, separate from the normal run log.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python eval/run_eval.py` finds src/ and eval/

from dotenv import load_dotenv  # noqa: E402

from eval.metrics import CaseResult, evaluate_case, render_report, results_to_json  # noqa: E402
from src.agents import ExtractionLLM, analyze_packet, model_name  # noqa: E402
from src.run import CASES_DIR, ENV_FILE, load_case  # noqa: E402
from src.tools import PROJECT_ROOT  # noqa: E402

LABELS_PATH = PROJECT_ROOT / "data" / "labels.json"
EVAL_AUDIT_PATH = PROJECT_ROOT / "audit" / "eval_runs.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "eval" / "results.md"


def case_ids_for(labels: dict, group: str) -> list[str]:
    """Hand-labeled cases first, then generated ones, each in id order."""
    hand = sorted(k for k in labels if k.startswith("hand_"))
    generated = sorted(k for k in labels if k.startswith("gen_"))
    return {"hand": hand, "generated": generated, "all": hand + generated}[group]


def run_once(
    case_ids: list[str],
    labels: dict,
    *,
    llm: ExtractionLLM | None,
    audit_path: Path,
    cases_dir: Path = CASES_DIR,
    progress=None,
) -> list[CaseResult]:
    """Run every case once through analyze_packet and compare each with its label."""
    results = []
    for index, case_id in enumerate(case_ids, start=1):
        order, note = load_case(case_id, cases_dir)
        run = analyze_packet(order, note, llm=llm, audit_path=audit_path)
        result = evaluate_case(case_id, labels[case_id], note, run)
        results.append(result)
        if progress:
            progress(index, len(case_ids), result)
    return results


def main(
    argv: list[str] | None = None,
    *,
    llm: ExtractionLLM | None = None,
    env_file: Path | None = ENV_FILE,
    labels_path: Path = LABELS_PATH,
    cases_dir: Path = CASES_DIR,
    audit_path: Path = EVAL_AUDIT_PATH,
) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Packet Analyst on the labeled cases.")
    parser.add_argument("--runs", type=int, default=3, help="number of full passes (default 3)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="markdown report to write")
    parser.add_argument("--json", type=Path, help="raw results JSON (default: --out with .json)")
    parser.add_argument("--group", choices=["all", "hand", "generated"], default="all")
    parser.add_argument("--baseline", type=Path, help="baseline results JSON to compare with")
    parser.add_argument("--changes", type=Path, help="markdown text describing what changed")
    parser.add_argument("--render", type=Path, help="re-render the report from a results JSON; no model calls")
    args = parser.parse_args(argv)

    json_path = args.json or args.out.with_suffix(".json")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    changes = args.changes.read_text() if args.changes else None

    if args.render:
        data = json.loads(args.render.read_text())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_report(data, baseline=baseline, changes=changes))
        print(f"Wrote {args.out}")
        return 0

    if env_file is not None:
        load_dotenv(env_file)
    try:
        model = "test-llm" if llm is not None else model_name()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    labels_bytes = labels_path.read_bytes()
    labels_hash = hashlib.sha256(labels_bytes).hexdigest()
    labels = json.loads(labels_bytes)
    case_ids = case_ids_for(labels, args.group)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model} | cases: {len(case_ids)} | runs: {args.runs}")
    all_runs: list[list[CaseResult]] = []
    for run_number in range(1, args.runs + 1):
        def progress(index: int, total: int, r: CaseResult, run_number: int = run_number) -> None:
            ok = "ok" if r.status_match and not (r.missed or r.extra) else "DIFF"
            print(
                f"run {run_number}/{args.runs} [{index}/{total}] {r.case_id}: "
                f"{r.actual_status} (expected {r.expected_status}) {ok}",
                flush=True,
            )

        all_runs.append(
            run_once(case_ids, labels, llm=llm, audit_path=audit_path, cases_dir=cases_dir, progress=progress)
        )
        json_path.write_text(json.dumps(results_to_json(model, all_runs), indent=2) + "\n")

    if hashlib.sha256(labels_path.read_bytes()).hexdigest() != labels_hash:
        print("error: the labels file changed during the evaluation; results are not trusted", file=sys.stderr)
        return 3

    data = results_to_json(model, all_runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(data, baseline=baseline, changes=changes))
    print(f"Wrote {args.out} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
