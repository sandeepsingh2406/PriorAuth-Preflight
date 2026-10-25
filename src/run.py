"""Run one case from the command line.

Usage: uv run python -m src.run --case <case_id>
"""
import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agents import AnalysisRun, ExtractionLLM, analyze_packet, model_name
from src.models import Order
from src.tools import PROJECT_ROOT

CASES_DIR = PROJECT_ROOT / "data" / "cases"
ENV_FILE = PROJECT_ROOT / ".env"
CASE_ID = re.compile(r"^[A-Za-z0-9_]+$")


def load_case(case_id: str, cases_dir: Path = CASES_DIR) -> tuple[Order, str]:
    """Read data/cases/<case_id>/order.json and note.txt. Only plain ids are accepted."""
    if not CASE_ID.match(case_id):
        raise ValueError(f"Invalid case id {case_id!r}: use letters, digits and _ only")
    folder = cases_dir / case_id
    if not folder.is_dir():
        raise FileNotFoundError(f"No case folder for {case_id!r} in {cases_dir}")
    order = Order.model_validate_json((folder / "order.json").read_text())
    return order, (folder / "note.txt").read_text()


def format_report(run: AnalysisRun) -> str:
    """Status, each gap with its quoted evidence, and the escalation reason if any."""
    report = run.report
    fields = {f.name: f for f in report.fields}
    lines = [f"Case {report.case_id} ({report.payer_id}): {report.status}"]
    if not report.gaps:
        lines.append("Gaps: none")
    else:
        lines.append("Gaps:")
        for gap in report.gaps:
            lines.append(f"  - {gap.item} [{gap.reason}]: {gap.detail}")
            field = fields.get(gap.item)
            quotes = []
            if field is not None and field.quote:
                quotes.append(field.quote)
            if field is not None:
                quotes += field.conflicting_quotes
            if quotes:
                lines += [f'      evidence: "{q}"' for q in quotes]
            else:
                lines.append("      evidence: no evidence found in the note")
    lines += [f'Unreadable text: "{passage}"' for passage in report.unreadable_passages]
    if report.escalation_reason:
        lines.append(f"Escalation reason: {report.escalation_reason}")
    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    *,
    llm: ExtractionLLM | None = None,
    env_file: Path | None = ENV_FILE,
) -> int:
    """Run one case and print its report. Exit code 2 for bad input or configuration."""
    parser = argparse.ArgumentParser(description="Check one synthetic prior authorization packet.")
    parser.add_argument("--case", required=True, help="case id, e.g. hand_01")
    args = parser.parse_args(argv)

    if env_file is not None:
        load_dotenv(env_file)
    try:
        if llm is None:
            model_name()  # fail early with a clear message if no model is configured
        order, note = load_case(args.case)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(format_report(analyze_packet(order, note, llm=llm)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
