"""The one-screen Gradio app (SPEC section 10).

Left: choose a case or paste a note, a payer and the order, then Run. Right: the status,
the gaps and evidence with quotes, the drafted request (editable, never sent), the
decision buttons, and the tool-call trace. All logic is in plain functions so it can be
tested without a browser; build_app only wires them to Gradio components.

Run: uv run python -m src.app   (local only, http://127.0.0.1:7860)
"""
import datetime
import html
import sys
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel

from src.agents import (
    AnalysisRun,
    ExtractionLLM,
    OutreachLLM,
    analyze_packet,
    model_name,
    record_decision,
)
from src.models import GapReport, Order
from src.run import CASES_DIR, ENV_FILE, load_case

PAYERS = ["payer_a", "payer_b", "payer_c"]
MAX_NOTE_CHARS = 20_000
PASTED_CASE_ID = "pasted_note"
BANNER = (
    "# PriorAuth Preflight\n"
    "Synthetic data only. This tool checks a packet and drafts an internal request. "
    "**It never sends, submits or emails anything**: you review everything."
)
STATUS_STYLE = {
    "READY": ("#1b7f3b", "All required items are documented with verified quotes."),
    "NOT_READY": ("#b45309", "Some required items are missing or out of date."),
    "NEEDS_REVIEW": ("#b91c1c", "A person needs to read the evidence."),
}


class CaseForm(BaseModel):
    """What is on the left of the screen, as plain strings."""

    case_id: str = ""
    note: str = ""
    payer_id: str = "payer_a"
    order_date: str = ""
    drug: str = ""
    dose: str = ""
    schedule: str = ""


@dataclass
class Session:
    """One person's current run. Kept in Gradio's per-session state, never in a global."""

    run: AnalysisRun


@dataclass
class RunView:
    """Everything the right-hand side shows after Run, as plain values."""

    error: str | None = None
    notice: str | None = None
    status_html: str = ""
    evidence_html: str = ""
    trace: str = ""
    show_draft: bool = False
    draft_label: str = ""
    subject: str = ""
    body: str = ""
    show_reviewed: bool = False
    session: Session | None = None

    @property
    def draft(self):
        return self.session.run.draft if self.session else None


def list_cases(cases_dir: Path = CASES_DIR) -> list[str]:
    """Hand-labeled cases first, then generated ones, each in id order."""
    names = [p.name for p in cases_dir.iterdir() if p.is_dir()]
    return sorted(n for n in names if n.startswith("hand_")) + sorted(n for n in names if n.startswith("gen_"))


def load_case_into_form(case_id: str) -> CaseForm:
    """Read a case from data/cases into the form. Only plain case ids are accepted."""
    order, note = load_case(case_id)
    return CaseForm(
        case_id=case_id,
        note=note,
        payer_id=order.payer_id,
        order_date=order.order_date.isoformat(),
        drug=order.drug,
        dose=order.dose,
        schedule=order.schedule,
    )


def case_id_for(selected: str | None, note: str) -> str:
    """The audit case id: the chosen case only while the note is exactly that case's note."""
    if selected:
        try:
            if load_case(selected)[1] == note:
                return selected
        except (ValueError, FileNotFoundError):
            pass
    return PASTED_CASE_ID


def _to_order(form: CaseForm) -> tuple[Order | None, str | None]:
    """Check the form. Returns an Order, or a plain-words message saying what to fix."""
    if not form.note.strip():
        return None, "Paste or choose a note first."
    if len(form.note) > MAX_NOTE_CHARS:
        return None, f"The note is too long (limit {MAX_NOTE_CHARS:,} characters)."
    if form.payer_id not in PAYERS:
        return None, "Choose a payer from the list."
    try:
        order_date = datetime.date.fromisoformat(form.order_date.strip())
    except ValueError:
        return None, "Enter the order date as YYYY-MM-DD."
    for label, value in [("drug", form.drug), ("dose", form.dose), ("schedule", form.schedule)]:
        if not value.strip():
            return None, f"Enter the {label} from the order."
    order = Order(
        case_id=case_id_for(form.case_id, form.note),
        payer_id=form.payer_id,
        order_date=order_date,
        drug=form.drug.strip(),
        dose=form.dose.strip(),
        schedule=form.schedule.strip(),
    )
    return order, None


# --- rendering: every piece of dynamic text goes through html.escape ---------------------


def _e(text: object) -> str:
    return html.escape(str(text))


def _status_html(report: GapReport) -> str:
    color, blurb = STATUS_STYLE[report.status]
    return (
        f'<div style="display:inline-block;padding:6px 14px;border-radius:6px;background:{color};'
        f'color:#fff;font-weight:700;font-size:1.1em">{_e(report.status)}</div>'
        f'<span style="margin-left:10px">{_e(blurb)}</span>'
    )


def _quote(text: str) -> str:
    return f'<blockquote style="margin:4px 0 4px 12px;padding-left:8px;border-left:3px solid #999">{_e(text)}</blockquote>'


def _evidence_html(report: GapReport) -> str:
    fields = {f.name: f for f in report.fields}
    parts = []
    if report.escalation_reason:
        parts.append(f"<p><strong>Escalation reason:</strong> {_e(report.escalation_reason)}</p>")

    parts.append("<h4>Gaps</h4>")
    if not report.gaps:
        parts.append("<p>No gaps.</p>")
    else:
        items = []
        for gap in report.gaps:
            field = fields.get(gap.item)
            quotes = ([field.quote] if field and field.quote else []) + (field.conflicting_quotes if field else [])
            evidence = "".join(_quote(q) for q in quotes) or "<em>no evidence found in the note</em>"
            items.append(f"<li><strong>{_e(gap.item)}</strong> [{_e(gap.reason)}]: {_e(gap.detail)}{evidence}</li>")
        parts.append("<ul>" + "".join(items) + "</ul>")

    if report.unreadable_passages:
        parts.append("<h4>Unreadable text in the note</h4>")
        parts += [_quote(p) for p in report.unreadable_passages]

    rows = []
    for f in report.fields:
        if f.found and f.quote:
            date = f" (lab date {_e(f.date)})" if f.date else ""
            rows.append(f"<tr><td>{_e(f.name)}</td><td>found{date}</td><td>{_quote(f.quote)}</td></tr>")
        else:
            rows.append(f"<tr><td>{_e(f.name)}</td><td>not found</td><td></td></tr>")
    parts.append("<h4>Evidence for each item</h4><table><tr><th>Item</th><th>Result</th><th>Quote from the note</th></tr>")
    parts.append("".join(rows) + "</table>")
    return "".join(parts)


def _trace(run: AnalysisRun, model_label: str) -> str:
    lines = [f"Run id: {run.run_id}", f"Model: {model_label}", "Tools called, in order:"]
    lines += [f"  {i}. {name}" for i, name in enumerate(run.tools_called, start=1)]
    if run.draft_error:
        lines.append(f"Draft request failed: {run.draft_error}")
    return "\n".join(lines)


def run_case(
    form: CaseForm,
    *,
    llm: ExtractionLLM | None = None,
    outreach_llm: OutreachLLM | None = None,
    audit_path: Path | None = None,
    model_label: str = "not shown",
) -> RunView:
    """Check the form, run the pipeline with drafting on, and describe the result for the screen."""
    order, problem = _to_order(form)
    if problem:
        return RunView(error=problem)
    run = analyze_packet(
        order, form.note, llm=llm, outreach_llm=outreach_llm, audit_path=audit_path, draft=True
    )
    notice = None
    if run.draft_error:
        notice = (
            f"The draft could not be written ({run.draft_error}). The gaps below are still valid; "
            "write the request by hand."
        )
    return RunView(
        notice=notice,
        status_html=_status_html(run.report),
        evidence_html=_evidence_html(run.report),
        trace=_trace(run, model_label),
        show_draft=run.draft is not None,
        draft_label="Draft request to the care team (not sent)",
        subject=run.draft.subject if run.draft else "",
        body=run.draft.body if run.draft else "",
        show_reviewed=run.draft is None,
        session=Session(run=run),
    )


NOT_SENT = "Nothing was sent: this tool never sends anything. Use the text yourself if you want to."


def decide(
    session: Session | None, action: str, subject: str, body: str, *, audit_path: Path | None = None
) -> str:
    """Record the person's decision as a linked audit line and return a message for the screen."""
    if session is None:
        return "Run a case first."
    run = session.run
    draft = run.draft
    try:
        if action == "reviewed":
            if draft is not None:
                return "A draft is on screen, so this case cannot be marked reviewed; approve, edit or discard it."
            record_decision(run, "reviewed", audit_path=audit_path)
            return "Decision recorded: reviewed."
        if draft is None:
            return "A draft was not written for this run, so there is nothing to approve, edit or discard. Use Mark reviewed."
        changed = subject.strip() != draft.subject or body.strip() != draft.body
        if action == "edit":
            if not changed:
                return "No changes to record."
            record_decision(run, "edited", audit_path=audit_path)
            return f"Decision recorded: edited. {NOT_SENT}"
        if action == "approve":
            if changed:
                record_decision(run, "edited", audit_path=audit_path)
            record_decision(run, "approved", audit_path=audit_path)
            return f"Decision recorded: approved. {NOT_SENT}"
        if action == "discard":
            record_decision(run, "discarded", audit_path=audit_path)
            return f"Decision recorded: discarded. {NOT_SENT}"
    except OSError as error:
        return f"The decision could not be saved ({type(error).__name__}). Try again."
    return "That action is not available."


# --- the Gradio wiring -----------------------------------------------------------------


def _notice_html(view: RunView) -> str:
    message = view.error or view.notice
    return f'<p style="color:#b91c1c">{_e(message)}</p>' if message else ""


def build_app(
    *,
    llm: ExtractionLLM | None = None,
    outreach_llm: OutreachLLM | None = None,
    audit_path: Path | None = None,
    model_label: str = "not shown",
) -> gr.Blocks:
    """Build the screen. Models and the audit path can be injected for tests."""
    with gr.Blocks(title="PriorAuth Preflight", analytics_enabled=False) as demo:
        gr.Markdown(BANNER)
        session = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                case = gr.Dropdown(choices=list_cases(), value=None, label="Case (or leave empty and paste a note)", filterable=True)
                note = gr.Textbox(label="Clinical note (synthetic)", lines=14, max_lines=30)
                payer = gr.Dropdown(choices=PAYERS, value="payer_a", label="Payer")
                order_date = gr.Textbox(label="Order date (YYYY-MM-DD)")
                drug = gr.Textbox(label="Drug")
                dose = gr.Textbox(label="Dose")
                schedule = gr.Textbox(label="Schedule")
                run_button = gr.Button("Run", variant="primary")
            with gr.Column(scale=1):
                status = gr.HTML()
                notice = gr.HTML()
                evidence = gr.HTML()
                with gr.Group(visible=False) as draft_group:
                    subject = gr.Textbox(label="Draft request to the care team (not sent): subject", interactive=True)
                    body = gr.Textbox(label="Body (you can edit it)", lines=8, interactive=True)
                    with gr.Row():
                        approve = gr.Button("Approve", variant="primary")
                        edit = gr.Button("Edit")
                        discard = gr.Button("Discard")
                reviewed = gr.Button("Mark reviewed", visible=False)
                decision = gr.HTML()
                with gr.Accordion("Tool-call trace", open=False):
                    trace = gr.Textbox(label="Trace", lines=8, interactive=False)

        def on_case_change(case_id: str | None):
            if not case_id:
                return [gr.update()] * 6
            form = load_case_into_form(case_id)
            return form.note, form.payer_id, form.order_date, form.drug, form.dose, form.schedule

        def on_run(case_id, note_text, payer_id, date_text, drug_text, dose_text, schedule_text):
            form = CaseForm(
                case_id=case_id or "", note=note_text or "", payer_id=payer_id or "",
                order_date=date_text or "", drug=drug_text or "", dose=dose_text or "", schedule=schedule_text or "",
            )
            view = run_case(form, llm=llm, outreach_llm=outreach_llm, audit_path=audit_path, model_label=model_label)
            return (
                view.session,
                view.status_html,
                _notice_html(view),
                view.evidence_html,
                gr.update(visible=view.show_draft),
                view.subject,
                view.body,
                gr.update(visible=view.show_reviewed),
                view.trace,
                "",
            )

        def make_decider(action: str):
            def on_decision(current, subject_text, body_text):
                message = decide(current, action, subject_text or "", body_text or "", audit_path=audit_path)
                return f"<p>{_e(message)}</p>"

            return on_decision

        case.change(on_case_change, inputs=case, outputs=[note, payer, order_date, drug, dose, schedule])
        run_button.click(
            on_run,
            inputs=[case, note, payer, order_date, drug, dose, schedule],
            outputs=[session, status, notice, evidence, draft_group, subject, body, reviewed, trace, decision],
        )
        for button, action in [(approve, "approve"), (edit, "edit"), (discard, "discard")]:
            button.click(make_decider(action), inputs=[session, subject, body], outputs=decision)
        reviewed.click(make_decider("reviewed"), inputs=[session, subject, body], outputs=decision)
    return demo


def main(
    *,
    env_file: Path | None = ENV_FILE,
    llm: ExtractionLLM | None = None,
    outreach_llm: OutreachLLM | None = None,
) -> int:
    """Start the app on this computer only. Exit code 2 if no model is configured."""
    if env_file is not None:
        load_dotenv(env_file)
    try:
        label = "test-llm" if llm is not None else model_name()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    demo = build_app(llm=llm, outreach_llm=outreach_llm, model_label=label)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", share=False, show_error=False, inbrowser=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
