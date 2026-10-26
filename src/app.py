"""The one-screen Gradio app (SPEC section 10).

Left: choose a case or paste a note, a payer and the order, then Run. Right: the status,
the gaps and evidence with quotes, the drafted request (editable, never sent), the
decision buttons, and the tool-call trace. All logic is in plain functions so it can be
tested without a browser; build_app only wires them to Gradio components.

Run: uv run python -m src.app   (local only, http://127.0.0.1:7860)
"""
import datetime
import html
import re
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
from src.models import ChecklistItem, GapReport, Order, PayerChecklist
from src.run import CASES_DIR, ENV_FILE, load_case
from src.tools import lookup_payer_checklist

PAYERS = ["payer_a", "payer_b", "payer_c"]
MAX_NOTE_CHARS = 20_000
PASTED_CASE_ID = "pasted_note"
# Only synthetic record numbers are ever carried onto the screen.
RECORD_NUMBER = re.compile(r"(?:Record number|MRN):\s*(SYN-\d+)")
RUN_LABEL = "▶️ Run"
APPROVE_LABEL = "✅ Approve draft (logs decision only)"  # it records a decision; nothing is sent
EDIT_LABEL = "✏️ Edit"
DISCARD_LABEL = "🗑️ Discard"
REVIEWED_LABEL = "👀 Mark reviewed"

# Colour means what an item does to the status: green fine, amber not ready, red needs a person,
# gray not applicable, blue neutral. state -> (text colour, background, accent edge).
PALETTE = {
    "ok": ("#14532d", "#dcfce7", "#16a34a"),
    "warn": ("#78350f", "#fef3c7", "#d97706"),
    "bad": ("#7f1d1d", "#fee2e2", "#dc2626"),
    "muted": ("#374151", "#e5e7eb", "#9ca3af"),
    "info": ("#1e3a8a", "#dbeafe", "#3b82f6"),
}
# Solid colours with white text: status badges and the action buttons.
SOLID = {
    "badge-ready": "#15803d",
    "badge-not_ready": "#b45309",
    "badge-needs_review": "#b91c1c",
    "btn-approve": "#15803d",
    "btn-edit": "#1d4ed8",
    "btn-discard": "#b91c1c",
    "btn-reviewed": "#0f766e",
}
BANNER = """# PriorAuth Preflight

**How it works**
1. The model reads the note and quotes the evidence for each item the payer requires.
2. Plain code checks every quote word for word against the note, compares it with the payer's rules, and decides the status.
3. If items are missing, a second model drafts an internal request from the gap list only. A person reviews everything.

**Limits.** Synthetic notes and invented payer rules only. It checks documentation, not whether the treatment is right. It never sends, submits or emails anything. The model can miss or misread, so check the quotes."""
STATUS_STYLE = {
    "READY": ("✅", "badge-ready", "All required items are documented with verified quotes."),
    "NOT_READY": ("⏳", "badge-not_ready", "Some required items are missing or out of date."),
    "NEEDS_REVIEW": ("⚠️", "badge-needs_review", "A person needs to read the evidence."),
}
# Result of one item in the evidence table: palette state, icon, words. Words always stay.
RESULT_PILLS = {
    "found": ("ok", "✅", "found"),
    "not found": ("warn", "❓", "not found"),
    "conflicting": ("bad", "⚠️", "conflicting"),
    "not checked": ("muted", "➖", "not checked"),
}


def build_css() -> str:
    """The page's colour rules, generated from PALETTE and SOLID so colours live in one place.

    Fixed text only: nothing from a note or a form ever goes into the stylesheet.
    """
    rules = [
        ".pill{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;font-size:.88em;white-space:nowrap}",
        ".icon{margin-right:3px}",
        ".badge{display:inline-block;padding:6px 14px;border-radius:8px;color:#fff;font-weight:700;font-size:1.1em}",
        ".callout{padding:8px 12px;border-radius:8px;margin:6px 0;border-left:5px solid transparent}",
        ".legend{font-size:.9em;opacity:.85;margin:4px 0 8px}",
        ".evidence-table{border-collapse:collapse;width:100%}",
        ".evidence-table td,.evidence-table th{border:1px solid rgba(128,128,128,.35);padding:5px 8px;vertical-align:top;text-align:left}",
    ]
    for state, (text, background, edge) in PALETTE.items():
        rules.append(f".pill-{state}{{background:{background};color:{text}}}")
        rules.append(f".callout-{state}{{background:{background};color:{text};border-left-color:{edge}}}")
        rules.append(f".row-{state} td:first-child{{border-left:5px solid {edge}}}")
        rules.append(f".gap-{state}{{border-left:4px solid {edge};padding-left:8px}}")
    for name, colour in SOLID.items():
        if name.startswith("badge-"):
            rules.append(f".{name}{{background:{colour}}}")
        else:  # !important so the button wins over the theme's own button colours
            rules.append(
                f".{name}, .{name}:hover{{background:{colour} !important;color:#fff !important;border-color:{colour} !important}}"
            )
            rules.append(f".{name}:hover{{filter:brightness(.92)}}")
    return "\n".join(rules)


def build_theme() -> gr.themes.Soft:
    """A calm, rounded theme with an indigo primary (the Run button)."""
    return gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")


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
    subject: str = ""  # the draft as shown (record number swapped in); decisions compare against this
    body: str = ""


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


def record_number_from(note: str) -> str | None:
    """The synthetic record number in the note (SYN-nnnnn), or None. Real-looking numbers are ignored."""
    match = RECORD_NUMBER.search(note)
    return match.group(1) if match else None


def _show_record_number(text: str, case_id: str, record: str | None) -> str:
    """Swap the exact case id for the record number in text the person will read.

    This happens after the Outreach model has written the draft: the number comes from the
    note, so the model never sees it (rule 6). Without a record number the text is unchanged.
    """
    if not record:
        return text
    return re.sub(rf"(?<![\w-]){re.escape(case_id)}(?![\w-])", lambda _: record, text)


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


@dataclass
class LabRecency:
    """How old the lab is against the payer's window, in words for the screen."""

    age: int | None
    verdict: str
    text: str


def lab_recency(lab_date: datetime.date | None, order_date: datetime.date, window: int) -> LabRecency:
    """Recompute the lab age the way check_packet judges it (stale only when strictly over the window).

    check_packet only reports a gap when something is wrong, so it does not expose these numbers
    for a lab that is within the window. A test keeps this and check_packet in agreement.
    """
    if lab_date is None:
        return LabRecency(None, "date missing or unreadable", "Lab date missing or unreadable.")
    age = (order_date - lab_date).days
    verdict = "after the order date" if age < 0 else "stale" if age > window else "within window"
    unit = "day" if age == 1 else "days"
    text = f"Lab {lab_date}, order {order_date}: {age} {unit}, window {window}, {verdict}."
    return LabRecency(age, verdict, text)


# --- rendering: every piece of dynamic text goes through html.escape ---------------------


def _e(text: object) -> str:
    return html.escape(str(text))


def _status_html(report: GapReport) -> str:
    icon, badge, blurb = STATUS_STYLE[report.status]
    return (
        f'<span class="badge {badge}"><span class="icon" aria-hidden="true">{icon}</span>{_e(report.status)}</span>'
        f'<span style="margin-left:10px">{_e(blurb)}</span>'
    )


def _known_items() -> list[ChecklistItem]:
    """Every item any payer asks about, in the order of the payer that asks for the most."""
    checklists = [lookup_payer_checklist(p) for p in PAYERS]
    items = list(max(checklists, key=lambda c: len(c.items)).items)
    seen = {item.name for item in items}
    for checklist in checklists:
        for item in checklist.items:
            if item.name not in seen:
                items.append(item)
                seen.add(item.name)
    return items


def render_requirements(payer_id: str | None) -> str:
    """The selected payer's required items and lab window, read from data/checklists.yaml."""
    if payer_id not in PAYERS:
        return "<p>Choose a payer to see its requirements.</p>"
    checklist = lookup_payer_checklist(payer_id)
    items = "".join(
        f"<li><strong>{_e(i.name)}</strong>: {_e(i.description)}{' (a dated lab)' if i.kind == 'lab' else ''}</li>"
        for i in checklist.items
    )
    return (
        f"<p>These are invented payer rules for this prototype (from data/checklists.yaml), not real payer policy.</p>"
        f"<p>Payer {_e(payer_id)} requires:</p><ul>{items}</ul>"
        f"<p>Lab results must be dated within {checklist.lab_window_days} days of the order date.</p>"
    )


def pill(state: str, icon: str, text: str) -> str:
    """A small coloured label: an icon (hidden from screen readers) plus the words.

    The class comes only from the fixed palette, and the words are escaped.
    """
    if state not in PALETTE:
        raise ValueError(f"Unknown colour state {state!r}")
    return f'<span class="pill pill-{state}"><span class="icon" aria-hidden="true">{icon}</span>{_e(text)}</span>'


def _result_pill(kind: str) -> str:
    state, icon, text = RESULT_PILLS[kind]
    return pill(state, icon, text)


def _table_row(name: str, required_html: str, kind: str, extra: str, quote_html: str) -> str:
    state = RESULT_PILLS[kind][0]
    return (
        f'<tr class="row-{state}"><td>{_e(name)}</td><td>{required_html}</td>'
        f"<td>{_result_pill(kind)}{extra}</td><td>{quote_html}</td></tr>"
    )


def _quote(text: str) -> str:
    return f'<blockquote style="margin:4px 0 4px 12px;padding-left:8px;border-left:3px solid #999">{_e(text)}</blockquote>'


def _evidence_html(
    report: GapReport,
    checklist: PayerChecklist,
    lab_lines: dict[str, str] | None = None,
    checked: bool = True,
) -> str:
    fields = {f.name: f for f in report.fields}
    lab_lines = lab_lines or {}
    parts = []
    # A conflicting gap already explains itself next to its quotes; show the reason only when it adds something.
    conflicts_explain_it = any(g.reason == "conflicting" for g in report.gaps) and not report.unreadable_passages
    if report.escalation_reason and not conflicts_explain_it:
        parts.append(f"<p><strong>Escalation reason:</strong> {_e(report.escalation_reason)}</p>")

    parts.append("<h4>Gaps</h4>")
    if not report.gaps:
        parts.append("<p>No gaps.</p>")
    else:
        items = []
        for gap in report.gaps:
            field = fields.get(gap.item)
            quotes = ([field.quote] if field and field.quote else []) + (field.conflicting_quotes if field else [])
            evidence = f'<div style="margin-top:4px">{"".join(_quote(q) for q in quotes)}</div>' if quotes else ""
            items.append(
                f'<li style="margin-bottom:10px"><strong>{_e(gap.item)}</strong> <em>[{_e(gap.reason)}]</em>'
                f'<div style="margin-top:2px">{_e(gap.detail)}</div>{evidence}</li>'
            )
        parts.append("<ul>" + "".join(items) + "</ul>")

    if report.unreadable_passages:
        parts.append("<h4>Unreadable text in the note</h4>")
        parts += [_quote(p) for p in report.unreadable_passages]

    required = {item.name for item in checklist.items}
    conflicted = {g.item for g in report.gaps if g.reason == "conflicting"}
    required_pill = pill("info", "📌", "Required")
    not_required_pill = pill("muted", "➖", "Not required by this payer")
    rows = []
    for item in _known_items():
        f = fields.get(item.name)
        recency = f'<div style="margin:2px 0 4px 12px">{_e(lab_lines[item.name])}</div>' if item.name in lab_lines else ""
        if item.name not in required:
            rows.append(_table_row(item.name, not_required_pill, "not checked", "", ""))
        elif not checked:
            rows.append(_table_row(item.name, required_pill, "not checked", "", ""))
        elif item.name in conflicted:
            quotes = ([f.quote] if f and f.quote else []) + (f.conflicting_quotes if f else [])
            rows.append(_table_row(item.name, required_pill, "conflicting", "", "".join(_quote(q) for q in quotes) + recency))
        elif f is not None and f.found and f.quote:
            date = f" <small>(lab date {_e(f.date)})</small>" if f.date else ""
            rows.append(_table_row(item.name, required_pill, "found", date, _quote(f.quote) + recency))
        else:
            rows.append(_table_row(item.name, required_pill, "not found", "", ""))
    legend = " &middot; ".join(
        f'<span class="icon" aria-hidden="true">{icon}</span> {_e(label)}'
        for icon, label in [("✅", "found"), ("❓", "not found"), ("⏳", "stale"), ("⚠️", "conflicting"), ("➖", "not checked")]
    )
    parts.append(
        f'<h4>Evidence for each item</h4><p class="legend">{legend}</p>'
        '<table class="evidence-table"><tr><th>Item</th><th>Required?</th><th>Result</th>'
        "<th>Quote from the note</th></tr>"
    )
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
    checklist = lookup_payer_checklist(order.payer_id)
    lab_items = {item.name for item in checklist.items if item.kind == "lab"}
    lab_lines = {
        f.name: lab_recency(f.date, order.order_date, checklist.lab_window_days).text
        for f in run.report.fields
        if f.name in lab_items and f.found and f.quote
    }
    notice = None
    if run.draft_error:
        notice = (
            f"The draft could not be written ({run.draft_error}). The gaps below are still valid; "
            "write the request by hand."
        )
    record = record_number_from(form.note)
    subject = _show_record_number(run.draft.subject, order.case_id, record) if run.draft else ""
    body = _show_record_number(run.draft.body, order.case_id, record) if run.draft else ""
    return RunView(
        notice=notice,
        status_html=_status_html(run.report),
        evidence_html=_evidence_html(run.report, checklist, lab_lines, checked="check_packet" in run.tools_called),
        trace=_trace(run, model_label),
        show_draft=run.draft is not None,
        draft_label="Draft request to the care team (not sent)",
        subject=subject,
        body=body,
        show_reviewed=run.draft is None,
        session=Session(run=run, subject=subject, body=body),
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
        changed = subject.strip() != (session.subject or draft.subject) or body.strip() != (session.body or draft.body)
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
                with gr.Accordion("Payer requirements", open=False):
                    requirements = gr.HTML(value=render_requirements("payer_a"))
                order_date = gr.Textbox(label="Order date (YYYY-MM-DD)")
                drug = gr.Textbox(label="Drug")
                dose = gr.Textbox(label="Dose")
                schedule = gr.Textbox(label="Schedule")
                run_button = gr.Button(RUN_LABEL, variant="primary")
            with gr.Column(scale=1):
                status = gr.HTML()
                notice = gr.HTML()
                evidence = gr.HTML()
                with gr.Group(visible=False) as draft_group:
                    subject = gr.Textbox(label="Draft request to the care team (not sent): subject", interactive=True)
                    body = gr.Textbox(label="Body (you can edit it)", lines=8, interactive=True)
                    with gr.Row():
                        approve = gr.Button(APPROVE_LABEL, variant="secondary", elem_classes=["btn-approve"])
                        edit = gr.Button(EDIT_LABEL, elem_classes=["btn-edit"])
                        discard = gr.Button(DISCARD_LABEL, elem_classes=["btn-discard"])
                reviewed = gr.Button(REVIEWED_LABEL, visible=False, elem_classes=["btn-reviewed"])
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
        payer.change(render_requirements, inputs=payer, outputs=requirements)
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
    demo.launch(
        server_name="127.0.0.1", share=False, show_error=False, inbrowser=False, css=build_css(), theme=build_theme()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
