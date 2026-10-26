# PriorAuth Preflight: Spec

Status: v1 prototype. Keep this file short and current. If code and spec disagree, fix one of them the same day.

## 1. Problem

Practice staff prepare prior authorization (PA) packets for oncology treatments. If a payer-required item is missing or stale, the request is returned. That delays treatment and creates rework.

PriorAuth Preflight (a preflight check for prior authorization packets) reviews a draft packet before submission, flags what is missing with evidence, and drafts an internal request to the ordering provider's care team for the missing items. A person reviews everything. Nothing is submitted or sent automatically.

### Real-world terms used in this spec (general US process; practices differ, validate with real staff)

- Payer: the insurer (or the pharmacy benefit manager acting for it) that decides whether to cover a treatment.
- Practice: the organization of physicians and staff. A clinic or site of care is one of its locations. PA staff usually work in the same practice as the providers.
- Prior authorization: the payer's approval, required before a treatment (often a drug) is given or dispensed.
- Order: created by the treating oncologist or an advanced practice provider in the practice's record system. PA staff then assemble the packet and submit it.
- Missing items are requested internally from the ordering provider's care team (nurse, advanced practice provider, or physician).

## 2. Hypothesis and success measures

Hypothesis: catching documentation gaps before submission reduces returned requests and rework.

This prototype cannot test the real-world hypothesis. It tests whether the tool can find gaps reliably and safely on synthetic data. Measures:

| Measure | How it is computed | Target (report the real number, even if lower) |
| --- | --- | --- |
| Gap recall | Seeded gaps found / seeded gaps total | >= 0.90 |
| False alarm rate | Flagged gaps that are not real / flagged gaps total | <= 0.10 |
| Citation validity | Reported evidence quotes that exist verbatim in the note | 1.00 (enforced by code) |
| Escalation correctness | Ambiguous cases routed to human review | >= 0.80 |
| Safety invariant | Automatic submissions or sends across all runs | 0 (no such tool exists) |

## 3. Users and scope

User: PA staff at a practice.

In scope:
- Synthetic orders, synthetic clinical notes, and invented payer checklists.
- Checking documentation completeness only.
- A single screen UI and a command line runner.

Out of scope (do not build, even if easy):
- Real patient data, real payer rules, or any connection to real systems.
- Clinical judgment (whether the treatment is appropriate).
- Submitting, emailing, or faxing anything.
- Authentication, multi-user support, deployment pipelines.

## 4. Flow for one request

1. Input: order (JSON), clinical note (text), payer id.
2. `lookup_payer_checklist(payer_id)` returns the required items for that payer.
3. `extract_fields(note, checklist)` returns each required field with a value and the exact quoted span from the note, or `found = false`, plus any stretches of the note it could not read (`unreadable_passages`, copied verbatim).
4. `validate_citations(note, fields)` drops any field whose quote is not found verbatim in the note and marks it `found = false`. Matching is exact (no case or whitespace changes), and an empty or blank quote never counts. A failed field also loses its `value`, `quote` and `date`. A field is never upgraded to found. `conflicting_quotes` are filtered the same way, so only verbatim ones remain. Reported unreadable passages are checked the same way (`validate_passages`): text the model made up is dropped.
5. `check_packet(fields, checklist, order)` compares to the checklist (including recency windows) and returns the gaps: at most one per checklist item, in checklist order. First matching rule wins:
   - more than one entry for an item: `conflicting`
   - no entry, `found = false`, or a blank quote: `missing`
   - any surviving `conflicting_quotes` (checked before `missing`, so real contradicting evidence is never reported as just missing): `conflicting`
   - the `lab` item: no date, or a date after the order date: `conflicting`; older than the payer's window: `stale`. Stale means strictly more than the window, so a lab exactly `lab_window_days` old is accepted.
   - `treatment_plan`: the verified quote must contain the order's drug name (ignoring case and spacing) and its dose as a whole number (so 200 mg is not satisfied by 1200 mg), otherwise `conflicting`. The schedule is not compared.
   An order and checklist for different payers raise `ValueError`. Gap detail text never repeats quoted note text.
6. Route:
   - No gaps: status `READY`.
   - Gaps: status `NOT_READY`; hand off the gap list (an `OutreachRequest`, built from the report by plain code) to the Outreach agent to draft a request. Only `NOT_READY` packets get a draft: `READY` needs nothing requested, and `NEEDS_REVIEW` goes to a person first because conflicting or unreadable evidence has to be read before anyone is asked for anything. Drafting is opt-in (`draft=True`; the evaluation and gate never turn it on). If the Outreach step fails, the analysis stands and only the draft is missing; the audit line shows `draft_request_failed`.
   - Conflicting or unreadable evidence: status `NEEDS_REVIEW` with a reason. Any verified unreadable passage forces `NEEDS_REVIEW`, even when there are also missing items (the gaps stay listed). The reason is written by code and contains no quoted note text; the passages themselves are shown to the reviewer.
7. A human approves, edits, or discards the draft. The decision is written to the audit log.

## 5. Agents and tools

Build the single-agent version first. Split only if the single-agent version works end to end (first milestone).

| Component | Kind | Responsibility |
| --- | --- | --- |
| Packet Analyst | Agent + code pipeline | `analyze_packet` (plain code) runs steps 2 to 5 in fixed order, routes with `route`, escalates with `escalate_to_human`, writes one audit line, and returns a `GapReport`. The SDK agent itself has no tools and no handoffs; it is used only inside `extract_fields`. If the model call fails, the packet goes to `NEEDS_REVIEW` with the error type as the reason |
| Outreach | Agent | `draft_request` gives it only an `OutreachRequest` (the gap list, no `fields`), never the note or quotes from it, and refuses any other input or any status but `NOT_READY`. It has no tools and no handoffs and writes only a subject and body (`DraftText`); code adds the requested items from the gaps. The draft is never sent |
| `lookup_payer_checklist` | Code | Reads `data/checklists.yaml` |
| `extract_fields` | LLM, structured output | Reads the note, returns an `ExtractionResult` (one field per required item, with quotes). Prompts live in `src/prompts.py`. Model from `MODEL_DEV`, or `MODEL_DEMO` when `USE_DEMO_MODEL=true` |
| `validate_citations` | Code | Verbatim quote check |
| `check_packet` | Code | Deterministic comparison and recency rules |
| `write_audit_log` | Code | Appends a JSON line per run |
| `escalate_to_human` | Code | Returns a copy of the report set to `NEEDS_REVIEW` with a reason (a blank reason is refused; a second reason is appended to the first). It only builds a report and sends nothing |

Rule: the model reads and drafts; code decides and verifies.

Why code runs the pipeline instead of letting the agent call tools: the order of the checks and the status are guarantees, not suggestions. An agent choosing its own tool calls could skip or reorder a check, and the status would have to be recomputed by code anyway. Keeping the model to one structured extraction call makes each run cheaper, testable with the model mocked, and auditable.

Why a handoff: least privilege. The Outreach agent sees only the list of missing items, not the clinical note. A test must prove the note text never reaches it.

`GapReport.fields` carries quotes, which are verbatim excerpts of the note, so the Outreach agent is not given the `GapReport`. Plain code builds an `OutreachRequest` from it (case id, payer id, status, gaps, escalation reason; no `fields`), and that type is the Outreach agent's only input. The person reviewing in the UI still sees the full report with quotes. The test builds a report from a note with distinctive text and asserts that neither the quotes nor any long run of characters from the note appear in what Outreach receives. `check_packet` already keeps quoted note text out of each gap's `detail`.

## 6. Data models (Pydantic, in `src/models.py`)

- `ExtractionResult`: `fields: list[ExtractedField]`, `unreadable_passages: list[str]`. The structured output of the Packet Analyst.
- `ExtractedField`: `name`, `value: str | None`, `quote: str | None`, `found: bool`, `date: date | None` (the lab date, used for recency), `conflicting_quotes: list[str]` (other quotes that contradict `quote`; each is validated verbatim like `quote`, and any that survive make the item `conflicting`)
- `Gap`: `item`, `reason` (`missing` | `stale` | `conflicting`), `detail`
- `GapReport`: `case_id`, `payer_id`, `status` (`READY` | `NOT_READY` | `NEEDS_REVIEW`), `fields`, `gaps`, `escalation_reason: str | None`, `unreadable_passages: list[str]` (verified verbatim in the note)
- `DraftText`: `subject`, `body`. What the Outreach model writes; code adds `requested_items` (taken from the gaps, never from the model) to make a `Draft`.
- `Draft`: `subject`, `body`, `requested_items`
- `OutreachRequest` (built by `build_outreach_request` in `src/tools.py`, extra fields refused): `case_id`, `payer_id`, `status`, `gaps`, `escalation_reason`. The Outreach agent's only input. It has no `fields` and no `unreadable_passages`, so no text from the note.

Supporting models (also in `src/models.py`):

- `ChecklistItem`: `name`, `description`, `kind` (`field` | `lab`). A `lab` item is checked against the payer's recency window.
- `PayerChecklist`: `payer_id`, `lab_window_days` (> 0), `items`
- `Order`: `case_id`, `payer_id`, `order_date`, `drug`, `dose`, `schedule`. A lab is stale when `order_date` minus the lab date is greater than the payer's window.
- `AuditEntry`: `timestamp`, `run_id` (32 lowercase hex characters, optional so older lines still load), `case_id`, `tools_called`, `status`, `human_decision` (`approved` | `edited` | `discarded` | `reviewed` | none). Extra keys are rejected, so note text cannot be written to the log.

## 7. Synthetic data

- `data/checklists.yaml`: three invented payers ("Payer A", "Payer B", "Payer C") with different required items and a lab recency window (for example 30 days, 45 days, 60 days). No real payer names.
- Payer ids are `payer_a` (30 day lab window), `payer_b` (45 days) and `payer_c` (60 days). Item names are `diagnosis`, `stage`, `biomarker`, `prior_therapies`, `recent_lab`, `treatment_plan`. Payer A does not require `prior_therapies`; Payer C does not require `stage`; Payer B requires all six. Each payer has exactly one `lab` item, `recent_lab`.
- Required items (illustrative): confirmed diagnosis with code, cancer stage, relevant biomarker or test result, prior therapies tried with outcomes, recent lab result within the payer window, treatment plan (drug, dose, schedule).
- `scripts/gen_cases.py` generates about 30 notes from templates with seeded gaps (missing stage, stale labs, missing prior therapy, missing plan) and 4 to 5 ambiguous or contradictory notes. Use a fixed random seed.
- Case format: `data/cases/<case_id>/order.json` (an `Order`) and `data/cases/<case_id>/note.txt`. Generated cases are `gen_001` to `gen_030` (6 complete, 19 with one or two seeded gaps, 5 ambiguous); hand-written ones are `hand_01` to `hand_10`. See `data/cases/README.md`.
- Ambiguous kinds: `conflicting_stage`, `conflicting_dose`, `conflicting_lab`, `unreadable_biomarker`, `unreadable_lab_date`. For ambiguous cases the scored outcome is `expected_status` = `NEEDS_REVIEW`; their `expected_gaps` are informational and are not counted in gap recall or false alarm rate.
- `data/labels.json`: one entry per case id: `payer_id`, `expected_status`, `expected_gaps` (list of `item` + `reason`), `ambiguous`, `hand_labeled`, and an optional `ambiguity_kind` on generated cases. The generator rewrites only `gen_` entries and aborts if a `hand_labeled` entry uses a `gen_` id. Hand-write and hand-label 10 cases yourself. These 10 are the trusted core of the evaluation. Do not let the generator or the model overwrite them.
- All people, dates, and identifiers are fictional. No real PHI anywhere in the repo.

## 8. Safety and guardrails

- No tool exists that submits, emails, or sends anything. A test scans every module in `src/` and fails on any function named like a sending action (`send`, `submit`, `email`, `fax`, `post`, `upload`, `publish`, `notify`) or any import of a mail, socket or HTTP client module.
- Every reported gap or evidence item must carry a validated citation, or be marked not found.
- The agent abstains and escalates when evidence conflicts or the note is unreadable.
- The Outreach agent receives the gap list only (an `OutreachRequest`), never the note and never quotes from it.
- Every run is appended to `audit/runs.jsonl`: timestamp, run id, case id, tools called, status. The person's decision is a separate, later line with the same `run_id` (written by `record_decision`), so the log stays append-only and decisions link to runs exactly. With a draft on screen the decision is `approved`, `edited` or `discarded`; with no draft there is nothing to approve, so it is `reviewed`. A decision line holds only the decision, never note or draft text. A person may decide more than once; the last line for a `run_id` is the final word. The path is `AUDIT_LOG_PATH` if set, else `audit/runs.jsonl` under the project root. The writer accepts only an `AuditEntry` (which rejects extra keys), so note text cannot be logged. It appends one JSON line per run and raises on a write error rather than skipping the line.
- Secrets live in `.env` (git-ignored). Never in code, logs, or commits.
- Input limits are enforced in code (`src/guardrails.py`), not only in the UI: a note is at most 2,000 characters (an emoji counts as one), drug and schedule at most 80, dose at most 40, the order date must be a real `YYYY-MM-DD`, the payer must be one of the configured payers, and a note containing the internal prompt markers (`<<<NOTE`, `NOTE>>>`) is refused because it could fake the end of the note. Rejected input returns a clear message, makes no model call and writes no audit line. The note box shows a live character counter and a "Synthetic data only. Do not enter real patient information." banner.
- `DEMO_MODE=1` (off by default) is for public hosting: 10 runs per session, a daily cap of `DAILY_RUN_CAP` runs (default 200) shared by all sessions, a queue limit of 2, the audit log in a fresh temp folder, and SDK tracing switched off in code. A capped request gets a friendly message and no model call; invalid input never uses up a run; a run whose model call fails still counts. Unexpected errors show one fixed short message with no stack trace, and logs get only the exception's type, never note text, order fields or keys.

## 9. Evaluation (`eval/run_eval.py`)

- Runs all cases, computes the measures in section 2, and writes `eval/results.md`.
- Run it 3 times and report the range, because LLM output varies.
- Report the 10 hand-labeled cases separately from the generated ones.
- List the failures and what changed to fix them. Keep the honest numbers.

## 10. UI (`src/app.py`, Gradio)

One screen. Left: choose a case or paste a note, choose a payer, Run. Right: status badge, gap list with quoted evidence, the drafted request, and Approve, Edit, Discard buttons. A small panel shows the tool-call trace.

Built in `src/app.py` (`uv run python -m src.app`):

- Left: a case dropdown (hand-labeled cases first) that fills the note, payer and order fields, or leave it empty and paste a note and fill the order by hand, then Run. A chosen case keeps its id in the audit log only while the note is unchanged; otherwise the case id is `pasted_note`. Bad input (date, payer, empty or very long note, empty order fields) gives a plain message and runs nothing.
- Right: the status badge, the gaps with their quoted evidence (both quotes for a conflict, "no evidence found in the note" for a missing item), any unreadable text, an evidence table with a quote for every item, the escalation reason if any, and a collapsible tool-call trace with the run id.
- A `NOT_READY` packet shows an editable draft with a neutral "Approve draft (logs decision only)" button, Edit and Discard. Approve after changing the text records `edited` then `approved`; Edit records `edited` only if something changed. Every other case has no draft and shows Mark reviewed (`reviewed`). All decisions go through `record_decision` and the confirmation says nothing was sent.
- Under the title, a short "How it works" (three steps) and "Limits" block. A collapsed "Payer requirements" panel shows the selected payer's required items and lab window, read from `data/checklists.yaml` through `lookup_payer_checklist`, and updates when the payer changes.
- The evidence table lists every known item, each marked "Required" or "Not required by this payer" (shown as "not checked", with no quote). A required item the model left out shows as "not found"; after a model failure required items show "not checked". An item with conflicting evidence (including a lab with no readable date) shows as "conflicting" with every quote, so the table never says "found" for something the gap list says is contradicted. The lab row also reads "Lab 2026-04-10, order 2026-05-11: 31 days, window 30, stale." (or "within window", "after the order date", or "Lab date missing or unreadable."). `check_packet` reports a gap only when something is wrong, so the screen recomputes the age with `lab_recency`; a test keeps it in agreement with `check_packet` at every boundary.
- Each gap shows its detail and its quotes on separate lines. A missing item says "Not documented in the note: ..." once. The escalation reason is shown only when it adds something (unreadable text or a model failure), not when a conflicting gap already explains it.
- The draft names the note's synthetic record number (`SYN-nnnnn`) instead of the case id. The Outreach model never sees the number (it comes from the note, rule 6): it writes with the case id and the screen swaps the exact id for the number afterwards. Without a `SYN-` record number the case id stays. Approve and Edit compare against the text as shown.
- Colour and icons. Colour means what an item does to the status: green is fine (found, within the window, READY), amber is not ready (not found, stale, NOT_READY), red needs a person (conflicting, a lab after the order or with no readable date, NEEDS_REVIEW, errors), gray is not applicable (not required, not checked), blue is neutral (Required, Edit). Status badges, result and requirement pills, row edges, gap reasons, the lab line, callouts and the action buttons all follow it. Approve is green, Edit blue, Discard red and Mark reviewed teal; Run uses the theme's indigo. Colour is never the only signal: every pill, badge and callout keeps its words, icons are hidden from screen readers, and a legend sits above the table. All colours live in one palette in `src/app.py`, the CSS is generated from it and passed to `launch`, and a test checks that every text and background pair meets the WCAG AA contrast ratio (4.5:1).
- All note text is HTML-escaped before it reaches the page. State is per browser session (`gr.State`), never a module variable. The app listens on `127.0.0.1` only, with no share link and Gradio telemetry off. Gradio runs one analysis at a time per button by default; `default_concurrency_limit=1` is set explicitly.

## 11. Known limits

- Checklists are illustrative, not real payer policy.
- Synthetic notes are cleaner than real ones.
- The evaluation set is small.
- `validate_citations` verifies the quote, not the `date` the model attaches to it. A model could quote a real lab line but report a wrong date, which would skew the recency check. Dates in notes come in mixed styles, so verifying them needs a date parser that is not built.
- It checks documentation completeness, not clinical appropriateness.
- Code proves a quote exists in the note, and that the plan names the ordered drug and dose; it does not prove a quote is relevant to the item it is cited for. A model that lies about relevance (for example after a prompt injection) can satisfy an item with an irrelevant but real quote, which is why the screen shows every quote for a person to check. A test pins this behaviour.
- The per-session cap lives in the browser session, so reloading the page starts a new one; the daily cap is the real ceiling. Both counters are in memory and reset when the process restarts or a new UTC day begins. There is no per-IP limit.
