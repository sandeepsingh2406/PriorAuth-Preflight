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
3. `extract_fields(note, checklist)` returns each required field with a value and the exact quoted span from the note, or `found = false`.
4. `validate_citations(note, fields)` drops any field whose quote is not found verbatim in the note and marks it `found = false`. Matching is exact (no case or whitespace changes), and an empty or blank quote never counts. A failed field also loses its `value`, `quote` and `date`. A field is never upgraded to found. `conflicting_quotes` are filtered the same way, so only verbatim ones remain.
5. `check_packet(fields, checklist, order)` compares to the checklist (including recency windows) and returns the gaps: at most one per checklist item, in checklist order. First matching rule wins:
   - more than one entry for an item: `conflicting`
   - no entry, `found = false`, or a blank quote: `missing`
   - any surviving `conflicting_quotes` (checked before `missing`, so real contradicting evidence is never reported as just missing): `conflicting`
   - the `lab` item: no date, or a date after the order date: `conflicting`; older than the payer's window: `stale`. Stale means strictly more than the window, so a lab exactly `lab_window_days` old is accepted.
   - `treatment_plan`: the verified quote must contain the order's drug name (ignoring case and spacing) and its dose as a whole number (so 200 mg is not satisfied by 1200 mg), otherwise `conflicting`. The schedule is not compared.
   An order and checklist for different payers raise `ValueError`. Gap detail text never repeats quoted note text.
6. Route:
   - No gaps: status `READY`.
   - Gaps: status `NOT_READY`; hand off the gap list (an `OutreachRequest`, built from the report by plain code) to the Outreach agent to draft a request.
   - Conflicting or unreadable evidence: status `NEEDS_REVIEW` with a reason.
7. A human approves, edits, or discards the draft. The decision is written to the audit log.

## 5. Agents and tools

Build the single-agent version first. Split only if the single-agent version works end to end (first milestone).

| Component | Kind | Responsibility |
| --- | --- | --- |
| Packet Analyst | Agent | Runs steps 2 to 5, decides the route, produces a `GapReport` |
| Outreach | Agent | Receives only an `OutreachRequest` (the gap list, no `fields`), never the note or quotes from it; drafts the internal request to the ordering provider's care team |
| `lookup_payer_checklist` | Code | Reads `data/checklists.yaml` |
| `extract_fields` | LLM, structured output | Reads the note, returns fields with quotes |
| `validate_citations` | Code | Verbatim quote check |
| `check_packet` | Code | Deterministic comparison and recency rules |
| `write_audit_log` | Code | Appends a JSON line per run |
| `escalate_to_human` | Code | Returns a copy of the report set to `NEEDS_REVIEW` with a reason (a blank reason is refused; a second reason is appended to the first). It only builds a report and sends nothing |

Rule: the model reads and drafts; code decides and verifies.

Why a handoff: least privilege. The Outreach agent sees only the list of missing items, not the clinical note. A test must prove the note text never reaches it.

`GapReport.fields` carries quotes, which are verbatim excerpts of the note, so the Outreach agent is not given the `GapReport`. Plain code builds an `OutreachRequest` from it (case id, payer id, status, gaps, escalation reason; no `fields`), and that type is the Outreach agent's only input. The person reviewing in the UI still sees the full report with quotes. The test builds a report from a note with distinctive text and asserts that neither the quotes nor any long run of characters from the note appear in what Outreach receives. `check_packet` already keeps quoted note text out of each gap's `detail`.

## 6. Data models (Pydantic, in `src/models.py`)

- `ExtractedField`: `name`, `value: str | None`, `quote: str | None`, `found: bool`, `date: date | None` (the lab date, used for recency), `conflicting_quotes: list[str]` (other quotes that contradict `quote`; each is validated verbatim like `quote`, and any that survive make the item `conflicting`)
- `Gap`: `item`, `reason` (`missing` | `stale` | `conflicting`), `detail`
- `GapReport`: `case_id`, `payer_id`, `status` (`READY` | `NOT_READY` | `NEEDS_REVIEW`), `fields`, `gaps`, `escalation_reason: str | None`
- `Draft`: `subject`, `body`, `requested_items`
- `OutreachRequest` (not built yet; added with the Outreach agent): `case_id`, `payer_id`, `status`, `gaps`, `escalation_reason`. The Outreach agent's only input. It has no `fields`, so no quotes from the note.

Supporting models (also in `src/models.py`):

- `ChecklistItem`: `name`, `description`, `kind` (`field` | `lab`). A `lab` item is checked against the payer's recency window.
- `PayerChecklist`: `payer_id`, `lab_window_days` (> 0), `items`
- `Order`: `case_id`, `payer_id`, `order_date`, `drug`, `dose`, `schedule`. A lab is stale when `order_date` minus the lab date is greater than the payer's window.
- `AuditEntry`: `timestamp`, `case_id`, `tools_called`, `status`, `human_decision` (`approved` | `edited` | `discarded` | none). Extra keys are rejected, so note text cannot be written to the log.

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
- Every run is appended to `audit/runs.jsonl`: timestamp, case id, tools called, status, human decision. The path is `AUDIT_LOG_PATH` if set, else `audit/runs.jsonl` under the project root. The writer accepts only an `AuditEntry` (which rejects extra keys), so note text cannot be logged. It appends one JSON line per run and raises on a write error rather than skipping the line.
- Secrets live in `.env` (git-ignored). Never in code, logs, or commits.

## 9. Evaluation (`eval/run_eval.py`)

- Runs all cases, computes the measures in section 2, and writes `eval/results.md`.
- Run it 3 times and report the range, because LLM output varies.
- Report the 10 hand-labeled cases separately from the generated ones.
- List the failures and what changed to fix them. Keep the honest numbers.

## 10. UI (`src/app.py`, Gradio)

One screen. Left: choose a case or paste a note, choose a payer, Run. Right: status badge, gap list with quoted evidence, the drafted request, and Approve, Edit, Discard buttons. A small panel shows the tool-call trace.

## 11. Known limits

- Checklists are illustrative, not real payer policy.
- Synthetic notes are cleaner than real ones.
- The evaluation set is small.
- `validate_citations` verifies the quote, not the `date` the model attaches to it. A model could quote a real lab line but report a wrong date, which would skew the recency check. Dates in notes come in mixed styles, so verifying them needs a date parser that is not built.
- It checks documentation completeness, not clinical appropriateness.
