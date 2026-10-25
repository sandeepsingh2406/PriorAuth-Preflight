# CLAUDE.md

Project: PriorAuth Preflight (repo name `PriorAuth-Preflight`). A prototype that reviews synthetic prior authorization packets, flags missing documentation with quoted evidence, and drafts an internal request to the ordering provider's care team. Read `SPEC.md` first. It is the source of truth for scope.

## Hard rules

1. Synthetic data only. Never add real patient data, real payer names, or real identifiers.
2. Never commit secrets. `.env` is git-ignored. Read keys from environment variables only. Never print keys or put them in logs.
3. No tool may submit, email, send, or post anything. Do not add one. The human approval step is the end of the pipeline.
4. The model reads and drafts. Plain code decides and verifies: checklist lookup, citation validation, gap comparison, and the audit log are deterministic functions with tests.
5. Every quoted piece of evidence must be validated as a verbatim substring of the note. Unvalidated evidence counts as not found.
6. The Outreach agent receives the `GapReport` only, never the clinical note. Keep the test that proves this.
7. Do not edit `data/labels.json` entries marked `"hand_labeled": true`. Those are the trusted evaluation core.
8. Stay inside the scope in `SPEC.md`. If a request seems to need something out of scope, stop and ask.

## Stack

- Python 3.12 with `uv`. Dependencies are in `pyproject.toml`; pin versions.
- OpenAI Agents SDK for agents, tools, handoffs, guardrails, and tracing.
- Pydantic for data models and structured outputs.
- Gradio for the UI. `pytest` for tests.
- Model names come from environment variables (`MODEL_DEV`, `MODEL_DEMO`). Never hard-code a model name in logic.

Before using an SDK feature, check the installed package's source or docs rather than relying on memory. Do not invent APIs. If unsure, write a tiny test script and run it.

## Layout

```
src/models.py     Pydantic models
src/tools.py      lookup, validate_citations, check_packet, audit log, escalate
src/agents.py     Packet Analyst and Outreach agents, handoff, guardrails
src/run.py        run one case from the command line
src/app.py        Gradio UI
scripts/gen_cases.py   seeded synthetic case generator
eval/run_eval.py       evaluation harness, writes eval/results.md
data/             checklists.yaml, cases/, labels.json
tests/            pytest
audit/            runs.jsonl (git-ignored)
NOTES.md          build log (local, git-ignored; see below)
PLAN.md           schedule and checkpoints (local, git-ignored)
```

## Commands

```
uv sync                                  install
uv run pytest -q                         run tests
uv run python scripts/gen_cases.py       regenerate synthetic cases (fixed seed)
uv run python -m src.run --case <id>     run one case
uv run python eval/run_eval.py           run the evaluation
uv run python -m src.app                 start the UI
```

## How to work

- Plan first. For any task bigger than a small fix, propose a short plan and wait for approval.
- One component per task. Keep diffs small and reviewable.
- Tests first for anything in `src/tools.py`. Run `uv run pytest -q` after every change and report the result.
- Do not add dependencies without asking.
- Prefer boring, readable code with type hints and short docstrings. No clever abstractions.
- Prompts live in `src/prompts.py` as named constants, not scattered inline.
- Log tool calls and outcomes to the audit log, never full note text or keys.
- When something fails twice, stop, explain what you tried, and ask rather than looping.

## Git

- Small commits at each green checkpoint, with a clear message ("Add check_packet with recency rules and tests").
- Before the first push and before every push to the public repo, run a secret scan (for example `gitleaks detect` or `detect-secrets scan`) and confirm `.env` and `audit/` are not tracked.
- `.gitignore` must include `.env`, `audit/`, `NOTES.md`, `PLAN.md`, `.venv/` and `__pycache__/`. Create it first, before any commit.
- Never force-push. Never rewrite published history.

## NOTES.md (keep it current)

After each work session, append a short entry:
- What was asked for.
- What the AI produced that was wrong, missing, or insecure, and how it was found and fixed.
- A decision made and why.

Be specific and honest. `NOTES.md` and `PLAN.md` are local working files and are git-ignored; do not commit them and do not copy their contents into public files (README, SPEC, code comments).

## Definition of done for a change

Tests pass, the diff has been read by a human, no secrets in the diff, `SPEC.md` still matches behavior, and `NOTES.md` is updated.
