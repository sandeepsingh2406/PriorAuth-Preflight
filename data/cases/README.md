# Cases

Each case is a folder with two files:

```
data/cases/<case_id>/order.json
data/cases/<case_id>/note.txt
```

All content is fictional. Never put real patient data here.

## Generated cases

`gen_001` to `gen_030` are produced by `scripts/gen_cases.py` with a fixed seed.
Do not edit them by hand; regenerate with `uv run python scripts/gen_cases.py`.

## Hand-written cases

Write your own as `hand_01` to `hand_10`. The generator never creates, changes or
deletes `hand_*` folders, and it keeps every entry in `data/labels.json` that is
not a `gen_` entry.

`order.json` (dates are `YYYY-MM-DD`; `payer_id` is `payer_a`, `payer_b` or `payer_c`):

```json
{
  "case_id": "hand_01",
  "payer_id": "payer_b",
  "order_date": "2026-05-04",
  "drug": "Examplumab",
  "dose": "200 mg",
  "schedule": "every 21 days"
}
```

`note.txt` is free text. Start it with `SYNTHETIC NOTE` and use a fake record
number like `SYN-90001`.

Add a label for each case to `data/labels.json`:

```json
"hand_01": {
  "payer_id": "payer_b",
  "expected_status": "NOT_READY",
  "expected_gaps": [{"item": "stage", "reason": "missing"}],
  "ambiguous": false,
  "hand_labeled": true
}
```

- `expected_status`: `READY`, `NOT_READY` or `NEEDS_REVIEW`.
- `expected_gaps`: `item` is one of `diagnosis`, `stage`, `biomarker`,
  `prior_therapies`, `recent_lab`, `treatment_plan` and must be required by that
  payer in `data/checklists.yaml`. `reason` is `missing`, `stale` or `conflicting`.
- `hand_labeled: true` marks the trusted evaluation core. Do not change these
  entries after labeling.
- `ambiguity_kind` is optional and only used on generated cases.
