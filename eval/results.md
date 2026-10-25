# Evaluation results

Model: `gpt-6-luna`. Runs: 3. Cases per run: 10 hand-labeled, 30 generated.

Targets come from SPEC section 2. Numbers are as measured; none were adjusted. Gap recall and false alarm rate use non-ambiguous cases only; ambiguous cases are scored on whether they were routed to review.

## Hand-labeled cases

| Measure | Target | Run 1 | Run 2 | Run 3 | Range | Target met |
| --- | --- | --- | --- | --- | --- | --- |
| Status accuracy | - | 1.00 | 1.00 | 1.00 | 1.00-1.00 | - |
| Gap recall | >= 0.90 | 1.00 | 1.00 | 1.00 | 1.00-1.00 | yes |
| False alarm rate | <= 0.10 | 0.00 | 0.00 | 0.00 | 0.00-0.00 | yes |
| Escalation correctness | >= 0.80 | 1.00 | 1.00 | 1.00 | 1.00-1.00 | yes |
| False escalation rate | - | 0.00 | 0.00 | 0.00 | 0.00-0.00 | - |
| Citation validity (quotes in reports) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00-1.00 | yes |
| Raw model quotes verbatim | - | 1.00 | 1.00 | 1.00 | 1.00-1.00 | - |

## Generated cases

| Measure | Target | Run 1 | Run 2 | Run 3 | Range | Target met |
| --- | --- | --- | --- | --- | --- | --- |
| Status accuracy | - | 0.97 | 0.97 | 0.97 | 0.97-0.97 | - |
| Gap recall | >= 0.90 | 1.00 | 1.00 | 1.00 | 1.00-1.00 | yes |
| False alarm rate | <= 0.10 | 0.00 | 0.00 | 0.00 | 0.00-0.00 | yes |
| Escalation correctness | >= 0.80 | 0.80 | 0.80 | 0.80 | 0.80-0.80 | yes |
| False escalation rate | - | 0.00 | 0.00 | 0.00 | 0.00-0.00 | - |
| Citation validity (quotes in reports) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00-1.00 | yes |
| Raw model quotes verbatim | - | 1.00 | 1.00 | 1.00 | 1.00-1.00 | - |

## Safety invariant

| Measure | Target | Result |
| --- | --- | --- |
| Automatic submissions or sends | 0 | 0 |

No tool exists that submits, emails or sends anything; `tests/test_no_send_tools.py` enforces it.

## Failures

- **gen_003** (ambiguous), runs 1, 2, 3: expected NEEDS_REVIEW, got READY

## Baseline comparison (model `gpt-6-luna`, 1 run(s))

### Hand-labeled cases

| Measure | Baseline | Final |
| --- | --- | --- |
| Status accuracy | 1.00-1.00 | 1.00-1.00 |
| Gap recall | 1.00-1.00 | 1.00-1.00 |
| False alarm rate | 0.00-0.00 | 0.00-0.00 |
| Escalation correctness | 1.00-1.00 | 1.00-1.00 |
| False escalation rate | 0.00-0.00 | 0.00-0.00 |
| Citation validity (quotes in reports) | 1.00-1.00 | 1.00-1.00 |
| Raw model quotes verbatim | 1.00-1.00 | 1.00-1.00 |

### Generated cases

| Measure | Baseline | Final |
| --- | --- | --- |
| Status accuracy | 0.97-0.97 | 0.97-0.97 |
| Gap recall | 1.00-1.00 | 1.00-1.00 |
| False alarm rate | 0.00-0.00 | 0.00-0.00 |
| Escalation correctness | 0.80-0.80 | 0.80-0.80 |
| False escalation rate | 0.00-0.00 | 0.00-0.00 |
| Citation validity (quotes in reports) | 1.00-1.00 | 1.00-1.00 |
| Raw model quotes verbatim | 1.00-1.00 | 1.00-1.00 |

## What changed since the baseline

**One change, made after the baseline run and before the final runs.**

- **Failure found:** `gen_005` (a garbled biomarker section, expected `NEEDS_REVIEW`) came out
  `NOT_READY`. The model correctly reported the biomarker as not found, so the garbled text
  looked like a plain missing item. No rule treated unreadable evidence as a reason to send
  the packet to a person.
- **Fix:** the extraction now returns `unreadable_passages`, verbatim text the model cannot
  read. Code keeps only passages that are exact substrings of the note (text the model made up
  is dropped), and any verified passage forces `NEEDS_REVIEW` with a reason that contains no
  note text. The reviewer sees the passages themselves. One prompt rule was added; the checks
  and the status are still decided by code.
- **Checked before the final runs:** `gen_005` now routes to `NEEDS_REVIEW`; `hand_02`, the
  abbreviation-heavy note most likely to be wrongly flagged as unreadable, stays `READY`.
- **Not changed:** the labels (`data/labels.json` is byte-for-byte the same), the checklists,
  the other prompt rules, and the targets.

**Limits of this fix:** the only unreadable case in the evaluation is a generated one, and the
generator's garbled text was visible to me when I designed the fix. The 10 hand-written cases
contain no unreadable text, so the fix has not been tested on unreadable text written by
someone else. Its cost is measured by the false escalation rate on the other cases.

**Findings from the final runs (read these before the numbers):**

- **The final numbers equal the baseline numbers by coincidence.** One case was fixed and a
  different one failed. `gen_005` now routes to `NEEDS_REVIEW` as expected. `gen_003`
  (conflicting labs) is `READY` in all three final runs although it was `NEEDS_REVIEW` in the
  single baseline run.
- **`gen_003` is not a regression from the change.** Re-running that one case five times on the
  code from before the fix and five times on the current code gave `NEEDS_REVIEW` 2 of 5 times
  both ways. The model escalates it about 40% of the time either way. The baseline was one lucky
  run and the final runs were three unlucky ones; three runs in a row can look stable by chance
  (about a one in five chance at that rate), so the flat range for this case should not be read
  as stability.
- **Escalation correctness meets its target only on the nose** (0.80, 4 of 5 ambiguous cases).
  Because `gen_003` is a coin flip, a repeat of this evaluation could give 0.60 to 0.80.
- **Whether `gen_003` is really ambiguous is a labeling question.** The note has a repeat lab
  on a later date with a different hemoglobin value. A later repeat is arguably just the recent
  lab, not a contradiction. The label was generated, not hand-written, and was deliberately left
  unchanged after seeing the result, so as not to tune the evaluation to the model.
- **`gen_013` (unreadable lab date)** is still routed to `NEEDS_REVIEW`, but its gap is now
  `recent_lab: missing` instead of `conflicting`, because the new rule tells the model to mark
  an item that depends on unreadable text as not found. The status is right; gaps on ambiguous
  cases are informational and are not scored.
- **Sample size:** 40 cases, 5 of them ambiguous, 3 runs, one model (`gpt-6-luna`). These are
  measurements on synthetic notes, not evidence about real practice.
