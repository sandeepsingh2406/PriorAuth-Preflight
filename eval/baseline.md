# Evaluation results

Model: `gpt-6-luna`. Runs: 1. Cases per run: 10 hand-labeled, 30 generated.

Targets come from SPEC section 2. Numbers are as measured; none were adjusted. Gap recall and false alarm rate use non-ambiguous cases only; ambiguous cases are scored on whether they were routed to review.

## Hand-labeled cases

| Measure | Target | Run 1 | Range | Target met |
| --- | --- | --- | --- | --- |
| Status accuracy | - | 1.00 | 1.00-1.00 | - |
| Gap recall | >= 0.90 | 1.00 | 1.00-1.00 | yes |
| False alarm rate | <= 0.10 | 0.00 | 0.00-0.00 | yes |
| Escalation correctness | >= 0.80 | 1.00 | 1.00-1.00 | yes |
| False escalation rate | - | 0.00 | 0.00-0.00 | - |
| Citation validity (quotes in reports) | 1.00 | 1.00 | 1.00-1.00 | yes |
| Raw model quotes verbatim | - | 1.00 | 1.00-1.00 | - |

## Generated cases

| Measure | Target | Run 1 | Range | Target met |
| --- | --- | --- | --- | --- |
| Status accuracy | - | 0.97 | 0.97-0.97 | - |
| Gap recall | >= 0.90 | 1.00 | 1.00-1.00 | yes |
| False alarm rate | <= 0.10 | 0.00 | 0.00-0.00 | yes |
| Escalation correctness | >= 0.80 | 0.80 | 0.80-0.80 | yes |
| False escalation rate | - | 0.00 | 0.00-0.00 | - |
| Citation validity (quotes in reports) | 1.00 | 1.00 | 1.00-1.00 | yes |
| Raw model quotes verbatim | - | 1.00 | 1.00-1.00 | - |

## Safety invariant

| Measure | Target | Result |
| --- | --- | --- |
| Automatic submissions or sends | 0 | 0 |

No tool exists that submits, emails or sends anything; `tests/test_no_send_tools.py` enforces it.

## Failures

- **gen_005** (ambiguous), run 1: expected NEEDS_REVIEW, got NOT_READY
