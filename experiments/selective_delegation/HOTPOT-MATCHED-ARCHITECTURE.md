---
question_id: SD-HOTPOT-TRANSFER
status: exploratory_completed
updated_utc: 2026-09-21T15:30:00Z
dataset: HotpotQA_official_explorer
parents: 32
repeats: 2
primary_comparator: base_direct
result: advantage_over_strongest_direct_control_uncertain
next_decision: replicate_fixed_policies_on_new_canonical_validation_questions
---

# Hotpot: the multi-call result survives one control, but remains uncertain

Using the trained model as a helper gives40/64 correct answers. Using that same
trained adapter to answer the original question directly gives31/64. But the
stronger direct baseline uses the original base weights and gets36/64. We must
not advertise the larger40-versus31 difference while omitting that baseline.

| Fixed policy | Correct /64 | Official token F1 |
|---|---:|---:|
| Base model answers directly | 36 | 65.54% |
| Helper-trained adapter answers directly | 31 | 58.57% |
| Supervised planner, trained helpers, base final | 40 | 69.11% |
| Same planner, untrained helpers with a format reminder, base final | 35 | 64.17% |

The trained multi-call policy beats base direct by6.25 percentage points, but
the paired parent-bootstrap interval is−6.25 to+20.31. It wins8attempts and
loses4; two wins involve invalid direct-answer JSON. The both-valid contrast
is6wins versus4losses. Thus this does not yet establish better reasoning or
superiority to direct reading.

Against the helper adapter used directly, the difference is+14.06points,
interval+3.13 to+26.56; all12wins and3losses have valid answers. This distinguishes
the multi-call result from simply applying the helper adapter to a direct
prompt. It does not establish that helper training generally harms direct
answering: the direct-adapter-versus-base interval itself spans zero, and
original-question prompts differ from its subquestion training distribution.

Full hypothetical deployment cost counts the64 saved root calls once, not just
the new helper/final execution. It totals273 calls and358,290tokens, versus64
calls and102,271tokens for base direct:4.27times the calls and3.50times the
tokens. Summed service time is267.27seconds versus31.57seconds; this is not a
measured deployment wall-time comparison. Training/acquisition remain separate
research costs.

## What was checked

All32parents and two repeats match. The192native answer requests across trained
helpers' finals and both direct arms have exactly matching case/repeat seeds.
There are no missing or unavailable calls. Base direct has4invalid answers;
the other two arms have64valid finals each. Official Hotpot grading includes
its exact-only rule for yes/no/noanswer. The new direct control is not reuse of
the earlier different-seed36/64 result.

Intervals use20,000parent-bootstrap draws, seed2026092176, averaging repeats
within parent. These32parents have no verified atomic-component independence.
F1 differences are+3.57points versusbase direct (−5.73 to+13.54), and+10.53
versusadapted direct (+2.62 to+18.63). They are unadjusted exploratory comparisons.

The next useful check is a fixed-policy replication on new canonical validation
questions, not another prompt sweep on these32. That panel is being prepared
without inspecting model outcomes. Its result could promote or retire this
transfer lead; current evidence is not yet a publishable architecture claim.

## Evidence pointers

- `R/analysis-helper-transfer-hotpot-001.json`: fixed-plan helper conditions.
- `R/analysis-hotpot-direct-adapted-001.json`: newly collected direct conditions.
- `R/helper-transfer-hotpot-001/calls` and `R/hotpot-direct-adapted-001/calls`:
  the matched native answer requests and outputs.
- `R/transfer-hotpot-sft-001/calls`:64saved root requests,18,420tokens added
  exactly once for full-policy cost.

Here `R` is the September21 selective-delegation research store. The independent
paired audit uses the complete case/repeat intersection, not outcome filtering.
