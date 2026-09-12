---
date: 2026-09-12
cutoff_utc: "2026-09-12T05:22:00Z"
question: rq:rl-effective-feedback
status: helper_intervention_and_replay_complete
evidence_level: exploratory_two_training_contexts
---

# Correct helper information substantially improved this diagnostic

The most useful new finding is that improving the helper's local answers made
a large difference without training the root model. This gives us a concrete
research target: improve the information returned by subtasks, then check whether
the complete solution improves too.

## The experiment in plain language

The helper reads records and assigns categories. The root writes Python that
uses those categories to filter records, count them, or add their weights.
We compared three ways of supplying the helper's reply while keeping the root
weights and requested tasks unchanged:

| Helper information supplied to the root | Correct final answers | Wrong final answers |
|---|---:|---:|
| The helper model's original replies | 26 of 48 | 22 |
| The same saved replies, delivered again | 27 of 48 | 21 |
| Correct categories supplied by the evaluator | 44 of 48 | 4 |

Every attempt had an observed final response. The last condition supplies only
local categories, not the answer to the root's question. The root must still
choose the requested records and perform the calculation. It is a diagnostic
use of known labels, **not a deployable system or a learned model improvement**.

The saved-reply control matters because replacing a model call with a callback
also changes delivery and timing. It preserves the original helper content while
using the same delivery mechanism as the correct-label condition. Correct labels
produced 18 wins and one loss relative to that control, a net gain of 17 answers.

## What did we learn about the remaining mistakes?

Two correct-label failures counted selected records when the question asked for
their total weight. For example, selecting two records is not enough: their
weights may sum to five, so the answer should be five, not two.

The other two failures involved Python errors and repeated unsuccessful repairs.
One tried to put dictionaries into a set; another repeatedly generated invalid
Python syntax. Better helper answers therefore do not eliminate root-side
calculation and recovery problems.

Importantly, these four failures were not the same four cases that previously
disagreed with the calculation on their helper replies. Changing the helper
information can change the root's later program. The earlier error breakdown was
a useful lead, not a fixed partition or a prediction that 44 was an upper bound.

## How strong is this evidence?

The effect is much larger than the single answer changed by replaying the old
information. However, these are 12 questions from only two training contexts,
each sampled four times. We have not demonstrated transfer to new documents,
new task families, or another dataset.

Matched seeds also did not produce perfectly identical continuations. In the
replay control, one answer changed even though the initial root reply and the
first post-helper input were identical. In the correct-label versus replay
comparison, initial root replies differed on five attempts. We preserve these
differences and do not present the count difference as an exact paired causal
estimate or proof that every generated program was executed faithfully.

The real-helper condition used 182 root calls and 48 child calls. Replay used
148 root calls, and the correct-label condition used 191. The latter two used
zero child-model calls and 48 callbacks each. These are different kinds of work;
we cannot claim a deployable speedup from the diagnostic callback timings.

## What happens next?

1. Train the helper using local label correctness while leaving the root unchanged.
   The first numerical qualification attempt stopped before updating weights:
   its effective sample size was 36.6 out of 48, below the preset 38.4 minimum.
   All sampled actions were supported, all probabilities were finite, and the
   largest-weight check passed. This is a numerical qualification result, not
   evidence that helper RL improves or harms answers. We are reviewing an explicit
   exploratory continuation; the original no-update record stays intact.
2. Test a small harness change: ask the helper to process fewer records per call,
   merge its replies, and measure both accuracy and actual computation. Our earlier
   batch-size work makes this a practical alternative to changing model weights.
3. Repeat the [self-SFT comparison](2026-09-12-learning-comparison.md) with new
   sampling draws, then move promising changes to new source contexts and datasets.

These tests connect to the broader decomposition question: how much work should
we give each subtask, and when is further checking or splitting worth its cost?
We have not yet learned that decision or demonstrated general recursive planning.

## Evidence and current ownership

Under `/project/alex_phd/runs/rlm-research-r4`:

- `analyses/qs6-real-oracle-replay-endpoints-2026-09-12/OUTPUT_V1.json` is the
  authenticated three-condition result. SHA-256:
  `d91aea95bead75bd1cf2e7224e94cf58f4e5b0ca5c4f29ff69cd6054555e7cb1`.
  MAIN independently rebuilt it and checked exact equality.
- `analyses/qs6-oracle-mechanism-2026-09-12/REPORT.md`, `MECHANISM.json`, and
  `THREE_ARM.md` preserve the program examples, paired outcomes and limitations.
- `sidecars/root-qs6-leaf-rloo-onebatch-v1/outputs/attempt-001/` preserves the
  failed pre-update qualification. No optimizer step or trained checkpoint exists
  for this attempt.
- `operations/2026-09-12-after-leaf-selfsft-repeat/outputs/attempt-001/STATUS.json`
  records the active fresh-seed comparison. Its repaired, accepted receipt is
  `READY_V2.json`; a pre-launch receipt-key mismatch prevented the earlier version
  from starting. The fixed supervisor launched at 05:21:22 UTC.

Read the live `RESEARCH_QUEUE.md` before launching anything. The delivered meeting
deck remains at its September 11 cutoff; this report preserves the new evidence
for the next update. Git backs up this document, not the external model/run store.
