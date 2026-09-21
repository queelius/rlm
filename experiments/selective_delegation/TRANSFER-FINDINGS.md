# Can short-plan training handle longer questions?

Cutoff: September 21, 11:33 UTC. Exploratory results; direct-answer and
second-dataset transfer comparisons are still running. Checkpoints were fixed
before these outcomes were inspected.

## Four-hop MuSiQue: completed base versus supervised planner

The planner trained on two- and three-hop examples. We then evaluated it on 64
different four-hop questions, twice each. Helpers and final-answer weights did
not change, and every helper/final retained access to the original documents.

| Planner | Correct answers | Valid JSON plans | Completed final answers |
|---|---:|---:|---:|
| Released model | 27/128 | 108/128 | 84/128 |
| Supervised planner | 24/128 | 128/128 | 89/128 |

Supervised training did not improve answer accuracy on this panel. The paired
difference is −2.34 percentage points, with an exploratory parent-bootstrap
interval of −11.72 to +7.03 points. The 64 questions form only 24 connected groups
when shared atomic questions are considered; the parent interval is not a
fully independent-component uncertainty estimate. A secondary audit resampled
those whole groups and obtained an interval of −10.20 to +4.31 points. Neither
analysis establishes an improvement or a reliable decline.

Two distinct limitations matter:

- Every one of the released model's 20 invalid plans exhausted the 128-token root
  output cap. This is a fixed-budget result, not evidence that the model cannot
  produce valid JSON with more space. A larger-cap comparison is a possible
  follow-up, not an unreported repair to this run.
- The supervised model always finished its JSON, but 12 attempts on 9 questions
  stopped because a step referred to itself. None of its 256
  annotated training plans has that reference-order defect. It usually emitted
  two or three questions: 108/128 plans, versus 38/108 valid released-model plans.
  Shorter plans are not inherently wrong; a subquestion can combine several
  facts. The invalid references, rather than length alone, establish an
  execution defect.

For example, one trained plan's third question asked when a place became the
capital of `#3`. Here `#3` means the answer to that very third question, which is
not available yet. The executor correctly stopped; it did not substitute an
annotated answer or guess a missing dependency.

Helper-format failures remain common: 24 released-planner attempts and 27 trained
ones. This is why the already queued helper-training comparison remains useful,
but changing the helper cannot repair an invalid dependency in the root plan.
The structural issue may eventually justify comparing full plans with decisions
made one step at a time. Such a comparison must control information access and
cost; incremental planning itself is established prior art.

## Cost and evidence

All 1,131 calls returned; there were no missing episodes, generation failures,
unknown token counts, or unresolved start receipts. The released planner used
581 calls and 1,405,937 tokens; the supervised planner used 550 calls and 1,317,074
tokens. These are actual totals, not compute-matched runs.

External study root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`

- `analysis-transfer-musique-sft-001.json/.md`: regraded outcomes, paired changes,
  bootstrap recipe, input/source/checkpoint hashes, and overlap diagnostics.
- `analysis-transfer-structure-001.json/.md`: component-cluster intervals and
  checks of every parsed plan, including steps not reached during execution.
  The source is `audit_transfer_structure.py`; it found no additional defective
  plans hidden by earlier helper failures.
- `transfer-musique-fourhop-sft-001/calls/`: native roots, finish reasons, usage,
  helpers, and final answers.
- `transfer-musique-fourhop-sft-001/episodes/`: strict dependency failures and
  the saved partial traces, including parent `62563c117fe360c43cc0b763` above.
- `planner-sft-inputs-001/examples.jsonl`: the 256 train-only question-list targets.

## The four-update RL checkpoint also scores 24/128

The fresh RL readout completed 541 calls with no generation errors or missing
episodes. Its exact-match accuracy is unchanged from supervised training:
24/128. Five attempts improved and five worsened. Four wins and one loss had
valid answers in both conditions; the remaining changes involved protocol
failures. This differs from the earlier two/three-hop validation panel, whose
RL wins were all protocol-related. Neither pattern should be generalized beyond
its measured panel.

The paired RL-minus-SFT parent-bootstrap interval is −5.47 to +5.47 percentage
points; the same shared-component limitation applies. RL has 13 invalid
dependency attempts and 30 helper-format failures, versus 12 and 27 for SFT.
It used 1,293,232 tokens. See `analysis-transfer-musique-001.json/.md` for the
joined three-condition report. The earlier secondary component-cluster audit
concerns SFT versus base only, not this later RL comparison.

The result supports fixing execution before treating another planner update as
a likely answer improvement. It does not justify calling RL ineffective in
general: four updates on 16 repeated training questions are a very small dose.

This result does not show that decomposition or reinforcement learning cannot
work. It shows that this short supervised recipe did not establish longer-chain
answer improvement, despite making root output more compact and parseable.
