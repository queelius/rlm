# Fixed-endpoint SPO reproduction: 58.19% on GSM8K

Evidence cutoff: 14 October 2026, 18:10 UTC.

Our own RhoMath1.1B SFT-to-SPO run reached the predetermined 690 collections /
11,040 optimizer updates and scored **12,281/21,104 = 58.192760% mean pass@1**.
That is +18.418309 percentage points over matched SFT (8,394/21,104 =
39.774450%) and +1.492760 points over the paper's 56.7%.
This recovers the reported performance level in one training run with disclosed
implementation differences; it is not exact historical-code equivalence.

| Evaluation | Updates | Test correct / 21,104 | Test % |
| --- | ---: | ---: | ---: |
| 650 | 10,416 | 12,169 | 57.662055 |
| 660 | 10,576 | 12,182 | 57.723654 |
| 670 | 10,736 | 12,306 | 58.311221 |
| 680 | 10,896 | 12,307 | 58.315959 |
| Fixed final 690 | 11,040 | 12,281 | 58.192760 |

The final checkpoint is not the highest intermediate score. Scheduled points
remain in order; the final result was not selected by test accuracy.
Validation: 4,324/5,968. Non-held-out training subset: 5,390/5,952.
The independent audit checked all 33,024 saved responses and 21 native metrics:
original question identities and prompts, tree/Arrow equality, answer extraction
and official scoring. No final extraction failures; all responses ended with stop.

## Protocol and evidence

Single A100 allocation; original SFT initialization; author GSM8K splits
7,100/373/1,319. SPO-chain interval5, 64 questions ×8 responses, nine MC
continuations, global batch64, two update epochs, LR1e-6, seed42.
The original 1,000-collection LR horizon remained unchanged.
Final test uses 1,319 questions ×16 responses, temperature0.35,
effective top_p1.0, mean sampled-answer correctness—not majority vote,
once-hit, or greedy accuracy.

The process-local BOS-only adapter aligns training queries with generation.
The separate finite-zero BF16 adapter is also disclosed. Native EOS omission
remains; historical author implementation and checkpoint-selection details are
unresolved. Likelihood diagnostics support a mismatch, not a causal explanation
of the earlier failure. One seed does not establish training-seed robustness.
The authors' released 56.69% result validates our evaluator, not our own training.

The earlier own attempt remains a failure: 22.19% at7,856 updates, exit1 at7,995,
not allocation timeout. Its checkpoints and evidence are retained.
The fresh run exited0 without timeout. Owner processes and GPU applications were
absent at the terminal check. No successor or new RLM/CURL/DrGRPO job was launched.

- [Final all-answer audit](evidence/final690-audit-20261014.json)
- [Execution and terminal checks](evidence/final690-execution-20261014.json)
- [Fourteen-file checkpoint hash receipt](evidence/final11040-preservation-20261014.json)

Trainer/actor/active Adam counters are11,040; iteration690; scheduler11,039;
skipped steps0. The preserved full state includes tokenizer files.
All source/copy hashes match and inventories were stable. This is an independent
same-filesystem copy, not disaster backup, an all-tensor finiteness test or GPU restore.
The pinned audit retains a generic intermediate-point limitation; its explicit
wrapper semantics and independent counters identify this690 result as final.

## What transfers to RLM

This exercise supports confidence in our ability to run and audit an RL learning
loop: token alignment, reward checks, actual optimizer progress, preserved failure
evidence, fixed-endpoint evaluation, and independent scoring. It does not show
that SPO is the right RLM algorithm, nor that these gains transfer to tool use,
recursive decisions, cost-sensitive rewards or long-horizon credit assignment.
Those remain separate experimental questions. The next RLM study should carry
forward these checks, not assume the algorithm or every hyperparameter transfers.

External resume root:
`/project/alex_phd/runs/rl-paper-reproduction-20261012`.
Preserved final state:
`/home/atowell/research-runs/rl-paper-reproduction-20261012/preserved-checkpoints/spo-bos-only-from-sft-attempt1/ckpt--iter_0690--epoch_2.00--step_11040`.
Git publication does not back up these model weights.
