---
date: 2026-10-05
decision_fixed_utc: 2026-10-05T17:50:00Z
status: admitted_waiting_for_predecessor_gpu_lock
question: Do the early benefits of image matching and its smaller update repeat in fresh training seeds?
task: cartpole/swingup
training_seeds: [234, 567, 890]
arms: [curl, single_encoder_curl, no_curl]
primary_endpoint: 100000_training_simulator_steps
external_runs: /project/alex_phd/runs/curl-cartpole-fresh-seeds-20261005
---

# Does the result repeat when we train new models?

**Execution update, October 5 at 18:08 UTC:** the unchanged nine-job queue is
admitted and waiting behind the final encoder-strength run. Resources, source
hashes, owner identity and the shared lock were checked. No new model has
started training at this cutoff. The design below was fixed at 17:50 UTC,
before the third seed's smaller-update endpoints. Admission and launch receipts
live in the external run store; the sealed protocol there is unchanged.

## Reasoning at the design decision

The original three cartpole models all benefited from correct image matching
relative to the same learner without that exercise. But our smaller-update
follow-up is mixed so far: it helped the first seed and hurt the second.
Wrong picture matches remain harmful in both completed smaller-update seeds.
The final seed's two models are still unfinished at this design decision.

The next question is about **reproducibility and variation between trained
models**, not finding a setting that wins once. Three fresh seeds provide a
small, informative check; they do not establish a reliable ranking by themselves.

## Fixed comparison

Train nine fresh models, with no resumed weights or replay data. Each seed
gets all three conditions below. Keep the verified trainer and environment
unchanged, along with random image crops and the reward-learning algorithm.

| Condition | What the model learns from |
|---|---|
| Original CURL | Task rewards and matching two views of the same picture, using the original update rule. |
| Smaller-update CURL | The same rewards and picture matches, skipping only the extra dedicated encoder optimizer step. |
| No matching | Task rewards with the same random image crops, but no matching exercise. |

For every seed, report all three differences: original minus no matching,
smaller minus no matching, and smaller minus original. Use the fixed endpoint,
not the best checkpoint. Keep all failed and incomplete attempts visible.

Each model gets 100k training simulator steps: 12,500 decisions with action
repeat 8, including 1,000 initial collection decisions and 11,500 learning
updates. Evaluate every 500 decisions and at zero on the ten declared starts
10000--10009. The score is mean total reward over these ten tests, not a
percentage. Evaluation uses 260k additional interactions per model. Test
episodes are not independent training replicates.

Fix this rotated order before any new results:

| Seed | First | Second | Third |
|---|---|---|---|
| 234 | Original CURL | Smaller-update CURL | No matching |
| 567 | Smaller-update CURL | No matching | Original CURL |
| 890 | No matching | Original CURL | Smaller-update CURL |

Run all nine regardless of intermediate rewards or the remaining outcomes
from the previous batch. No seed or condition is selected using those results.

## What this can establish, and what it cannot

If the original benefit repeats, that adds evidence for the early benefit of
matching beyond crops in this cartpole setup. If the smaller update continues
to have mixed effects, that weakens a claim that it consistently improves the
original method. New reversals are informative, not reasons to discard seeds.
Do not claim equivalence from small average differences.

Report the added cohort separately first. An optional combined six-seed view
must include every original and added seed, be labeled secondary, and disclose
that the added cohort was chosen after examining earlier evidence. This is an
adaptive replication, not an untouched confirmatory study. Fixed shared test
starts do not establish robustness to other starting states or tasks. Equal
training seeds do not keep later experience identical after policies diverge.

We omit additional wrong-target runs here because their harm is already observed
under both update strengths, while the useful correct-matching effects are
uncertain. This new cohort therefore does **not** replicate the interaction
between matching correctness and update strength. The complete earlier
two-by-two comparison remains necessary and will not be interrupted.

## Readiness, resource limits and admission

The source is byte-identical to the verified encoder-strength snapshot, based
on implementation commit `f059d99a163f22baa5f05516db5a30d96e059425` and pinned
upstream commit `8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc`. No learner changes
or new environment are needed. A native-config scan found no previous cartpole
training at seeds 234, 567 or 890 in the existing CURL campaign roots. Reusing
these numbers from walking does not reuse cartpole experience or weights.

Budget approximately 120--150 minutes expected, 2,700 seconds per model
(6.75 hours summed caps), and 27 GB of additional checkpoint space. Save full
state at the existing 900-second target on natural boundaries and at completion.
Allocation deadline is epoch 1791387366. Preserve failed attempts and partial
checkpoints; do not overwrite a run with the same identifier.

Prepare immutable inputs on CPUs now. Do not launch a competing owner while
the current batch has an unlaunched job. Admit the waiting successor only
after the final current job has launched, or after its whole batch ends.
Training must wait on the same shared GPU lock. Before admission, recheck
space, account reserve, remaining allocation, input hashes and actual owner
identity. Add the new root to a successor observer configuration without
editing the live monitor's source. Inspect the first real response promptly.

`QUEUE.json`, `SOURCE.json`, and `HANDOFF.md` in the external store preserve
the inputs and preparation state. Preparation is not admission or completion.
This protocol received a read-only independent CPU review; it introduced no
GPU operations. The reviewer recommended the nine-run comparison over fifteen
runs that repeat the wrong-target conditions, and stressed separate-cohort
reporting and the limits of only three new seeds.
