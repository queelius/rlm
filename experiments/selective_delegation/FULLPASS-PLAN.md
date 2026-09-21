# Does planner RL help once the helper is better?

Accepted September21 at12:09UTC, before viewing any model outputs on the new
evaluation panel. This is an exploratory follow-up, not a confirmatory study.

The [helper comparison](HELPER-FINDINGS.md) changes the next decision. With the
planner fixed, helper training improves19/64 to28/64 correct; a formatting
reminder reaches22/64. Both fixes eliminate the observed helper-format failures.
The trained helper has seven wins and one loss against the reminder, all with
valid final answers. This supports retaining its weights for the next learning
test, while recognizing the small panel and the final model's ability to repair
or bypass incorrect helper answers.

## Training and comparison

Start the root from the same supervised checkpoint48, not the previous RL run.
Freeze helper checkpoint36 and retain the original helper prompt, without the
extra formatting reminder. Keep the released final model and full documents.
Two independent4B model copies fit on the reserved A100; only the root LoRA
parameters enter the optimizer. Actual role routing and zero trainable helper
parameters were verified on the first real training trajectories.

Use the existing deterministic shuffle of all256training questions, in sixteen
disjoint groups of16. Each question gets four fresh sampled root plans. The
maximum is16updates and1,024trajectories, rather than four updates repeatedly
using the same16questions. Root temperature0.8, helper/final temperature0.5,
terminal exact-answer reward, leave-other-three-out advantages, AdamW2e-5, and
one gradient pass per fresh batch remain unchanged. No reward shaping, KL
penalty, repeated optimization of old trajectories, or helper updates are added.

The existing admission check remains: stop if a batch contains fewer than two
question groups with distinct valid plans and differing rewards among valid
plans. This can stop an informative run before a full pass; report the actual
dose and the stopped batch rather than equating admission failure with general
RL failure. Do not replace a failed block with a hand-picked easier block.

Three cumulative GPU hours is the training cap. Save an adapter, optimizer and
RNG checkpoint after every successful update. Select the last committed update
at completion or admission stop, **never the checkpoint with the best evaluation
score**. A runtime failure is preserved, not silently converted into a lower
training dose. Independent controls can still run while a failure is inspected.

## New questions and what each contrast means

Use all64questions in `fresh-dev-inputs-003`, twice each:32two-hop and32three-hop
questions. The panel excludes original-study and earlier breadth-study parents
and atomic components; this is not a claim of no pretraining exposure. Its
selection did not use model performance or answer labels. Earlier preparation
versions001/002 are superseded.

| System | Planner | Helper | Final |
|---|---|---|---|
| Released planner | Base | Fixed helper-SFT36 | Base, full documents |
| Supervised planner | SFT48 | Same fixed helper | Same base final |
| Reward-trained planner | Last committed new RL | Same fixed helper | Same base final |
| Direct answer | None | None | Base, full documents |

All evaluation uses temperature0.5 and the same frozen source, cases and seed
formula. Each readout has a one-hour cap. The primary difference is **new RL
minus SFT**, with identical downstream models. Base-planner and direct-answer
controls show whether trained planning earns its added cost. The helper-effect
study used different downstream seeds and is not reused as this matched baseline.

Count every planned attempt. Report paired component-cluster intervals,
protocol-related versus both-valid changes, native calls/tokens, and example
traces. Rewards across changing training-question blocks are not a learning
curve. No claim of successful decomposition follows merely from a correct final
answer. These are short-context QA tests, not general Python execution or
learned recursive trees.

## Execution and provenance

Study root: `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- Immutable acceptance: `FULLPASS-DECISION-001.json`, including source, input and
  checkpoint hashes, selection reasoning, caps and stopping rule.
- Sealed source: `source-013/`; do not edit beneath the live owner.
- Supervisor: `launch_fullpass.py`, tool session52519; waits for the earlier
  SFT-dose readout before acquiring the existing exclusive GPU lock.
- Training: `rl-fullpass-001`; real responses verified by12:12UTC.
- Queued evaluation: `fresh-contract-sft-001` (base and SFT),
  `fresh-contract-direct-001`, then `fresh-contract-rl-001`.
- Data: [FRESH-DEV-PANEL.md](FRESH-DEV-PANEL.md). Earlier helper results and
  transfer tests are exposed development evidence, not fresh confirmation.
