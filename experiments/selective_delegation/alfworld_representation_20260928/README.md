# ALFWorld: transient index versus exact native command

CPU preparation, September28,2026. Parent dispatches; no GPU is launched by preparation.
Read [READINESS.md](READINESS.md) for the evidence and primary-source novelty boundary.

The experiment asks whether the action-target representation changes learning gain, not whether
hierarchy works. It reuses the authenticated original index-SFT checkpoint33 and trains one command
checkpoint from the same released Qwen3-4B,524 successful-game TRAIN demonstrations, row order,
seed2026092200,33 updates,LR1e-4 and rank8/alpha16/dropout0 LoRA. BF16 base and FP32 adapters match
the old recipe; the original optimizer/loss/checkpoint implementation is reused unchanged.

Both modes show the exact same indexed admissible-command list and full initial/current/history
public context. Index output is `{"action_index":N}`; command output is `{"command":"exact text"}`.
Command matching is literal: no casing, whitespace, approximate matching or hidden expert repair.
Native commands, inventory/transition semantics, public rejection history and three-invalid cutoff
are unchanged. Neither mode trims history; a context overflow remains unknown.

There are four own-interface readouts: index-base, index-trained, command-base, command-trained.
All96 episodes use12 prospectively selected official valid_unseen games×two seeds. Selection is
the first two games per family by SHA256(`2026092812:relative_path`) after excluding prior panels;
all12 native resets pass, with no replacement/outcome filtering. The games are new to the
campaign's prior panels, but cover only the same four scene IDs10/219/308/424. Readout seeds are
2026092813/14. No old episode outcomes are reused as new controls.

## Limits and execution

Per episode:50 native actions,2048 generated tokens,128 tokens per response,8192 context,
temperature0.5/top-p1/top-k0. One new SFT cap20min and four readout caps24min each:116 GPU minutes
maximum; useful estimate1–2h. Original SFT actually took287s; old24-slot flat cells took443–759
native seconds. Longer exact-command outputs can increase latency. The parent supervisor adds
two minutes per GPU stage and20 minutes for CPU audits/comparison,146 minutes overall.

External study: `R/alfworld-representation-20260928-001`, where R is the active
selective-delegation research store. `PREPARED-JOBS.json` has absolute commands and pins. It is not
an accepted scheduler; parent must queue after previous owners release the shared coordinator lock.

```bash
GPU_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
SOURCE=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/alfworld_representation_20260928
"$GPU_PY" "$SOURCE/alf_rep_train.py"
"$GPU_PY" "$SOURCE/alf_rep_collect.py" --mode index --actor base
"$GPU_PY" "$SOURCE/alf_rep_collect.py" --mode command --actor base
"$GPU_PY" "$SOURCE/alf_rep_collect.py" --mode index --actor trained
"$GPU_PY" "$SOURCE/alf_rep_collect.py" --mode command --actor trained
CUDA_VISIBLE_DEVICES= "$GPU_PY" "$SOURCE/alf_rep_compare.py"
```

New command checkpoint hashes remain unknown until authenticated complete step33. A missing or
failed command endpoint does not suppress the other three cells or become an observed failure.
One GPU training attempt only; the old trainer's resume switch merely admits its prewritten PLAN,
and the wrapper rejects existing owners. Checkpoints at0/8/16/24/32/33 preserve failed boundaries.

## Interpretation and verification

Primary: native-win own-interface gains and their interaction
`(command-trained-command-base)-(index-trained-index-base)`, clustered by game and scene.
Report all unknowns, invalid outputs, caps, repeated semantic commands, physical calls,
prompt/output tokens and measured native latency. Full native action replay runs on CPUs after
GPU collection. Every saved response is retokenized/decoded and authenticated before any host-only
adapter-neutral projection for reuse of the original receipt auditor.

Same optimizer updates do **not** equal token exposure: command supervision is5362 tokens versus
4060 indexed tokens (+32.1%). The original token-mean loss also changes per-example weighting as
target length changes. Command prompts total712335 tokens versus709191. Thus a positive interaction
supports the representation/training package, not a pure symbol-binding mechanism. A better
command-base score alone is not evidence of improved learning. No manager, RL or automatic dose
expansion follows this screen.

Eight focused tests exercise strict equality/duplicate rejection, identical contexts,524 exact
ordered targets, outcome-blind selection, no trimming, target validation, real saved native replay
and missing-aware gain arithmetic. The external final fixture is `runtime-fixture-002`: a scripted
TRAIN trajectory in both interfaces, each independently replayed in a fresh TextWorld process.
The command case additionally charges an invalid uppercase command without changing native state.
These fixtures are not model competence measurements.

`runtime-fixture-001` is preserved: its scientific seam passed, but later addition of initial-context
qualification changed the pinned job-preparation script. Fixture002 rebinds the final source set.
No accepted source, old run, environment, GPU job or Git state was modified by this task.
