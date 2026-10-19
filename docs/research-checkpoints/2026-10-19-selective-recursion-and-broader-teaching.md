---
date: 2026-10-19
status: live_session_checkpoint
allocation: "Slurm 6225, two A100 40GB (MIG 7g.40gb each), 2026-10-19 00:39 to 2026-10-22 00:39 UTC"
tracks: ["A: broader-world teaching transfer (approved pending comparison)", "B: learned selective recursion under a cost-aware verifiable reward"]
---

# Two tracks for the two-GPU window

The user approved two things on October 19: run the pending two-world teaching comparison
that the previous session had left awaiting approval, and commit the remaining allocation to a
new study of whether a small controller can learn *when* to decompose a task under a verifiable,
cost-aware reward. This note records the decisions and the calibration evidence so far. Results
tables will be added as runs complete; nothing here is a confirmed claim.

## Track A: teaching repair on two more recipe worlds (GPU 0)

Design as declared before approval: eight native-feasibility-checked roots, four per new world
(seeds 2026101501 and 2026101502, depth strata 3/4), evaluated by the four fixed first-fit
actors (original 23 updates, repaired 23, discovery 23, original 69; training seed 2026092208).
One deviation: the sealed collector evaluates one world per run and the sealed audit accepts only
16-episode outputs, so each (arm, world) run uses four rollout draws per task (seeds
2026101503-06), giving 128 attempts instead of the proposed 64. The sealed eight-task audit
cannot run on four-task panels; `experiments/teaching_broader_20261019/audit_lite.py` recomputes
every win from saved final inventories and reconciles plan identity and call records instead.

Store: `/project/alex_phd/runs/rlm-teaching-goal-20261014/broader-structure-001/`
(`BROADER-QUEUE.json`, `world*/eval-*/SUMMARY.json`, `LITE-AUDIT.json`).

Partial readout, world 2026101501 (16 attempts per arm): original23 2/16, repair23 6/16,
discovery23 3/12 so far. World 2 and the longer-training arm are pending.

## Track B: learned selective recursion (GPU 1 serving, GPU 0 training after Track A)

Question: can a 4B controller inside the RLM harness learn to answer directly when that is
cheap and reliable, and to decompose (chunk, extract per-record facts with helper calls,
aggregate exactly in Python) when the input is too long for one call, under a reward of
correctness minus a token-cost penalty? Does the learned behaviour transfer to OOLONG-synth?

Testbed: `experiments/selective_recursion_20261019/taskgen.py` generates store activity logs
of N natural-language records with negated and filler distractors, and five exactly verifiable
question types (distinct buyers, people who bought both items, net units, bought-then-returned,
and a single-record lookup of a purchase day). Length N is the main knob.

Calibration at temperature 0 with Qwen3-4B-Instruct-2507 (sweep-001-t0, 32 tasks per cell,
aggregate questions, partial):

| Policy | N=8 | N=32 | N=64 | N=128 | tokens at N=128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, one call | 0.75 | 0.47 | 0.22 | 0.06 | 6.6k |
| Base RLM controller (default harness) | 0.84 | 0.53 | 0.06 | 0.00 | 8.9k |
| Programmed split (12-line chunks, per-line extraction, exact aggregation) | 0.97 | 1.00 | 0.97 | 0.97 | 12.3k |

Observations that shaped the design:

- The base controller delegates the whole request (`model_complete(request)`) in essentially
  every rollout, including at temperature 1; spontaneous chunking appeared in 0 of 48 probe
  rollouts, helper use in 2. Pure RL from the base would have no decomposition to reinforce,
  so a teaching stage is used first.
- Per-line exhaustive extraction (one JSON object per input line, including "none") raised
  helper recall from 0.74 to 0.99; free-form extraction skipped ordinary purchase lines.
- The programmed split is as cheap as direct answering at N=8 because the direct model reasons
  at length, so for aggregation questions "always decompose" would be optimal. The lookup
  question type was added so that routing remains a real decision: a filtered single helper
  call answers lookups for a few hundred tokens at any length.
- The kernel shows the controller only the bootstrap plus the latest assistant/observation
  pair; a scripted expert therefore reads its stage from the observation namespace.
- The per-cell IPython execution cap of 120 s caused failures at N=512 under serving load; it
  is raised to 540 s with a 900 s run deadline for every policy.

Teaching: a deterministic expert controller (`teach.py`, cells in `expert_cells/`) is run through
the real kernel, so every recorded controller request is a genuine harness context and every
helper reply is a real model output. Expert accuracy so far is 190/198 across N=8..256, at
roughly 0.7k tokens (N=8) to 12k tokens (N=128). Correct trajectories become supervised pairs
(`teach-001/sft.jsonl`), the LoRA warm start (`sft_train.py`), and then on-policy RL
(`rl_train.py`, REINFORCE with leave-one-out baseline, reward = correct − 0.1 × tokens/10k with
a 50k-token cap, adapter hot-swapped into vLLM after every update).

Transfer target: OOLONG-synth validation rows up to 32k tokens
(`oolong_eval.py`; lenient exact match on the requested answer line, not the official scorer).
Base direct answered 4/8 and the base controller 0/8 on a smoke sample at 1k-4k tokens.

Run store: `/project/alex_phd/runs/selective-recursion-20261019/`.

## Limits to keep in view

Synthetic templates and vocabulary; one model family and size; one training seed per stage
until a signal appears; cost is counted in tokens reported by the serving endpoint; the
programmed split is a strong task-specific baseline written with knowledge of the generator.
