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

Readout (16 attempts per arm per world; world 2 longer-training arm still running at 05:00 UTC):

| Arm (first-fit actors, seed 2026092208) | World 2026101501 | World 2026101502 |
| --- | ---: | ---: |
| Original examples, 23 updates | 2/16 | 1/16 |
| Repaired examples, 23 updates | 6/16 | 5/16 |
| Discovery examples, 23 updates | 3/16 | 5/16 |
| Original examples, 69 updates | 8/16 | pending |

Repair beats the matched original teacher on both new worlds, as on the pilot world. It does not
beat the discovery teacher on world 2 and does not beat longer original training on world 1,
where the 69-update actor is the best arm (the opposite of the pilot world's second fit, where it
scored 0/16). The repair-versus-original contrast therefore replicates across three worlds and two
fits, while the comparison against stronger controls remains world-dependent. Lite audits agree
with the collector's native scores on every completed arm.

## Track B: learned selective recursion (GPU 1 serving, GPU 0 training after Track A)

Question: can a 4B controller inside the RLM harness learn to answer directly when that is
cheap and reliable, and to decompose (chunk, extract per-record facts with helper calls,
aggregate exactly in Python) when the input is too long for one call, under a reward of
correctness minus a token-cost penalty? Does the learned behaviour transfer to OOLONG-synth?

Testbed: `experiments/selective_recursion_20261019/taskgen.py` generates store activity logs
of N natural-language records with negated and filler distractors, and five exactly verifiable
question types (distinct buyers, people who bought both items, net units, bought-then-returned,
and a single-record lookup of a purchase day). Length N is the main knob.

Baselines at temperature 0 with Qwen3-4B-Instruct-2507 (sweep-001-t0 and sweep-002; 32 tasks per
cell for the four aggregation questions; accuracy, mean total tokens in parentheses):

| Policy | N=8 | N=32 | N=64 | N=128 | N=256 | N=512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct, one call | 0.75 (0.8k) | 0.47 (2.0k) | 0.22 (3.6k) | 0.06 (6.6k) | 0.00 (10.3k) | 0.00 (16.6k) |
| Base controller, default harness | 0.84 (3.8k) | 0.53 (5.6k) | 0.06 (7.4k) | 0.00 (8.9k) | 0.00 (15.8k) | 0.00 (24.4k) |
| Base controller, hinted harness | 0.84 (4.0k) | 0.47 (5.5k) | 0.03 (7.6k) | 0.00 (12.2k) | 0.00 (15.6k) | 0.00 (21.3k) |
| Programmed split (12-line chunks, per-line extraction, exact aggregation) | 0.97 (0.8k) | 1.00 (3.2k) | 0.97 (6.2k) | 0.97 (12.3k) | 0.75 (24.7k) | 0.84 (49.7k) |
| Scripted expert inside the harness (dev split, 16 per cell) | 1.00 (0.8k) | 1.00 (3.3k) | 0.81 (6.2k) | 1.00 (12.5k) | 0.81 (24.9k) | 0.88 (49.6k) |

Lookup questions (8 per cell): direct answers 1.00 / 0.86 / 1.00 at N=8/32/64 for 0.5k to 2.2k
tokens; the scripted expert answers 1.00 at every length for about 0.3k tokens by filtering lines in
Python and asking one helper; the base controller reaches 0.25 to 0.50 at N=256 to 512.

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

Transfer target: OOLONG-synth validation rows up to 32k tokens (`oolong_eval.py`; exact match
on the requested answer line with any key prefix, not the official scorer). Within each context
length the distinct context windows are split train/test, so every length has held-out windows.
A second scripted expert (`oolong_cells/`) classifies lines with helpers only when the question
needs labels and answers user/date counting questions in code alone; with the dataset's gold
labels substituted for the helper its logic matches 595/598 answers (OOLONG compares label shares
before/after a date). Held-out OOLONG results so far (accuracy, mean total tokens):

| Policy | 1k | 2k | 4k | 8k | 16k | 32k | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct (12 per length) | 0.50 (1.7k) | 0.33 (3.9k) | 0.33 (6.8k) | 0.08 (11.2k) | 0.00 (19.0k) | 0.08 (32.8k) | 0.22 (12.6k) |
| Base controller (12 per length) | 0.25 (5.0k) | 0.08 (10.1k) | 0.17 (11.7k) | 0.00 (13.4k) | 0.00 (19.8k) | 0.17 (33.5k) | 0.11 (15.6k) |
| Scripted expert in harness (43 to 50 per length) | 0.60 (0.5k) | 0.68 (1.4k) | 0.74 (3.1k) | 0.74 (6.5k) | 0.68 (14.2k) | 0.58 (27.6k) | 0.67 (9.1k) |

The expert's remaining errors are mostly six-way TREC classification by the 4B helper (counting
group 0.53 versus 0.82 to 0.84 for timeline and user groups). The teaching stage therefore has a
ceiling of roughly two thirds on this benchmark with this helper.

Run store: `/project/alex_phd/runs/selective-recursion-20261019/`.

## Limits to keep in view

Synthetic templates and vocabulary; one model family and size; one training seed per stage
until a signal appears; cost is counted in tokens reported by the serving endpoint; the
programmed split is a strong task-specific baseline written with knowledge of the generator.
