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

Readout (16 attempts per arm per world; complete at 05:27 UTC, receipt in `2026-10-19-teaching-broader-worlds.json`):

| Arm (first-fit actors, seed 2026092208) | World 2026101501 | World 2026101502 |
| --- | ---: | ---: |
| Original examples, 23 updates | 2/16 | 1/16 |
| Repaired examples, 23 updates | 6/16 | 5/16 |
| Discovery examples, 23 updates | 3/16 | 5/16 |
| Original examples, 69 updates | 8/16 | 3/16 |

Repair beats the matched original teacher on both new worlds, as on the pilot world. Against the
discovery teacher it ties or leads by small margins (6 vs 3, 5 vs 5). Against 69 updates of original
training it loses on world 1 (6 vs 8) and wins on world 2 (5 vs 3); that control is itself world-
and fit-sensitive (8/16 and 3/16 here, 5/16 and 0/16 on the pilot world's two fits). Pooled over the
two new worlds: original 3/32, repaired 11/32, discovery 8/32, original-69 11/32. The narrow
repair-over-original effect therefore replicates across three worlds and two fits; superiority to
the stronger controls does not. Lite audits agree with the collector's native scores on every arm,
and all eight queue jobs exited 0. Decision: this closes the approved comparison. The teaching line
is worth a short workshop-style write-up of the replicated contrast with its controls; it does not
justify further GPU time in this allocation.

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

## Teaching results so far (07:10 UTC)

| Warm start | Teaching data | Event-log dev (aggregation, N=8..512) | Event-log dev (lookup) | OOLONG held-out |
| --- | --- | --- | --- | --- |
| SFT-001 | event-log expert only (533 pairs) | 1.00 / 0.94 / 0.81 / 1.00 / 0.75 / 0.81 (same cost as expert) | 1.00 at every N, 2.6k tokens | 0/72; 45 runs never submit |
| SFT-002 | event-log + OOLONG expert (995 pairs) | 1.00 / 1.00 / 0.81 / 1.00 / 0.81 / 0.81 | 1.00 | 0/72 (72 traced: always the event-log inspect cell first) |

Two findings. First, a 4B controller fine-tuned on about 500 native-replay trajectories executes
the taught decomposition exactly as well as the scripted expert in distribution, including the
routing between a cheap filtered lookup and chunked extraction, where the untrained controller and
direct answering score 0 at 256 and 512 records. Second, narrow imitation does not transfer across
input formats: the event-log-only controller scores 0 on OOLONG, below the untrained controller.
The mixed warm start (SFT-002) also failed on OOLONG, and the traces show why: the controller's
first turn sees only bootstrap metadata, so an expert whose first cell depends on the task family
teaches a blind guess, and the majority template wins at temperature 0. This is a teaching-design
flaw of mine, not a model limitation. The corrected "unified" expert (`expert.py`) prints the
input's head and tail first and chooses the family-specific program from that observation; both
families were re-taught with it and with diversified program variants (chunk size 6/12/24, two
prompt phrasings, two lookup filters). SFT-004 trains on that data (580 + 456 pairs from correct
runs) and RL starts from it. Diversified variants have real cost/accuracy differences for RL to
exploit: on event-log tasks the 24-line chunk with the second prompt reaches 0.97 at the lowest
cost, while 6-line chunks are less accurate and exceed the helper-call budget at 512 records.

## Unified-peek warm starts (08:05 UTC)

Two more warm starts were trained on the corrected teaching (family-agnostic peek first, diversified
variants). SFT-004 used the original long OOLONG aggregation program; SFT-005 used short
question-specific aggregation cells composed by the expert (optional subset filter plus one 5-12 line
block; their logic matches 595/598 gold answers with gold labels).

| Controller | Event-log dev, aggregation (N=8..512) | OOLONG held-out, by length 1k/2k/4k/8k/16k/32k | OOLONG all (tokens) |
| --- | --- | --- | --- |
| Direct answering | 0.75 / 0.47 / 0.22 / 0.06 / 0.00 / 0.00 | 0.50 / 0.33 / 0.33 / 0.08 / 0.00 / 0.08 | 0.22 (12.6k) |
| Base controller | 0.84 / 0.53 / 0.06 / 0.00 / 0.00 / 0.00 | 0.25 / 0.08 / 0.17 / 0.00 / 0.00 / 0.17 | 0.11 (15.6k) |
| Scripted expert in harness | 1.00 / 1.00 / 0.81 / 1.00 / 0.81 / 0.88 | 0.60 / 0.68 / 0.74 / 0.74 / 0.68 / 0.58 | 0.67 (14.4k) |
| SFT-004 (unified peek, monolithic aggregate) | 1.00 / 1.00 / 0.88 / 1.00 / 0.69 / 0.88 | 0.08 / 0.00 / 0.08 / 0.08 / 0.08 / 0.08 | 0.07 (10.6k) |
| SFT-005 (unified peek, composed cells) | 1.00 / 1.00 / 0.81 / 0.94 / pending / pending | 0.58 / 0.33 / 0.75 / 0.58 / 0.58 / 0.17 | 0.50 (9.5k) |

SFT-004 peeks and classifies correctly but cannot reproduce the 3k-character aggregation program
from memory: 12 of its 25 inspected OOLONG failures contain syntax or name errors from drifting
mid-program. SFT-005, taught the same behaviour as short per-question programs, reaches 0.50 on
held-out OOLONG contexts with no code errors in the inspected trajectories, above direct answering
(0.22) and the untrained controller (0.11), at lower token cost than direct answering, and below the
expert's 0.67. Its 32k-token result (0.17 at only 5.9k tokens) comes from choosing 12-line chunks,
which exceeds the helper-call budget at that length; the diversified teaching also contains the
24-line variant that fits, so this is a length-dependent choice the cost-aware RL can learn. The
lesson for teaching tool-using controllers: program length matters more than program count; a
learner reproduces many short programs selected by the question more reliably than one long one.
On the event-log family SFT-004 and SFT-005 match the expert and pick the cheaper 24-line chunking
at temperature 0 (10 calls at 128 records versus 17 for SFT-001).

Mixed-task RL (`rl-mixed-001`) starts from SFT-005: 16 tasks per step (8 event-log, 8 OOLONG train
windows up to 8k tokens), 4 samples per task at temperature 1, reward = correct − 0.1 × tokens/10k,
REINFORCE with a leave-one-out baseline, LoRA updates on GPU 0 and adapter hot-swap into vLLM.

## First RL result (10:45 UTC)

`rl-mixed-001`: 30 updates from SFT-005, 64 rollouts per update (16 tasks × 4 samples; half
event-log tasks, half OOLONG training windows up to 8k tokens), temperature 1, reward
correct − 0.1 × tokens/10k, REINFORCE with a leave-one-out baseline, about 3.5 minutes per update
(rollouts on GPU 1, LoRA update on GPU 0, adapter hot-swapped into vLLM). Every task group had
reward variance from the first update (cost differences between program variants even when all
samples are correct). The small mixed dev set (46 tasks) stayed between 0.87 and 0.93 throughout,
so the full held-out evaluations decide:

| Controller | Event-log dev, aggregation 96 tasks (tokens) | Event-log lookups | OOLONG held-out 72 rows by length 1k/2k/4k/8k/16k/32k | OOLONG all (tokens) |
| --- | --- | --- | --- | --- |
| SFT-005 (warm start) | 86/96 (19k) | 24/24 | 7 / 4 / 9 / 7 / 7 / 2 of 12 | 36/72 = 0.50 (9k) |
| RL step 10 | 86/96 (21k) | 24/24 | 7 / 7 / 8 / 7 / 7 / 6 | 42/72 = 0.58 (14k) |
| RL step 30 (final) | 88/96 (20k) | 24/24 | 7 / 7 / 8 / 7 / 7 / 5 | 41/72 = 0.57 (14k) |
| Direct answering | 20/96 | 23/24 | 6 / 4 / 4 / 1 / 0 / 1 | 16/72 = 0.22 (13k) |
| Untrained controller | 14/96 | 24/24 | 3 / 1 / 2 / 0 / 0 / 2 | 8/72 = 0.11 (16k) |

RL raised held-out OOLONG accuracy from 0.50 to 0.57 while leaving the in-distribution event-log
results essentially unchanged. The gain is concentrated at 32k tokens, a length never seen in RL
training: SFT-005 chose 12-line chunks there and exceeded the helper-call budget in 9 of 12 runs
(cheap failures at 5k tokens), whereas the RL policy completes those runs (about 58 helper calls,
34k tokens) and solves 5 to 6 of 12. The RL policy also uses more helper calls on event-log tasks
(17 instead of 10 at 128 records), costing about 10 percent more tokens for the same accuracy, so
the cost term at 0.1 per 10k tokens is not yet shaping behaviour strongly. Pending: the same run
without the cost term (`rl-mixed-nocost-001`, running), a second seed (`rl-mixed-002`) and a
stronger cost term (`rl-mixed-lam03-001`), each followed by the same held-out evaluations. Twelve
rows per OOLONG length is a small sample; the 32k difference (2 versus 5 to 6 of 12) is the only
per-length change outside ordinary variation.

## Full-sample OOLONG comparison (12:10 UTC)

All 293 held-out OOLONG rows (43 to 50 per length), Wilson 95% intervals, mean total tokens:

| Controller | 1k | 2k | 4k | 8k | 16k | 32k | All | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Scripted expert in harness | 26/43 | 34/50 | 37/50 | 37/50 | 34/50 | 29/50 | 0.672 [0.62, 0.72] (14k) | 0 |
| SFT-005 warm start | 21/43 | 25/50 | 35/50 | 35/50 | 38/50 | 12/50 | 0.567 [0.51, 0.62] (10k) | 31 |
| SFT-005 + RL, 30 updates | 26/43 | 33/50 | 36/50 | 36/50 | 33/50 | 27/50 | 0.652 [0.60, 0.70] (14k) | 5 |
| Direct answering, one call | 22/43 | 14/50 | 11/50 | 8/50 | 2/50 | 4/50 | 0.208 [0.17, 0.26] (12k) | 0 |

Paired over the same rows, RL gains 39 and loses 14 against the warm start (152 both right, 88
both wrong). RL training saw OOLONG contexts of at most 8k tokens; the largest gain is at 32k,
where the warm start's helper-call budget failures (31 errors overall, nearly all at 32k) are gone.
By task group RL improves counting (65 to 84 of 156) and user questions (50 to 62 of 75) and loses
some timeline questions (51 to 45 of 62). The RL controller is within the expert's interval at the
expert's cost, and more than three times the accuracy of direct answering at a similar cost.
Helper-call counts show what RL changed: the warm start used more helper calls than the expert at
every length (14 versus 9 at 4k tokens, implying small chunks) and the RL policy matches the
expert's counts (10 at 4k, 16 at 8k, 31 at 16k, 59 at 32k). The second-seed and stronger-cost RL
runs are in progress.

## No-cost ablation (13:05 UTC)

`rl-mixed-nocost-001` repeats the RL run with the cost term removed (reward = correctness only),
same seed and task sequence. On the 72-row OOLONG sample it scores 40/72 versus 41/72 for the
cost-aware run and 36/72 for the warm start, with the same recovery at 32k tokens (6 of 12, no
budget errors) and the same cost (13k versus 14k tokens). On the event-log dev split it scores
85/96 aggregation and 24/24 lookups while keeping the warm start's chunking and cost (18k tokens),
whereas the cost-aware run drifted to smaller chunks and about 10 percent more tokens (20k). On all 293
held-out rows the no-cost adapter scores 187/293 = 0.638 [0.58, 0.69] against 191/293 for the
cost-aware run (paired: 14 rows only the cost-aware run gets right, 10 only the no-cost run), with
the same 32k recovery (28 versus 27 of 50). So at 0.1 per 10k tokens the cost term neither produced
the OOLONG gain nor reduced cost; the gain comes from the correctness signal teaching budget-safe
chunking at long inputs. A second seed of the
cost-aware run and a 0.3 cost term are running to separate seed noise from the cost term's effect.
(The no-cost run's step-30 dev entry was lost to an adapter-name collision with the first run,
fixed afterwards; its step-30 adapter was saved and is the one evaluated here.)

## Out-of-range length on the event-log family (14:00 UTC)

Event-log tasks with 1024 records (twice the longest training length; 16 aggregation and 4 lookup
tasks; helper-call limits raised to 160 for every controller):

| Controller | Aggregation | Helper calls | Tokens | Lookups |
| --- | ---: | ---: | ---: | ---: |
| Scripted expert (12-line chunks) | 13/16 | 111 | 99k | 4/4 |
| SFT-005 warm start | 14/16 | 60 | 94k | 4/4 |
| SFT-005 + RL (0.1 cost) | 9/16 | 111 | 102k | 4/4 |
| Direct answering | 0/16 | 1 | 29k | 1/4 |

The warm start generalizes to the longer input with 24-line chunks; the cost-aware RL policy, which
had drifted to 12-line chunks on this family, loses five tasks there because more helper calls
accumulate more extraction errors. So the same RL run moved chunking in opposite directions on the
two families: toward larger chunks on OOLONG (where that fixed budget failures) and toward smaller
chunks on event-log (where it had no in-range benefit and an out-of-range cost). Whether this is
seed noise or a systematic effect is what the second seed and the 0.3 cost run will show.

## Second seed (15:55 UTC)

`rl-mixed-002` (same recipe, seed 1): OOLONG 72-row sample 42/72 (seed 1: 41, no-cost: 40, warm
start: 36), zero budget errors at 32k; event-log dev aggregation 87/96 (seed 1: 88, no-cost: 85,
warm start: 86), lookups 24/24. Both cost-aware seeds moved event-log chunking from 24 to 12 lines
(7/10/17/30/57 helper calls at 32 to 512 records versus 5/7/10/17/32 for the warm start), while
the no-cost run kept the warm start's chunking. So that drift is reproducible under the 0.1 cost term,
not seed noise, although its mechanism is unclear: the 12-line variant costs about 10 percent more
tokens, so the cost term alone should push the other way (and the 0.3-cost run below does not
show it). On all 293 held-out rows seed 2 scores 187/293 = 0.638 [0.58, 0.69] (40 paired gains, 19
losses against the warm start; 27/50 at 32k). The three RL runs so far (0.652, 0.638 with no
cost term, 0.638 second seed) therefore agree on a gain of 7 to 9 points over the warm start's
0.567, each driven by the 32k recovery and small gains at 1k to 2k. The 0.3-cost run is in
progress.

## Stronger cost term (18:30 UTC)

`rl-mixed-lam03-001` (lambda 0.3, seed 0, otherwise identical): training-sample tokens fell from
12.3k (steps 1 to 10) to 11.8k (steps 21 to 30) while training accuracy stayed at 0.83; the 0.1
runs showed no token decline. Held out: OOLONG 72-row sample 41/72 at 13k tokens (0.1 runs: 41
and 42 at 14k), 2 budget errors; event-log dev aggregation 85/96 at 18k tokens (0.1 runs: 88 and
87 at 20k), lookups 24/24. It kept the warm start's 24-line chunking (5/7/10/17/30 helper calls),
as did the no-cost run, whereas both 0.1 seeds moved to 12-line chunks. So the cost term buys
about 7 percent fewer tokens for about 1 to 3 points on the small event-log split and nothing on
OOLONG; the chunk-size drift is not monotone in lambda (present in two 0.1 runs, absent in one run
each at 0 and 0.3), and its mechanism remains unresolved. The RL accuracy gain over the warm start
is therefore the correctness signal's doing at every cost strength tried (0, 0.1, 0.3). On all 293
held-out rows the 0.3-cost run scores 185/293 = 0.631 [0.57, 0.68] at 13k tokens (38 paired gains,
19 losses against the warm start; 29/50 at 32k; 8 budget errors against 2 to 5 for the other RL
runs), so the four RL runs span 0.631 to 0.652 against the warm start's 0.567.

## Out-of-range length, all RL adapters (18:50 UTC)

The 1024-record check was repeated for the other RL adapters (same 16 aggregation and 4 lookup
tasks, helper-call limit 160):

| Controller | Aggregation | Helper calls | Tokens | Lookups |
| --- | ---: | ---: | ---: | ---: |
| Scripted expert (12-line chunks) | 13/16 | 111 | 99k | 4/4 |
| SFT-005 warm start (24-line chunks) | 14/16 | 60 | 94k | 4/4 |
| RL 0.1 cost, seed 0 (12-line) | 9/16 | 111 | 102k | 4/4 |
| RL 0.1 cost, seed 1 (12-line) | 12/16 | 111 | 104k | 4/4 |
| RL no cost (24-line) | 11/16 | 57 | 91k | 4/4 |
| RL 0.3 cost (24-line) | 11/16 | 57 | 91k | 4/4 |

This retracts the earlier reading that the chunk drift caused the out-of-range loss: the no-cost
and 0.3-cost adapters kept the warm start's chunking and still lost three tasks each, and the
second 0.1 seed with 12-line chunks lost only two. Every RL adapter is 2 to 5 tasks below the warm start at twice the
training length, on 16 tasks, which is consistent with a small loss of out-of-range robustness
from RL or with noise; it is not a chunk-size effect.

## Routing experiment, teaching stage (16:30 UTC)

To make "do not decompose" a taught and rewarded option, a direct route (peek, then one
delegated model call) was added to the expert and taught on half of the short inputs
(`teach-004`: event-log seeds 24-31; `teach-oolong-005`: OOLONG train windows). The teaching
data itself shows where each route pays (helper tokens, excluding controller overhead):

| Family, length | Direct route | Decomposition |
| --- | ---: | ---: |
| Event-log, 8 records | 22/23 (0.8k) | 17/17 (0.8k, plus one extra controller turn) |
| Event-log, 32 records | 6/14 (1.8k) | 26/26 (2.6k) |
| OOLONG, 1k tokens | 16/32 (3.2k) | 18/23 (0.6k) |
| OOLONG, 2k tokens | 16/24 (4.1k) | 23/26 (1.5k) |

So the correct policy is family- and length-dependent: direct only for the shortest event-log
inputs; decomposition everywhere on OOLONG, where the direct model reasons at length and is both
worse and costlier. SFT-006 trains on all unified teaching including these routes (only correct
trajectories, so the direct route survives mainly at 8 records), and RL runs with cost 0.1 and 0.3
then test whether the controller learns this routing (`pipeline10_gpu0.log`).

## Routing experiment, warm start and redesign (19:20 UTC)

SFT-006 (all unified teaching, 2,408 pairs, 355 updates) performs like SFT-005 (OOLONG sample
34/72; event-log dev 87/96, lookups 24/24) and, classified from its traces with
`route_analysis.py`, never takes the direct route at temperature 0 on either family. Sampled at
temperature 1 on 8- and 32-record tasks it decomposes 63 of 63 aggregation tasks, so its
probability of the taught direct route on short event-log inputs is effectively zero: the
teaching set holds about 100 decompose trajectories at 8 records from the earlier set against 22
direct ones. On OOLONG at 1k tokens, by contrast, the first RL rollouts from SFT-006 take the direct
route about one time in five, where it is the worse and costlier option.

The RL run from SFT-006 (`rl-route-lam01`, 0.1 cost) therefore cannot discover direct answering on
event-log inputs, whatever the reward says; it can only test whether RL suppresses a bad direct
route on short OOLONG inputs. It runs to completion as that control. To test the question that
matters, whether RL learns *when not to decompose* when both routes are available, a
route-balanced warm start was built (`balance_routes.py`, `teach-balanced-007`, 1,456 pairs):
equal numbers of direct and decompose trajectories at 8 and 32 records and at 1k and 2k OOLONG
tokens (22/22, 6/8, 16/16, 16/16), all longer trajectories unchanged. SFT-007 trains on it after the
control run, followed by RL at 0.1 and 0.3 cost (`pipeline11_gpu0.log`), each evaluated at
temperature 0 and sampled at temperature 1 on short inputs to read the learned route shares.

Control result (21:35 UTC). `rl-route-lam01` (30 updates from SFT-006, 0.1 cost; final dev 0.891)
sampled no direct route on any task from step 8 to step 30 (per-step route shares recorded in
`route-shares.jsonl`; event-log 8 records 94/96 decompose plus lookups, OOLONG 1k 135/181 decompose,
0 direct), so the one-in-five direct share seen on OOLONG 1k in steps 1 to 4 was driven to zero: RL
suppressed the worse route. Held out at temperature 0 it decomposes everywhere, scores 39/72 on the
OOLONG sample (SFT-006 34/72; the earlier RL runs 40 to 42) at 12k tokens with 4 budget errors, and
86/96 on event-log aggregation (SFT-006 87/96), lookups 24/24, keeping the warm start's 24-line
chunking. So the control reproduces the earlier RL gain and shows the expected suppression, and it
cannot speak to discovering a good direct route, as predicted.

Balanced warm start (22:10 UTC). SFT-007 (1,456 pairs, 2 epochs) matches the earlier warm starts
on accuracy (OOLONG sample 34/72 at 9k tokens; event-log aggregation 85/96, lookups 24/24) and now
holds both routes with usable probability:

| Input | Sampling | Direct route | Decomposition |
| --- | --- | ---: | ---: |
| Event-log, 8 records | temperature 1 (36 samples) | 13/15 correct, 2k tokens | 20/21 correct, 5k tokens |
| Event-log, 32 records | temperature 1 (35 samples) | 4/7 correct, 3k tokens | 27/28 correct, 7k tokens |
| Event-log, 8 records | temperature 0 (held-out dev) | 4/4, 2k | 15/15, 5k |
| Event-log, 32 records and longer | temperature 0 | never | 15/16 and as before |
| OOLONG, 1k tokens | temperature 0 (12 rows) | 4/6 | 1/5 |
| OOLONG, 2k tokens and longer | temperature 0 | never | as before |

So the reward landscape the RL runs face is: at 8 records the direct route is as accurate and costs
half the tokens (the right policy keeps it); at 32 records it loses accuracy (the right policy
drops it); on OOLONG the teaching data said direct is worse, though this small held-out sample
disagrees at 1k. RL from SFT-007 at 0.1 cost (`rl-route7-lam01`) is running, with 0.3 cost to
follow; per-step route shares are being recorded for both.

First balanced RL steps and a reward-scale problem (22:40 UTC). In the first five updates of
`rl-route7-lam01`, the direct route on OOLONG 2k (1/6 correct) and 1k (10/12 correct against
25/25 for decomposition, barely cheaper) was driven to zero within three steps, as the reward
prescribes. On 8-record event-log tasks, where the direct route was 5/5 correct at 2.7k tokens
against 11/15 at 5.4k for decomposition, its mean advantage was +0.005: the absolute cost term pays
0.1 per 10k tokens, so saving 2.7k tokens is worth 0.03 reward, two orders of magnitude below a
correctness difference, and 8-record tasks are one sixth of event-log draws. Under this reward the
policy can learn *not* to take a worse direct route but has almost no signal to *prefer* a cheaper
equally accurate one; the 0.3-cost run triples a signal that is still small. A scale-free variant
was therefore added to the trainer (`--cost-mode relative`: penalty lambda times the rollout's
tokens over its task group's mean tokens, so two routes on the same short task differ by a
constant fraction of lambda, while groups on one route are unaffected) and queued from SFT-007 at
lambda 0.5 with two seeds (`pipeline12_gpu0.log`), after the absolute 0.1 and 0.3 runs.

Step-10 checkpoint of the absolute-0.1 balanced run (22:55 UTC). The direct route is extinct on
every input: 8-record event-log tasks 0 direct in 40 rollouts after step 5 (16/16 decompose plus
lookups in each bucket), OOLONG 1k 0 direct in 48. Dev accuracy 0.891 and dev tokens at 8 records
unchanged at 4.9k, i.e. the policy pays for decomposition where the direct route was free. So under
an absolute token cost the negative signal from the lengths where direct answering fails (32
records, OOLONG 2k) generalizes through the shared cell text and removes the route everywhere,
faster than the tiny positive signal at 8 records can defend it. The run completes as the control;
the queue was reordered so the two relative-cost arms run next and the absolute-0.3 arm last
(`pipeline13_gpu0.log`).

Absolute-0.1 balanced run, complete (00:50 UTC). Over 30 updates the direct route was sampled
only in steps 1 to 5 and never after (8 records: 0 of 112 rollouts from step 6; OOLONG 1k: 0 of
193). Held out, the adapter decomposes everywhere at temperature 0 and in all 32 sampled
8-record aggregation tasks at temperature 1 (the warm start took the direct route in 15 of 36), and otherwise shows the usual RL
gains: OOLONG sample 39/72 at 13k tokens (warm start 34/72, 2 budget errors against 11), event-log
aggregation 87/96 (warm start 85/96), lookups 24/24. So with a balanced prior and an absolute cost
term, RL converges on "always decompose" and gives up the free saving on short inputs.

Relative-cost run, first five steps (01:10 UTC). Under the group-relative cost (lambda 0.5) the
direct route first surged (step 2: all OOLONG 1k, 2k and 4k rollouts direct, correct 5/8, 2/8 and
1/2) and then collapsed: from step 4 on, 0 direct in 48 OOLONG and 24 event-log rollouts, including
8 records where at step 3 direct and decomposition were both 4/4. Dev at step 5: 0.87, 8-record
tokens 4.9k (decomposition). So a stronger, scale-free cost signal changes the dynamics (a surge
rather than a slow fade) but not, so far, the outcome: the route decision generalizes across
lengths through the shared program text, the lengths where direct answering fails dominate the
update, and once the direct probability is near zero there is no exploration to recover it. The
run continues to step 30 with the seed-1 replicate next. To separate "too little short-input
signal" from "cannot condition on length", a third relative-cost run with short inputs
over-weighted in the training draw (8 records x3, 32 x2, OOLONG 1k x3, 2k x2; `--train-ns`,
`--oolong-train-lens`) is queued before the absolute-0.3 arm (`pipeline14_gpu0.log`). By step 10
the relative-cost run had not recovered the route (0 direct in 20 8-record and 56 OOLONG-1k
rollouts from step 4). Because the collapse is an exploration failure, a per-token KL penalty to the
warm start (k3 estimator, coefficient 0.1, `--kl-coef`) was added to the trainer and queued after
pipeline 14 in two arms, with uniform and with short-weighted training draws, preceded by a one-step
smoke test of the new code path (`pipeline15_gpu0.log`).

Relative-cost run, complete (03:35 UTC). `rl-route7-rel05` (30 updates, final dev 0.891) never
sampled the direct route after step 3 (8 records: 0 of 102 rollouts; OOLONG 1k: 0 of 199). Held
out it decomposes everywhere at temperature 0 and in all 32 sampled 8-record aggregation tasks at
temperature 1, with the usual RL gains: OOLONG sample 42/72 at 13k tokens with no budget errors
(warm start 34/72; absolute-cost arm 39/72), event-log aggregation 86/96 (warm start 85/96),
lookups 24/24, and it kept the warm start's 24-line chunking. So a scale-free cost term with a
balanced prior still converges on "always decompose". The seed-1 replicate is running; the
short-weighted and KL-regularized arms follow.

Where the gradient on the direct route comes from (04:55 UTC, `advantage_mass.py`). Summing the
leave-one-out advantages that landed on direct-route rollouts (aggregation tasks; route inferred
from turn and call counts, which matches the trace classification where traces remain):

| Run | 8 records | OOLONG 1k | OOLONG 2k and longer | Total | Last step with a direct rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| absolute 0.1 | +0.02 (1 rollout) | -1.30 (12) | -0.69 (6) | -1.97 | 3 |
| relative 0.5, seed 0 | +0.57 (4) | -1.16 (18) | -0.81 (12) | -1.40 | 3 |
| relative 0.5, seed 1 | none sampled | +0.04 (12) | -0.17 (4) | -0.12 | 2 |

In the first two runs the negative mass from OOLONG inputs, where direct answering is worse, is
two to sixty times the positive mass at 8 records, which is the shared-cell-text account. The
seed-1 run is more telling: the route vanished after step 2 with essentially zero net advantage
on its rollouts. Two mechanisms can do that without a negative signal on the route itself: the
loss sums token log-probabilities, so a correct 300-token decomposition cell receives about ten
times the gradient of a correct 30-token direct cell at equal advantage, and 8-record tasks were
never drawn in that run's first two steps. A sequence-normalized loss (`--seq-norm`, per-sequence
mean as in GRPO) was added and queued with the relative cost, alone and with short weighting
(`pipeline16_gpu0.log`), after the KL arms, which started concurrently with pipeline 14 at
04:55 UTC so that one run's rollouts overlap the other's updates (each RL step had left the
training GPU idle about 60 percent of the time).

KL penalty at the usual strength is inert (05:40 UTC). `rl-route7-rel05-kl` (relative cost 0.5,
k3 KL to the warm start with coefficient 0.1, seed 0) reproduced the seed-0 relative run step for
step, including the collapse of the direct route by step 4, with a mean per-token KL of 1 to 5
times 10^-4. The reason is arithmetic: the route decision is a handful of tokens in cells of
hundreds, so a KL averaged over all tokens barely registers a change that moves the direct
probability from 0.4 to near zero at one position, and at coefficient 0.1 the restoring term is
smaller than the policy-gradient term even at that position. The run was stopped at step 5 and its
artifacts kept. The sequence-normalized arms were started in its slot, and a KL arm with
coefficient 5 (alone and with short weighting) was queued after them (`pipeline17_gpu0.log`).

Relative-cost run, seed 1, complete (06:00 UTC). `rl-route7-rel05-s1` reproduces seed 0: the
direct route was sampled only in steps 1 and 2 and never after; held out it decomposes everywhere
(0 of 32 sampled 8-record tasks at temperature 1), scores 43/72 on the OOLONG sample at 14k tokens
(seed 0: 42/72) and 87/96 on event-log aggregation (seed 0: 86/96), lookups 24/24. Unlike seed 0
it moved to 12-line chunks on the event-log family (7/10/17/30/57 helper calls), as the two
absolute-0.1 seeds of the earlier study did. So the extinction of the direct route under the
group-relative cost is a two-seed result, and the usual RL gain comes with it.

A sampling artifact, found and fixed (06:28 UTC). The trainer drew training tasks by walking the
length-by-question grid cyclically, eight event-log cells per step over 30 cells, so 8-record tasks
appeared only in bursts (steps 3, 4, 7, 8, 11, 15, ... in the uniform runs; steps 1 and 5 in the
short-weighted run, whose weighting only lengthened the bursts). In the steps between bursts
there was no positive signal for the direct route at all while the negative signal from OOLONG
inputs acted every step. That is the cleanest account yet of the seed-1 collapse with near-zero
net advantage on direct rollouts, and it confounds both remedy arms run so far: short weighting
(`rl-route7-rel05-short`, direct still 10 of 11 correct at 8 records in step 1, extinct by step
5) and sequence normalization (`rl-route7-rel05-seq`, extinct by step 4). Both were stopped at
step 5 with their artifacts kept, and the queued strong-KL and balance pipelines were cancelled.
The trainer now has `--sample-mode stratified` (independent weighted draw every step, shortest
length guaranteed in the first slot). Pipeline 20 reruns the remedies on this footing in two
concurrent lanes, all with the relative cost and sequence-normalized loss: plain, balance term,
strong KL; and short-weighted, short-weighted plus balance.

Interim, stratified lanes at steps 3 to 4 (06:57 UTC). With every step containing short tasks, the
dynamics invert: the direct route now spreads instead of dying. Lane A (plain): 8 records 12/12
direct and correct at step 2, 4/4 at step 3, while OOLONG 1k is already 10/16 and 4/8 direct and
2k 3/4. Lane B (short-weighted): by step 4, 8 records 16/16 direct, but also 32 records 1/8 direct
(wrong), OOLONG 1k 9/16 and 2k 4/12 direct. So the policy is not learning a length-conditional
rule; it is flipping the route globally toward whichever mode has the edge in the current draw, the
mirror image of the earlier collapse. This is the "all-one-mode states are stable, mixed splits are
not" dynamic of Kassenaar et al. in both directions, and it argues that the missing piece is
conditioning on length, not reward shaping: the consistent-teacher SFT-008 test and the
library-with-bandit design (route chosen per input bucket from verified statistics) address that
directly. To be confirmed at steps 10 and 30 with held-out numbers.

Consistent teacher: the rule is representable (07:15 UTC). SFT-008 was trained on the same
trajectories as SFT-007 except that at 8 records only direct trajectories were kept and at every
other length only decomposition (1,757 pairs, 2 epochs). Held out, it routes by length exactly as
taught: at 8 records direct in 19 of 20 greedy runs and 38 of 40 samples at temperature 1 (3k
tokens against 5k for decomposition), at 32 records decomposition in 16 of 16 greedy runs and 31 of
32 samples, decomposition everywhere longer, OOLONG unchanged at 34/72. Accuracy at 8 records is
14/16 against 16/16 for the always-decompose controllers, a small price for the direct route's
errors on this family. So a 4B controller can represent and learn length-conditional routing from
a consistent teacher; what RL could not do is find or keep it. The decisive follow-up, RL from
this correctly routing warm start under the relative cost with stratified sampling, is queued next
in lane A ahead of the balance and KL arms.

Stratified lanes at step 5 (07:25 UTC): one mode each, in opposite directions. Lane A (plain)
settled back to always-decompose (8 records 12/12 decompose at step 5, dev 0.891). Lane B
(short-weighted) went all-direct everywhere (8 records 13/16, 32 records 8/12, 64 records 2/4,
OOLONG 1k 4/8 and 4k 5/8 direct) and its dev accuracy fell from 0.848 to 0.391, with 0 of 5 at
256 and 512 records. Same warm start, same reward, same sampling; only the share of short tasks in
the draw differed, and that decided which single mode the policy fell into. The policy does not
condition the route on length; it flips globally toward the mode that wins the current mix. Lane B
was stopped at step 6 with its artifacts kept, and its slot was given to the decisive arm, RL from
the correctly routing warm start SFT-008 (`rl-s8-strat`), with the short-weighted-plus-balance
run after it; the plain lane continues to step 30 and is followed by the balance and strong-KL
arms from SFT-007.

Library experiment L1, prompt-side version, negative (07:50 UTC). The two routine families
(`skills/eventlog.py`, `skills/oolong.py`: parsing, chunked helper extraction or classification,
exact aggregation, lookup, as functions with a usage note) were offered through the harness policy
text as one `exec` line plus the function signatures. The untrained 4B used them in 2 of 8 smoke
tasks; the taught SFT-005 controller fell from 86/96 to 17/96 on the event-log dev split
(lookups 24/24 to 4/24, 17 budget or no-final errors): it called the library names in its first
cell before defining them, then tried `from skills import eventlog` and other imports that do not
exist, mixing its trained routine with the new functions. Two lessons. A prompt-side library is
out of distribution for a controller taught a fixed routine and too fiddly for an untrained one;
the library has to be *prebound* by the kernel, present in the namespace from turn 1 and listed in
the bootstrap like the ABI helpers, which is a kernel feature (a harness prelude recorded in the
harness digest), and the controller then has to be taught to use it (L4). The remaining L1
evaluations were stopped; the kernel prelude is the next implementation step.

RL from the correctly routing warm start, first three steps (08:10 UTC). `rl-s8-strat` starts
from SFT-008, which routes by length (direct at 8 records, decomposition elsewhere), under the
relative cost, sequence-normalized loss and stratified sampling. Step 1 looked like the warm
start. By step 2 the direct route had spread to 64 records (1 of 12 correct) and OOLONG 1k (8 of
15); by step 3 it was the route on every length sampled: 32 records 4/4, 64 records 5/16, OOLONG
1k 5/8, 2k 2/3, 4k 0 of 8 with 8 budget failures. So on-policy RL did not keep a correct
length-conditional policy it was handed; within two updates it generalized the cheaper cell to
lengths where it fails outright. At the step-5 checkpoint its dev accuracy had fallen from 0.87 to 0.304, the all-direct
state; the run was stopped there with its step-5 adapter kept and evaluated. Together with
E07-7 (SFT can learn the rule) and E07-8 (the stratified lanes flip to opposite single modes),
this closes the question for this setup:
the rule is representable, but policy-gradient training with outcome-minus-cost rewards moves the
route globally, not conditionally. Routing has to be decided outside the weights (library
statistics, a probe) or trained with a mechanism that ties the decision to the input signature.

Library through the kernel prelude, validated (08:20 UTC). The kernel now supports a harness
prelude (`PreludeSpec`: source executed once before the first cell, names listed in the prompt,
hidden from `SHOW_VARS`, restored before every cell, digest in the harness fingerprint; tests in
`tests/test_prelude.py`). With the event-log skills prebound this way, a scripted controller that
only calls them (`expert_cells_library/`: `split_log`, `extract_events`, `answer_eventlog`,
`lookup_day`, two to three one-line cells per task) scores 87/96 on aggregation and 24/24 on
lookups on the event-log dev split with no errors, against 88/96 and 24/24 for the original expert
whose cells carry the full programs. So the library reproduces the routine mechanically, and the
controller's program shrinks from about 60 lines to three calls. The taught SFT-005 controller,
given the same prelude, calls `split_log` and `extract_events` correctly but then writes its own
aggregation instead of calling `answer_eventlog` (smoke: 1 of 8 correct), as a controller trained
on the full routine would; the untrained base is evaluated next, and a controller taught on
library-calling trajectories (SFT-009) is being built from the oracle's native replay.

Untrained controller plus prebound library (08:50 UTC). The untrained Qwen3-4B controller, with
the event-log skills prebound through the prelude and listed in its prompt, uses the library on
every task and scores 87/96 on aggregation on the event-log dev split (16/16 at 8, 32 and 128
records; 13/16 at 64, 256 and 512), against 46/192 for the same model without the library in the
baseline sweep (0 of 32 at 128 records and above), and equal to the scripted expert and to the
taught SFT-005 controller. Its cells are three lines, so total tokens at 8 records are about 1k
against 5k for the taught decomposition. Lookups were 0/24 because the controller always chose
`extract_events` plus `answer_eventlog`, and `answer_eventlog` returned "unknown" for lookup
questions; the library was repaired so `answer_eventlog` answers lookups from the extracted
events (earliest matching purchase), and the evaluation is being rerun. This is the first
"small model plus library" number: with the routine in the library, no training of the controller
was needed for the decomposition itself.

Untrained controller plus repaired library (09:25 UTC). With `answer_eventlog` handling lookup
questions, the untrained Qwen3-4B controller scores 85/96 on aggregation and 23/24 on lookups on
the event-log dev split with one error, against 86/96 and 24/24 for the taught SFT-005 controller
and 88/96 and 24/24 for the scripted expert: parity with no training of the controller at all,
and about 1k tokens per 8-record task against 5k. The one cost it does not yet optimize is inside
the library: it answers lookups by extracting every event (17k tokens) rather than through the
cheaper `lookup_day` path the expert used (2k), which is exactly the kind of choice a per-bucket
outcome statistic should make. Taken with the routing results, the picture for a small controller
is: put the routines in the library, let the model call them, and decide between alternatives
from verified statistics rather than from the weights.

Teaching the call pattern (SFT-009, 09:56 UTC). A controller fine-tuned on 651 library-calling
pairs (native replay of the scripted library caller on the teaching split, through the prelude
harness) scores 85/96 on aggregation and 24/24 on lookups on the event-log dev split, the same
accuracy as the untrained controller with the same library (85/96, 23/24), and it takes the taught
cheap lookup route every time (2k tokens on lookups against 17k untrained). It imitates the three
one-line cells of its teacher, so an 8-record task costs 4k tokens against 1k for the untrained
controller, which writes the same calls in one cell. Teaching therefore buys the route choice, not
the capability; the next entry gets the route choice without any training.

Untrained controller plus the OOLONG library (10:05 UTC). On the 72-row OOLONG sample (six
context lengths, 1k to 32k tokens), the untrained Qwen3-4B controller with the OOLONG skills
prebound (`parse_oolong`, `classify_rows`, `answer_oolong`) scores 43/72 with no errors at 10k mean
tokens, against 36/72 (10 errors) for the taught SFT-005 controller and 41/72 for the best RL
controller (14k tokens) on the same rows; it called the library on every task. A first attempt of
this evaluation failed on every row because the skill registry recognised the task family from an
attribute only the teaching script's task objects carry, so the event-log library was prebound
instead; the registry now reads the family from the evaluator's task fields (failed run preserved
as `oolong-lib-base/FAILED.txt`). Two families now show the same picture: the capability lives in
the library, and the untrained controller is as good a caller as the taught or RL-trained ones.

Route choice from statistics, no training (L3a-lite, 10:40 UTC). `skills/stats.py` builds a
table of routine outcomes keyed by an input signature computed from the request text alone
(family, power-of-two bucket of the line count, question kind parsed from the question), from
logged runs on the teaching and train splits only: the scripted library caller's replay and the
baseline sweep (direct answer and own-program routes through the controller). The skills harness
renders the retrieved history as a short note in the policy text, with the routine of highest
accuracy minus 0.05 per 10k tokens recommended. With this note the untrained Qwen3-4B controller
follows the recommendation on 120 of 120 dev tasks and scores 87/96 on aggregation and 24/24 on
lookups with no errors, at 14.3k mean tokens per task (lookups 1.6k), against 85/96, 23/24 and
16.8k without the note, 86/96, 24/24 and 15.8k for the taught SFT-005 controller, and 88/96,
24/24 and 17.0k for the scripted expert. So the route choice that every RL form lost (E07-3 to
E07-9) is recovered by a lookup table over verified outcomes, and nothing is trained. The first
version of the table keyed on the record count instead of the visible line count and on raw
instead of bucketed lengths, and scored a routine that never succeeded above one that did at
long lengths with a 0.2 penalty; both were caught on the teaching-split check before the
evaluation ran, and the run was restarted (two aborted starts, no rows kept).

Growth, before (E08, 10:40 UTC). Three question types the shipped library does not cover were
added to the generator (`top_buyer`, `purchase_span`, `busiest_day` with a tie rule; old task ids
unchanged). With the shipped library the untrained controller scores 0/72 on their dev tasks: it
calls `answer_eventlog`, receives "ANSWER: unknown", and submits it, every time, at 17k tokens,
because the library's docstring claims coverage of every question type. The direct answer scores
14/24, 7/24 and 8/24 (about 6 of 8 at 8 to 64 lines, about 0 above), so a controller that trusts
an overclaiming library is worse than no library on anything the library does not cover. Two
remedies run next: an honest library whose aggregator returns None for uncovered questions with
guidance to look at the question before aggregating (interim: the controller then writes its own
correct aggregation for purchase span and busiest day, but formats every top-buyer answer as
prose and fails the exact match), and the librarian, which proposes a routine, verifies it on
cached model extractions of the teaching split, and registers it with its verification record in
the statistics.

Honest library (E08-1, 10:55 UTC). With the aggregator returning None for uncovered questions
and the usage note telling the controller to print the question and a sample of events and
aggregate in the next cell rather than guess in the first (a first version let it guess in the
same cell, before it had seen the question, and it wrote a generic fallback for the five covered
forms: 0/3, restarted), the untrained controller scores 34/72 on the three held-out types: busiest
day 17/24, purchase span 16/24, top buyer 1/24. The top-buyer failures are format, not logic: the
gold name leads the submitted answer in 20 of 24 cases ("ANSWER: Greta bought the most units of
blender ..."), which the exact-match check rejects; the 512-line tasks hit the helper-call budget
(8 errors). So an honest library turns 0/72 into 34/72 with no new routine, the controller can
write the aggregation it needs once it has looked at the question, and the remaining losses are
exactly what a verified routine removes: exact output format and budget-safe chunking. The
librarian's result follows.

Growth, after (E08-2, 12:00 UTC). The librarian (`librarian.py`) asked the 4B for eight
candidate routines per held-out form, given the event schema, the shipped aggregator as a style
reference and three teaching examples with gold answers, and verified each candidate on the cached
model extractions of the 48 teaching tasks of that form. Top buyer: all eight candidates verified
47/48; purchase span: all eight 48/48; both registered under `skills/learned/` with their
verification rows entering the statistics table as a routine. Busiest day (tie rule): all eight
candidates verified 0/48, and two repair rounds that showed the proposer its best candidate with
three failing cases (question, expected, got, events) produced sixteen more candidates, five
distinct programs in all, every one with the same bug: the day totals are sorted ascending and the
first element returned, so the routine answers the day with the fewest units. A one-token fix
(descending sort) verifies 48/48 on both extracted and gold events, so the harness is sound and
the rejection is the proposer's. With the two accepted routines and the rebuilt note, the
untrained controller scores 24/24 on top buyer and 24/24 on purchase span on dev (0/24 and 0/24
before growth; 1/24 and 16/24 with the honest library), calling the learned routine every time,
with no errors at 17.6k mean tokens. The first after-growth run was invalid and is preserved
(`e08-dev-heldout-after/FAILED.txt`): the registry read the learned skill's family header from
the first five lines and the librarian wrote it on the sixth, so nothing was bound, and the
signature keyed the new forms as "other"; the librarian now registers each form's question
pattern alongside the routine, and the signature extractor consults them. Lesson for the deck: on
a form a small proposer can write, verified growth takes the controller from zero to perfect with
no training; on a form it cannot, verification keeps the wrong routine out and the failure is
visible in the candidates file. A stronger proposer (Qwen3-8B, 4-bit) is being tried on the
rejected form.

A stronger proposer recovers the rejected form (E08-3, 12:10 UTC). Qwen3-8B in 4-bit AWQ
(Hugging Face `Qwen/Qwen3-8B-AWQ`, revision 4da05a8e, Apache-2.0; provenance file beside the
weights) was served on the spare memory of GPU 0 with thinking enabled and a 6,000-token output
budget. Given the same librarian prompt for busiest day, all eight of its candidates verified 48/48
on the teaching extractions after 3.8k to 5.2k tokens of reasoning each, where all 24 of the 4B's
candidates had verified 0/48. The first candidate was registered, the statistics rebuilt, and the
untrained 4B controller then scores 24/24 on busiest day on dev with no errors (0/24 with the
shipped library, 17/24 writing its own aggregation with the honest library). So the loop's
ceiling is the proposer, not the caller: a 4B caller with an 8B-written, verified routine is
exact, and the verification step is what makes mixing proposer capacities safe. The three learned
routines are committed under `skills/learned/` with the librarian's verification records.

OOLONG, 293-row held-out set (E09, 12:20 UTC). The untrained Qwen3-4B controller with the OOLONG
library prebound scores 197/293 = 0.672 on the full held-out set with no errors at 11.0k mean
tokens: equal to the scripted expert (0.672), above the best RL controller (0.652 at 14.3k tokens,
5 errors) and the taught SFT-005 controller (0.567, 31 errors), and far above the direct answer
(0.208). By length: 26/43 at 1k, 34/50 at 2k, 37/50 at 4k, 37/50 at 8k, 34/50 at 16k, 29/50 at
32k; the 32k rows, where SFT-005 ran out of helper budget, are where the library's chunking pays.
By task type the weak spot is numeric-one-class (24/71), a library limitation shared with the
expert. So on the second family and the paper-grade split, the untrained caller with the
library matches the programmed expert that the whole RL campaign was trying to approach.

Online selection, design (E10, 13:35 UTC). `online_bandit.py` streams 240 tasks from a fresh
seed namespace (all nine question forms, six lengths, shuffled) through the untrained controller;
for each task a policy picks a routine from the applicable set, the harness note names that
routine as the one to use, the answer is verified, and the table is updated with the chosen arm's
outcome. Four arms run in parallel: Thompson sampling, greedy and uniform random from an empty
table, and greedy warm-started from the teaching table. Two false starts are preserved
(`e10-bandit-*-partial1`, `-partial2`): the first was contaminated when the third learned routine
was registered mid-stream (the arm set and the busiest-day signature changed under the running
processes); the second let every routine compete for every signature with an optimistic prior,
so 240 tasks were spent forcing inapplicable routines (lookup on a count question) and the random
arm scored best (0.41). The fix is a library contract, not a bandit trick: each routine declares
the forms it applies to (the shipped aggregator five, the cheap lookup one, each learned routine
its registered form), and the policy chooses among applicable routines and the direct answer,
which is exactly the direct-versus-routine question by length; statistics are pooled across
lengths within a form as a prior worth four observations. The streams are running with this
design.

Caller capacity, first rung (S1, 2026-10-21 11:30 UTC). The controller was served separately
(non-thinking Qwen3) while the helpers stayed the 4B. The untrained Qwen3-1.7B cannot follow the
kernel protocol without help: on event-log dev it scores 0/120, echoing the bootstrap envelope as
its cell until the turn limit. With the prebound library and the statistics note it scores 77/120.
Every task on which it wrote the recommended library call was correct (77 of 77); the 43 failures
are protocol loops (35 out of turns) and all 24 lookups, where it never wrote the cheap lookup cell.
On the four grown held-out forms it scores 28/96, again correct whenever it calls the learned
routine (19 of 19), and on OOLONG 1/72. A LoRA fine-tune of the 1.7B on the 651 event-log
library-calling pairs (S1b) changes the profile rather than the level: 91/96 on the held-out
forms and 19/24 on lookups, but 55/120 on dev (both-items 0/24) because it imitates the teacher's
inspect-then-act cells and loses track across turns (54 out of turns), and 12/72 on OOLONG, a
family it never saw. The first S1b evaluation was invalid (every row a connection failure: the
chain read an old startup line before the restarted server with the adapter was up) and is
preserved with a note; readiness is now checked through the model list.

Routine variants on OOLONG (E12, 12:05 UTC). The library's weak OOLONG type is numeric-one-class
(24/71 on the held-out set), off by small counts, so classification noise rather than aggregation
logic. Six classification variants were scored end to end on 96 OOLONG teaching rows, weighted to
that type: shipped 12-row chunks 0.552, 6-row 0.469, 24-row 0.531, task description added 0.562,
three-chunking vote 0.521, description plus vote 0.583 (paired against the shipped routine: 10
wins, 7 losses, not distinguishable). Numeric-one-class stays at 21 to 27 of 60 under every
variant. So this error is a helper-quality ceiling that no routine choice removes; the next arm
(E12b) keeps the routine and swaps in the 8B as helper to measure that ceiling directly.

Helper size does not lift the OOLONG count ceiling (E12b, 12:45 UTC). With the routine fixed and
the helper swapped for the 4-bit Qwen3-8B (non-thinking), the shipped routine scores 0.521 on the
same 96 teaching rows (4B helper 0.552) and numeric-one-class 22/60 (4B 23/60); the
description-prompt variant drops to 0.406. On the 293-row held-out run the numeric answers miss by
a median of about 10 percent of the true count already at 1k tokens (about ten rows), and exact
accuracy falls from about half at 1k to 1 of 11 and 2 of 12 at 16k and 32k. About one row in ten is
labelled differently from the gold label by both helpers, and an exact count over many rows needs
all of them right, so the ceiling is structural for exact-match counts. Label ambiguity in the
source datasets (TREC coarse categories, SMS spam) would explain why a larger helper does not help,
but that is not verified here. Consequence for the plan: this error is not removable by routine
choice, routine growth, or a modestly larger helper; it is a measurement ceiling to report, not a
target for the library.

Train/serve template mismatch voids the first small-caller fine-tunes (13:10 UTC). The 0.6B and
1.7B are hybrid-thinking models; served with thinking disabled, vLLM's chat template appends an
empty think block after the assistant header, but `sft_train.py` formatted training prompts
without it (the 4B Instruct model has no thinking mode, so this never mattered before). A one-cell
distillation adapter trained this way wrote the correct OOLONG cell but dropped the opening code
fence on 63 of 63 tasks, which the strict protocol rejects. The S1b numbers above (fine-tuned 1.7B:
dev 55/120, held-out forms 91/96, OOLONG 12/72) are therefore confounded and kept only as a failed
attempt (`NOTE.txt` beside each run). `SR_NO_THINK=1` now formats training prompts with the
serving template; the 1.7B library and one-cell adapters and a 0.6B one-cell adapter are being
retrained and re-evaluated (S1e).

Learned retrieval key (E13, 2026-10-21 14:30 UTC). Each task input was embedded with the 4B
controller's own last hidden layer (mean over the first 768 and last 256 tokens) and the route
(direct answer or library) chosen by its 8 nearest neighbours among 540 teaching-split tasks of
both families with logged outcomes under both routes, by mean accuracy minus a token penalty.
Evaluated offline on 413 held-out tasks (120 event-log dev, 293 OOLONG held-out) whose outcomes
under both routes are logged. With the default small penalty (0.05 per 10k tokens) the library
dominates and every policy, learned or hand-built, reduces to it: always-library 0.738, hand
signature 0.736, learned key 0.738, oracle 0.768. Under cost pressure the choice matters: at 0.3
the learned key (hidden state plus log length) scores 0.705 at 11.2k tokens against 0.695 at 11.0k
for the hand signature and 0.734 at 10.7k for the oracle (event-log 0.783, 0.750, 0.792); the
hidden state alone misses length (0.702 at 11.7k); at 1.0 the two are level (0.639, 0.644). So the
controller's own representation, with length appended, is as good a retrieval key as hand-chosen
features without any feature engineering, and slightly better at moderate cost pressure; neither
closes the gap to the oracle, which is the direct route on short tasks where both routes are right.

Caller-capacity ladder, complete (S1, S1e; 2026-10-21 14:40 UTC). Controllers served separately
(non-thinking Qwen3), helpers fixed at the 4B, the prebound library plus the statistics note (v4)
throughout. Correct answers:

| Caller | event-log dev, no library | dev, library + note | four grown forms | OOLONG-72 |
| --- | --- | --- | --- | --- |
| 0.6B untrained | 35/85 (stopped early) | 11/120 | 18/96 | 1/72 |
| 0.6B one-cell fine-tune | | 110/120 | 94/96 | 43/72 |
| 1.7B untrained | 0/120 | 77/120 | 28/96 | 1/72 |
| 1.7B three-cell fine-tune | | 109/120 | 70/96 | 3/72 |
| 1.7B one-cell fine-tune | | 109/120 | 93/96 | 42/72 |
| 4B untrained | 46/192 (train split) | 111/120 | 94/96 | 43/72 |
| 8B (4-bit) untrained | 48/120 | 112/120 | 93/96 | 43/72 |

Once the routines live in the library and the route choice in the table, the caller's size stops
mattering from 4B to 8B, and a 0.6B caller fine-tuned on 436 of the untrained 4B's own correct
one-cell library calls (teaching splits of both families; LoRA r32, three epochs, about six
minutes) matches them on every column. What the small model lacks untrained is protocol
discipline, not capability: the untrained 1.7B is right on every task where it writes the
recommended call, and the untrained 0.6B does better without the library (35/85) than with it
(11/120), its longer prompt derailing it. The teacher's form matters: the same 1.7B fine-tuned on
the scripted three-cell teacher (inspect, extract, answer) reaches 109/120 on dev but only 70/96
on the grown forms and 3/72 on OOLONG, a family absent from that teacher's data; the one-cell
teacher covered both families. The OOLONG 293-row runs and a second 0.6B seed are running to
confirm the 72-row sample.

Confirmed on the full held-out set (S2, 14:46 UTC). On the 293-row held-out OOLONG set the
one-cell-distilled 0.6B and 1.7B callers each score 197/293 = 0.672 with no errors at about 10.5k
tokens, identical to the untrained 4B caller with the library (197/293) and the scripted expert
(0.672), and above the best RL-trained 4B controller (0.652). A 0.6B controller with a verified
library therefore matches everything the RL campaign produced at 4B.

Another model family and a second seed (S2, S3; 15:20 UTC). Phi-4-mini-instruct (3.8B, Microsoft,
not Qwen) as caller with the Qwen 4B helpers: untrained with the library and note it scores 69/120
on event-log dev, 94/96 on the grown forms and 0/72 on OOLONG (every OOLONG task a protocol loop);
LoRA-tuned on the same 436 one-cell traces (fused-projection targets, three epochs) it scores
105/120, 94/96 and 43/72, matching the Qwen callers. A second seed of the 0.6B one-cell adapter
scores 107/120, 93/96 and 42/72 (first seed 110/120, 94/96, 43/72). The recipe (prebound library,
outcome note, distil the strong caller's one-cell calls) is therefore not specific to the Qwen
family or to one training seed.

What the small caller learns (S4, interim): a 0.6B tuned only on the 220 event-log traces writes
well-formed cells that call the OOLONG library it has never been trained on, but composes them
wrongly (it submits the return value of `classify_rows`, which returns nothing, instead of passing
the rows to `answer_oolong`), so OOLONG is 0/42 so far. Protocol discipline transfers across
families; composing an unfamiliar library does not. A mixed subset of only 55 traces (19 of them
OOLONG) already gives 25/33 on OOLONG so far.

What the 0.6B learns, and how many traces it needs (S4, L3b; 15:50 UTC). Adapters of the 0.6B on
subsets of the 436 one-cell traces (same recipe):

| Training traces | event-log dev | grown forms | OOLONG-72 |
| --- | --- | --- | --- |
| 55 mixed (19 OOLONG) | 87/120 | 93/96 | 42/72 |
| 110 mixed | 91/120 | 94/96 | 43/72 |
| 220 mixed | 110/120 | 94/96 | 43/72 |
| 436 mixed (S1e) | 110/120 | 94/96 | 43/72 |
| 220 event-log only | 95/120 | 94/96 | 0/72 |

OOLONG saturates with 19 in-family traces and event-log dev at about 220; the grown forms, whose
routines are absent from every training trace, are solved by all of them. An adapter that never
saw OOLONG writes protocol-correct cells against the OOLONG library but composes them wrongly. With
OOLONG outcomes added to the statistics table (v5), the note now carries the OOLONG recipe; that
lifts the event-log-only 0.6B from 0/72 to 11/72 but no further: it writes `parse_oolong` then
`answer_oolong` and drops the `classify_rows` step the recipe shows, reproducing the two-step shape
of its event-log training rather than the recipe in front of it (it is right only on questions
that need no labels). Untrained 0.6B, 1.7B and Phi-4-mini stay at 0/72 with the note (protocol
loops), and the 4B is unchanged (42/72 against 43/72). Retrieval of a recipe does not replace a
handful of in-family traces for a 0.6B caller.

Why OOLONG counts miss (E15, 17:20 UTC). OOLONG-synth's labelled context
(`context_window_text_with_labels`, re-downloaded at the pinned revision) gives a gold label per
row, so the library's `classify_rows` (4B helper, 12-row chunks) was scored row by row on 2,537
teaching-window rows. Agreement with gold is 0.903 on SMS spam and 0.789 on TREC coarse, and the
disagreements are systematic, not noise: spam is called ham 77 times against 23 the other way, and
TREC abbreviation and entity questions are called "description and abstract concept" 144 times
(location called entity 41 times). An exact count over tens to hundreds of rows with a directional
per-row error of 10 to 20 percent is wrong almost surely, which is the numeric-one-class ceiling
of E12 and E12b, and explains why neither routine variants nor a larger helper moved it. The bias
is consistent enough to correct: a counting routine that inverts the confusion matrix estimated on
teaching labels is the obvious library-level repair, and verification can decide whether it pays.

Confusion-corrected counting does not help (E16, 17:45 UTC). On the 71 held-out numeric-one-class
rows, the library with the 4B helper scores 24/71 (reproducing the 293-row run exactly); inverting
the teaching-window confusion matrix on the subset's predicted counts also scores 24/71 (6 wins, 6
losses). The plain counts are already unbiased on average across questions (mean signed error
-0.12 on spam, -0.18 on TREC, since questions ask about different labels), so there is little bias
to remove; the correction trims spam's mean absolute error from 3.4 to 2.9 but doubles TREC's (11.3
to 21.4), the six-label confusion matrix being too ill-conditioned to invert. The count ceiling is
per-question variance from row-level label ambiguity, not a correctable bias.

Long contexts, 64k to 512k (E14, 17:25 UTC). OOLONG-synth rows at 64k, 128k, 256k and 512k tokens
(spam and TREC coarse, 10 per length and dataset, re-downloaded at the pinned revision) with the
library and note, helpers the 4B, budgets raised to 4,000 helper calls. The untrained 4B and the
distilled 0.6B and 1.7B callers score identically: 15/20, 16/20, 12/20 and 7/20 (50/80), at about
1.0 token per input token (65k to 515k mean) and 49 to 386 seconds per row; every caller wrote the
same library call, so the outcome is set by the helpers. The controller never sees the input, so
a 0.6B caller handles inputs eight times longer than the 4B's own 64k window, where direct
answering is impossible. Accuracy falls with length through the count questions
(numeric-one-class 2/17; relative frequency 34/43), the per-row label noise of E15 accumulating
over more rows. A first 1M-token row with the 0.6B caller was correct (1,457 helper calls, about
1.01M tokens, 6.8 minutes); the remaining 1M rows are running.

Training the helper on disjoint items does not help (E17, 18:05 UTC). OOLONG-synth builds every
window from one pool of SMS-spam and TREC items: all 863 spam and 1,135 TREC instances in the
teaching windows also occur in held-out or long-context windows, so the helper cannot be trained
on OOLONG teaching labels without label leakage. Instead the original datasets were downloaded
(ucirvine/sms_spam at cae486f9; TREC train_5500 and TREC_10 from the CogComp source, checksums in
the provenance file) and only items absent from every OOLONG window kept: 172 SMS items (160 ham,
12 spam) and 1,735 TREC questions, formatted as 320 classify-routine chunks for a LoRA on the 4B
helper. Held-out per-row agreement with gold: spam 0.907 base against 0.562 tuned (the tuned
helper calls spam ham 452 times, the cost of a 12-spam training pool), TREC 0.759 against 0.750.
On the 293 held-out OOLONG rows with the tuned helper: 143/293 (spam 68/143, TREC 75/150) against
197/293 (112/143, 85/150) with the base helper. The spam drop is a data-imbalance artefact; the TREC
result is the informative one: 1,735 disjoint items labelled with the same scheme give no gain,
which points to instance-level ambiguity of the TREC coarse categories as the ceiling, not to a
shortage of supervision. Training the helper is still the right place for training in this
system, but on these two label sets the gold labels themselves bound what it can reach.

The paper's OOLONG setting, and beyond (E14b, E14c, E19; 19:00 UTC). The RLM paper's Table 1 reports
OOLONG trec_coarse at 131K tokens on 50 tasks; OOLONG-synth's validation split has exactly 50
trec_coarse rows at that length, so the task set is very likely the same. With the distilled 0.6B
caller, the library, and 4B helpers, all 50 score 0.440 exact match and 0.453 with OOLONG's partial
credit for numeric answers (0.75 to the power of the absolute error; the paper does not state its
rule). For reference, the paper reports GPT-5 base 44.0, RLM(GPT-5) 56.0 to 58.0 depending on
recursion depth, and Qwen3-Coder-480B base 36.0 and RLM 26.0 to 48.0. The 50 spam rows at 131K score
0.760 (0.805 partial). At 1M tokens the 0.6B caller is correct on 5 of 8 rows (about 1,460 helper
calls and 6.8 minutes per row) and at 2M on 2 of 4. Scoring script: `oolong_score.py`.

Below 0.6B: SmolLM2-360M (S5, 19:35 UTC). SmolLM2-360M-Instruct (Llama architecture, 8k window,
Hugging Face revision a10cc151) as caller: untrained it never submits (every task out of turns);
LoRA-tuned on the same 436 one-cell traces it scores 109/120 on event-log dev and 42/72 on OOLONG,
level with the 0.6B and the 4B, but 48/96 on the four grown forms: perfect on busiest day and
purchase span, 0/24 on top buyer and same-day cross, where it writes the busiest-day routine instead
of the routine the statistics note recommends. At 360M the calling discipline is still learnable
from traces; reading the retrieved recommendation and copying the right routine name is where
capacity first runs out.

An explicit payload helps the 360M (S5b, 19:35 UTC). With the note rendered as one instruction
("Routine to use for this input: X. Write it as the first cell: ...", `--force-recommended`)
instead of the table of verified history, the tuned 360M scores 76/96 on the grown forms (48/96
with the table): top buyer 24/24 (from 0), busiest day and purchase span 24/24, same-day cross 4/24
(from 0), where it still writes the busiest-day routine on 20 of 24 tasks despite the instruction.
For very small callers the format of the retrieved payload matters: an imperative with the exact
cell is followed far more reliably than a ranked table, though a strongly trained pattern can still
override it.

External baseline: the RLM paper's released RLM-Qwen3-8B (E18, 22:45 UTC). The released
post-trained model (mit-oasys/rlm-qwen3-8b-v0.1) was run in its authors' scaffold
(alexzhang13/rlm at 854e688; settings from the scaffold's OOLONG environment: context in the REPL,
root prompt = dataset instruction plus question, 12 iterations, sub-call max_tokens 4096, local REPL,
depth 1, the same model for root and sub-calls), served by vLLM at its native 40,960-token window
with an fp8 KV cache, on the same 293 held-out OOLONG rows. It scores 85/293 = 0.290 exact (0.321
with OOLONG partial credit; 94 with a lenient bare-answer parse): better than direct answering
(0.208; 59 to 35 discordant rows, p = 0.017) and far below the untrained 4B with the library (0.672;
130 to 18, p below 1e-21), at about 460 seconds per row. 59 rows failed outright, 48 because the
scaffold's growing root conversation exceeded the model's window, and the rest by timeout; even if
every failed row had been correct the score would be 0.49, so the gap is not a window artefact.
Caveats: one run; the spam instruction is our adaptation of the scaffold's TREC wording; the
scaffold's local REPL is not thread-safe (the first attempt, with threads, is preserved as failed;
the reported run used 16 single-row processes); the model may have been trained on OOLONG-synth
windows, which would favour it. The released model cannot run our kernel's protocol (four of four
smoke tasks out of turns).

Long store logs (E20, 23:25 UTC). Event-log tasks at 1,024, 2,048 and 4,096 records (about 1,300 to
5,100 lines, 20k to 80k tokens), all nine question forms including the four grown ones, four dev
seeds: the untrained 4B with the library and the v5 note scores 98/108 (32/36, 34/36, 32/36 by
length) with no errors, at 90k to 363k tokens per task; the direct answer scores 3/108 (2/36, 1/36,
and 0/36 at 4,096 records, where every input exceeds the 4B's window). Every form is at 9 to 12 of
12, the grown routines included. With OOLONG (E14), the long-context result now holds in both
families: in the event-log family, whose labels are unambiguous, accuracy does not fall with
length at all.

Window sensitivity for the external baseline (E18b, partial at the allocation's end). The 48
rows on which RLM-Qwen3-8B overflowed its 40,960-token window were rerun with the window extended
to 131,072 tokens (YaRN rope scaling, factor 4 over 32,768). By the end of the allocation, 40 of the 48 had
finished: 13 correct, against 31 for the library on the same rows. Substituting these
into the full run raises the baseline only from 0.290 to 0.334 (98/293); the remaining 8 rows were
still running when the allocation ended, and could add at most 8 more (0.362). The window was not
what held the baseline back.

## 22 October 2026 (allocation 6599)

Housekeeping. The second 0.6B one-cell seed scores 197/293 = 0.672 on the held-out OOLONG set (identical to
seed 0). The external baseline's window rerun completed: 16 of the 48 overflow rows correct at a 131k
window, which puts the released RLM-Qwen3-8B at 101/293 = 0.345 (from 0.290); the library is 0.672.

MRCR as a new family (E21, 02:00 UTC). OpenAI MRCR (MIT; long multi-turn chats in which one writing ask
recurs 2 or 4 times and the last message asks for the i-th reply with a random string prepended; official
SequenceMatcher grader). The two source files hold different length bins, so rows were split within each
of the eight official bins (recovered by length rank): 30 teaching rows from the bins up to 32k tokens,
and disjoint held-out sets of 64 rows (8 per bin) for 2 and 4 needles. Results so far:

| MRCR held-out | 2 needles | 4 needles |
| --- | --- | --- |
| Direct answer, 4B (fails above its 64k window) | 0.23 | 0.15 |
| RLM, untrained 4B, no library | 0.18 | 0.11 |
| Hand-written routine, no model at all | 1.000 | 1.000 |

A 20-line deterministic routine (parse the final instruction, count exact matches of the ask among user
turns after the few-shot message, return the string plus the next assistant turn) scores 1.000 at every
length up to 1M tokens. MRCR is therefore solvable by a program; for an RLM the open question is whether
the system finds that program. The 4B as library proposer (bootstrap_library.py: contract, three teaching
examples with gold answers, candidates verified end to end in a separate process on all 30 teaching rows,
two repair rounds) produced no working library in 24 candidates: first-round code misparsed the
instruction or hard-coded an example's string, and repair rounds returned fragments that failed to load.
The 4-bit 8B proposer (thinking on) is running.

MuSiQue long haystacks (E22). MuSiQue answerable questions (CC BY 4.0) embedded among paragraphs of other
questions of the same split up to 8k, 32k and 128k tokens (teaching from MuSiQue train, held-out from dev);
answer F1. Baselines, 30 dev questions per size: direct 0.52, 0.25, and overflow at 128k; the untrained 4B
in the RLM without a library delegates the whole input and scores the same (0.48, 0.25, overflow). Library
bootstraps with the 8B and 4B proposers are running.

MRCR bootstrapped by the 8B (E21, 02:15 UTC). The 4-bit Qwen3-8B proposer (thinking on) produced an
accepted library on its fourteenth candidate (repair round 1, candidate 5; earlier best 0.214): 30/30 exact
on teaching. It was reviewed before use: 40 lines of pure string parsing (the "Prepend ... to the Nth ..."
instruction, user turns that begin "write a <topic>", the next assistant turn), no imports, no I/O, no
helper calls, nothing copied from the three shown examples. It matches asks by prefix where the hand-written
routine matches exactly. Held out:

| MRCR held-out, 64 rows per column, 8k to 1M tokens | 2 needles | 4 needles |
| --- | --- | --- |
| Direct answer, 4B | 0.23 | 0.15 |
| RLM, untrained 4B, no library | 0.18 | 0.11 |
| 8B-written library run alone (no model) | 1.000 | 1.000 |
| Distilled 0.6B caller (trained on event-log and OOLONG only) + 8B-written library | 1.000 | 1.000 |
| Untrained 4B caller + 8B-written library | running | running |

The whole chain now runs without hand-written code for a new family: a proposer writes the library from
three examples, verification on 30 teaching rows accepts it, and a 0.6B caller that never saw MRCR calls it
correctly at every length to 1M tokens. The library also covers 4 needles though it was taught only on 2.
Two caveats keep the claim honest: MRCR is solvable by a short program, so this measures whether the
system finds the program, not long-context reading; and proposer capacity is decisive (the 4B failed all
24 candidates; the 8B succeeded on the fourteenth, at about 180k proposer tokens). Bootstrapped libraries
are kept in `skills/bootstrapped/` and selected as registry variants, because a file in `skills/learned/`
would shadow a variant mid-run.

MuSiQue interim (E22, E23). The reference filter-then-read routine run alone on the 90 held-out questions:
0.434 F1 overall. Paper-style recursion without a library (depth 1, untrained 4B): 0.254 (depth 0: 0.245;
both overflow at 128k), so recursion alone does not help. A component ablation (BM25 only, filter only,
both, both trimmed by rank, gold paragraphs only, random paragraphs) is queued to locate the gain; the
reference routine trims its union by position, which may drop gold paragraphs at long lengths.

MuSiQue with the library, and recursion (E22, E23, E25; complete at 02:57 UTC). Answer F1, 30 held-out dev
questions per cell (helper tokens per question in brackets):

| Arm | 8k | 32k | 128k |
| --- | --- | --- | --- |
| Direct answer, 4B | 0.515 | 0.248 | overflow |
| RLM, no library, depth 0 | 0.485 | 0.251 | overflow |
| RLM, no library, depth 1 (paper-style recursion) | 0.514 | 0.250 | overflow |
| Reference routine alone (no controller) | 0.439 | 0.399 | 0.464 |
| Untrained 4B + reference library | 0.549 (13k) | 0.353 (39k) | 0.475 (133k) |
| Distilled 0.6B + reference library | 0.574 | 0.356 | 0.448 |
| 4B + recursive library (child RLM per hop) | 0.289 (41k) | 0.290 (140k) | 0.340 (513k) |

At 8k direct reading is as good as any routine; the library pays from 32k and is the only way to answer at
128k (0.45 to 0.48); at 512k the reference routine still scores 0.355 (20 questions). The distilled 0.6B
caller matches the untrained 4B on a family it never saw. Recursion does not help: paper-style depth 1 is
level with depth 0, and a library that hands each hop to a child RLM is worse at every length at three to
four times the tokens, in line with a reproduction that found deeper recursion overthinks on retrieval
(arXiv 2603.02615).

Retrieval depth (E22h, 10:52 UTC). BM25 top 60, two rounds and three rounds in one run, both samples (180),
fault-tolerant follow-up queries:

| Arm | 8k | 32k | 128k | all | recall | paragraphs | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 top 60 | 0.479 | 0.257 | 0.413 | 0.383 | 0.91 | 59 | 8.3k |
| Two rounds | 0.490 | 0.374 | 0.486 | 0.450 | 0.90 | 38 | 8.5k |
| Three rounds | 0.493 | 0.346 | 0.493 | 0.444 | 0.94 | 53 | 15.7k |

Two rounds over BM25: +0.067 (paired se 0.028), replicating E22f (+0.058). A third round adds nothing (-0.006,
se 0.025; +0.006 at 128k) at twice the helper tokens. The earlier tie at 128k (E22f) was sample noise: here two
rounds gain there too. Two rounds are enough.

Round-two distillation, held-out discovery (10:58 UTC). Two discovery seeds per arm, 74 family attempts:

| Proposer | all | string formats (42) | chat (16) | semantic (16) |
| --- | --- | --- | --- | --- |
| 4B Instruct, untrained | 9 | 1 | 8 | 0 |
| + SFT round one (42 pairs, no chat) | 16 | 13 | 3 | 0 |
| + SFT round two (68 pairs, 13 chat) | 20 | 13 | 7 | 0 |

Adding chat-format libraries repaired the chat loss without giving back the string-format gain. Round two's
chat families share their question types with the chat training families, so chat is no longer a clean
held-out test of a new format.

Round-two distillation on MRCR (10:52 UTC): with 68 pairs including 13 chat-format libraries, the tuned 4B reaches
0.214, 0.214 and 0.152 on the three MRCR seeds (round one: 0.000 to 0.034), the same partial solution the
non-thinking 8B and the thinking 4B found, and no accepted library.

Scale (E22g, 10:13 UTC). At 512k tokens (about 4,000 paragraphs), 30 held-out questions, both routines run alone
under kernel semantics on one helper server: two-round retrieval (v4b) 0.381 F1 at 8k helper tokens and 91 s
per question; filter+BM25 (v3) 0.326 at 498k tokens and 209 s (+0.056, paired se 0.086). v4b's cost does not
grow with the haystack (retrieval is BM25 on the CPU); v3's filter reads every paragraph, so its cost is linear.
At 512k the cheap routine is sixty times cheaper and at least as accurate. Through the distilled 0.6B caller
on the same 30 questions: 0.392 (scripted 0.381), no errors, no routine faults, 10k tokens and 18 seconds per
question.

The two-round library through a caller (E25e, 09:36 UTC). v4b (two-round retrieval, 800-token follow-up,
tolerant of helper faults) against v3 (filter + BM25, fixed reader), distilled 0.6B caller, one helper server
for both, both question samples (180):

| Library | 8k | 32k | 128k | all | routine faults | tokens per question |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 0.517 | 0.312 | 0.460 | 0.430 | 8 | 57k |
| v4b | 0.530 | 0.392 | 0.502 | 0.475 | 0 | 10k |

+0.045 (paired se 0.034) at a sixth of the tokens and with no routine faults. v4b becomes the MuSiQue library.
With v4b the untrained 4B caller scores 0.575/0.387/0.560 (0.507) on the first sample against 0.524 for the
distilled 0.6B (-0.017, se 0.028), with no faults in either; the untrained 4B with the first library had scored
0.459 with 14 routine faults. A library that does not fault makes caller size nearly irrelevant here, as on
OOLONG and MRCR.
Taken with E22c to E22f: on MuSiQue the useful capacity is in choosing evidence (a second, targeted
retrieval), not in a learned filter over every paragraph nor in a bigger reader, and the cheapest library that
does this is also the best.

Distilling library writing into the 4B (E26b, 09:21 UTC). A LoRA (rank 32, 3 epochs, 4.5 minutes on half an
A100 slice) on 42 verified (prompt, library) pairs that the thinking 8B wrote for training families (formats
pipe, kv, jsonl, csv, table; code only, the thinking dropped; repair-round successes paired with the round-0
prompt). Held-out discovery, the same 37 families and settings as the untrained 4B:

| Proposer | solved | of which string formats | chat | proposer tokens per family |
| --- | --- | --- | --- | --- |
| 4B Instruct, untrained | 5 | 0 | 5 of 9 | 4.0k |
| 4B Instruct + LoRA on the 8B's libraries | 9 | 7 | 2 of 9 | 2.6k |
| Qwen3-8B-AWQ, thinking (the teacher) | 20 | 12 | 8 of 9 | 49k |

Paired by family the tuned model gains 8 and loses 4 (exact McNemar p 0.39), so the net gain is not
established; the pattern is clearer: every gain is a string format (prose, XML-like, table), formats never
solved before and unlike the training formats only in surface, and every loss is the chat format, which no
training family used. On MRCR (a chat format) it fails all three seeds (best 0.034). Distillation transfers
parsing skill across string formats and erodes message-list handling, at a twentieth of the teacher's tokens.
Second discovery seed (09:49 UTC): untrained 4 of 37, tuned 7. Over both seeds (74 family attempts): untrained 9,
tuned 16 (p 0.19); on string formats 1 vs 13 of 42 (12 gained, none lost, exact McNemar p 0.0005); on chat 8 vs
3 of 16 (p 0.18). The string-format gain is established; the chat loss is the price of a training set without
chat.
Next: a second round with more pairs (the seed-variant training run) plus the model's own verified
successes as a guard against forgetting, and two discovery seeds per arm to tighten the comparison.

Helper ceiling for the semantic families (E26, 09:20 UTC). The thinking 8B solved 20 of 37 held-out generated
families, none of the 8 that need the helper (semantic_total: sum the units of items of a category). Those are
near misses, not proposer failures: the best libraries call the helper 40 to 60 times per task and score 0.77
(chat), 0.83 (csv), 0.67 (table), 0.47 (pipe). The 4B helper classifies the 75 lexicon items with accuracy
0.973 (a drill as an instrument; one verbose reply), so a task built on about 30 items is fully right with
probability near 0.44 if errors were spread evenly; an exact-match threshold of 0.9 is out of reach, as with
OOLONG's label ceiling. Helper-dependent families are scored against a threshold set from the helper's
accuracy (0.7 here: 2 of 8 for the 8B), and the benchmark reports them separately.

LongBench-v2 (E32, external validity, 09:15 UTC). 150 questions stratified by length class (Apache-2.0,
revision 2b48e494), four options, chance 0.25; direct answering reads the context cut to its first and last
24k tokens (exact, Qwen3 tokenizer); the generic library takes brief notes from every 24k-character part with the
4B helper and chooses over the notes.

| Arm | short | medium | long | all | tokens per question |
| --- | --- | --- | --- | --- | --- |
| Direct, truncated | 0.46 | 0.36 | 0.28 | 0.367 | 42k |
| Map-reduce library, run alone | 0.42 | 0.32 | 0.34 | 0.360 | 366k |
| RLM, untrained 4B, no library | 0.34 | 0.02 | 0.00 | 0.120 | 10k (83 overflow errors) |

The library ties direct reading (-0.007, paired se 0.051) at nine times the tokens; its only positive class
(long, +0.06, se 0.072) is within noise, and long questions sit near chance for every arm. By domain it helps
single-document QA and structured data (0.32 to 0.39, 0.33 to 0.42) and hurts multi-document QA (0.44 to
0.28), all on small cells. A boundary for the thesis: libraries pay where the task decomposes into local
judgments (aggregation, lookup, retrieval of a few supporting passages); LongBench-v2's whole-document
reasoning questions do not, and a 4B helper's notes lose what the choice needs. The paper-style RLM without a
library is far below both.

Scripted evaluation versus the kernel (09:10 UTC). With the 0.6B caller, the two-round library (v4) scored
0.328 against 0.507 for v3 on the same questions (-0.179, se 0.045), although the routine run alone matched or
beat its alternatives. The cause is a difference of semantics: in the RLM kernel `ask` raises a typed fault
when a reply is not "completed" (for example cut off at its token cap) or is empty, while the scripted
harnesses returned the partial text. v4's follow-up query had a 200-token cap, the 4B often explains before its
QUERY line, so in the kernel the call raised, the routine crashed, and the caller fell back to sending the whole
request to the helper. Traces with a routine fault: v1 10 of 90, v3 4, v4 46. This also explains why v3 beat v1
under the caller (fewer cut-off reads), and means the reader fallback of E22d never runs inside the kernel (the
cut-off read raises first). Fixes: scripted harnesses can now mirror the kernel (`run_routine.py
--kernel-semantics`; `KERNEL_SEMANTICS=1` for the bootstrap verifier, which also swallowed request errors),
and v4b allows 800 tokens for the follow-up and treats a fault as "no text". Rule: a routine is verified
under the semantics it will run under. The v4 caller runs are marked superseded; v4b and v3 are being rerun on
one helper server, both samples.

Two-round retrieval (E22f, 08:41 UTC). Putting capacity into selection: BM25 top 20 for the question, then
the 4B helper reads those and writes one query for the missing fact (usually the bridge entity), BM25 top 20
new paragraphs for that query, and the same 4B reader over the union. Both question samples (180), paired
against one-round BM25 top 60 in the same run:

| Arm | 8k | 32k | 128k | all | recall | paragraphs read | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 top 60 | 0.464 | 0.286 | 0.437 | 0.396 | 0.91 | 59 | 8.3k |
| Two-round retrieval | 0.543 | 0.378 | 0.438 | 0.453 | 0.87 | 31 | 7.5k |

+0.058 (paired se 0.033, n 180; +0.024 and +0.091 on the two samples) with fewer tokens and half the
paragraphs: a targeted second query beats a wider net, and it beats the filter+BM25 routine (0.423 pooled) at a
seventh of its cost. The gain is at 8k and 32k; at 128k the two tie, so the follow-up query does not yet find
the bridge paragraph in a thousand-paragraph haystack. Next: two-round retrieval as the MuSiQue library (v4)
with the callers, and a third round or query rewriting at 128k.

Reader capacity (E22e, 08:12 UTC). The 4-bit Qwen3-8B with thinking as the reader (6,000-token cap,
same fallback), on both question samples (180 questions), against the 4B reader on the same questions:

| Reader | BM25 top 60: 8k / 32k / 128k / all | gold paragraphs: all |
| --- | --- | --- |
| 4B | 0.483 / 0.311 / 0.422 / 0.405 | 0.625 |
| 8B, thinking | 0.566 / 0.310 / 0.382 / 0.419 | 0.666 |

Paired 8B minus 4B: +0.014 with BM25 retrieval (se 0.032), +0.041 with gold paragraphs (se 0.029). A bigger
reader does not close the 0.2 to 0.25 gap between gold paragraphs and retrieved ones; that gap is in
selecting the two to four supporting paragraphs among 60 candidates, which neither BM25 nor the 4B filter does
well. Capacity in a MuSiQue routine belongs in selection (multi-hop retrieval, iterative query reformulation),
not in reading.

MuSiQue bootstrap and the verification distribution (E22b, 07:13 UTC). The 8B proposer's MuSiQue bootstrap
ended below its threshold (best 0.429 of 0.6 over 24 candidates), but the cause is the design: all 30 teaching
tasks were 8k haystacks, where reading everything works, so verification ranked first a candidate that sends
the whole haystack to the helper, cut at 150,000 characters. On a mixed teaching set (10 tasks each at 8k, 32k,
128k) that candidate scores 0.30/0.45/0.00 (all 128k tasks overflow), against 0.45/0.75/0.65 for the reference
v3 routine (0.617 overall). Verification has to cover the lengths a library will face, or it selects
libraries that fail silently at scale. The bootstrap is being rerun on the mixed set; a library within 0.05 of
the reference (0.57) counts as accepted.

Routing by length on fresh questions (E31, 05:12 UTC; pre-registered at 04:50, see
[the pre-registration](assets-e31-preregistration.md)). Rule read off E22d: filter+BM25 at 8k and 128k, BM25
top 60 at 32k. Test: 90 new dev questions (3 shared with the earlier sample), the fixed reader, one server.

| Arm, fresh sample | 8k | 32k | 128k | tokens at 128k |
| --- | --- | --- | --- | --- |
| BM25 top 60 | 0.467 | 0.178 | 0.426 | 8k |
| Filter only | 0.224 | 0.328 | 0.309 | 124k |
| Filter + BM25 | 0.378 | 0.228 | 0.453 | 126k |
| Gold paragraphs | 0.626 | 0.484 | 0.687 | 1k |

Primary: routed minus filter+BM25 everywhere -0.017 (paired se 0.023, z -0.73, n 90), so the pre-registered
"retire" branch fired: the 32k gap seen on the first sample (+0.063 for BM25-60) reversed (-0.050). The
rankings move as much at the other lengths (at 128k the filter alone was best on the first sample and is worst
here). With 30 questions per cell, the differences between these routines are smaller than the variation
between question samples, so a statistics table built from cells of this size would choose by noise; selection
memory for close routines needs larger cells or pooled priors (as in the hierarchical bandit, E10). Routing
did save tokens (48k vs 57k per question) at no measurable loss.

Pooling both samples (180 questions, fixed reader) gives the comparisons that do replicate: filter+BM25 0.423,
BM25 top 60 0.405, filter alone 0.340, gold paragraphs 0.625. Filter+BM25 beats the filter alone by +0.083
(paired se 0.029) and does not measurably beat BM25 top 60 (+0.018, se 0.031), which costs about a seventh of
the tokens (8k vs 57k per question). The expensive helper filter earns little on top of BM25; the cheap
retrieve-then-read routine is the efficient MuSiQue library, and the remaining gap to the gold-paragraph
ceiling (about 0.2) is in retrieval and multi-hop reading, not in the reader's answer formatting.

Routine evolution (E24, complete 05:02 UTC). GEPA-style reflective search over the MuSiQue routine's two
prompts and four numbers (filter chunk size, BM25 depth, keep cap, reader token cap): an 8B reflector (thinking
on) saw the parent's failures on an 8-question teaching minibatch and proposed one configuration; a child that
beat its parent on the minibatch was scored on 24 teaching selection questions (32k and 128k). 20 iterations,
2.1 hours, 7 children accepted. The reflector rewrote the prompts in 16 of 20 proposals, and every accepted
child with rewritten prompts scored below the seed on the selection set (0.38 to 0.54 vs 0.556); the only
child to beat it changed numbers alone (chunk 15 to 6, BM25 depth 15 to 20, keep cap 60 to 40, reader cap
600 to 800): 0.636 on the selection set. Held out (90 questions, the three routines run alone on one server):

| Routine | 8k | 32k | 128k | all | tokens |
| --- | --- | --- | --- | --- | --- |
| Seed (reference v1) | 0.457 | 0.366 | 0.464 | 0.429 | 57k |
| Hand fix v3 (reader 1,500 + fallback) | 0.507 | 0.354 | 0.497 | 0.453 | 57k |
| Evolved (node 6) | 0.533 | 0.403 | 0.495 | 0.477 | 61k |

Evolved minus seed +0.048 (paired se 0.039), evolved minus v3 +0.024 (se 0.038), v3 minus seed +0.024 (se
0.025). The selection-set gain (+0.080) shrank by about 40 percent held out, and the search's useful finding
was the reader cap that the component ablation had already exposed. With a 4B helper, an 8B reflector's
prompt rewrites hurt; search over the routine's numbers helped. Comparison arms from the literature not yet
run: one offline pass over saved traces (CASD) and an RLM as the optimizer (RLMOpt).

Random-search null (E24c, 06:38 UTC): the same protocol and budget (20 iterations, minibatch acceptance,
24-question selection set) with children drawn uniformly over the four numbers and the prompts fixed at the
seed: 6 children accepted, none above the seed on the selection set (best 0.535 vs 0.556), 1.5 hours. The
reflective search found one configuration that beat the seed (0.636) in 20 iterations and random search none;
one search seed each, and the reflective winner's held-out gain was itself not significant, so this is weak
evidence that the reflector helps and none that it beats a hand fix guided by the component ablation.

Proposer ladder on MRCR (E21b, 04:48 UTC). Bootstrap seeds (8 candidates, two repair rounds, 24 at most;
seeds differ in the three shown examples and in sampling), accepted if the teaching mean reaches 0.9:

| Proposer | seeds accepted | best teaching mean per seed |
| --- | --- | --- |
| Qwen3-4B-Instruct-2507 (no thinking) | 0 of 1 | 0.000 |
| Qwen3-4B hybrid, thinking off | 0 of 1 | 0.000 |
| Qwen3-4B hybrid, thinking on | 0 of 2 | 0.214, 0.214 |
| Qwen3-8B-AWQ, thinking off | 0 of 2 (one confounded seed rerun) | 0.214, 0.154 |
| Qwen3-8B-AWQ, thinking on | 1 of 3 | 1.000, 0.000, 0.000 |

Discovering the MRCR program needs both the 8B and thinking, and even then succeeds on some seeds and not
others; a single success is a sample, not a capability. The generated-family discovery benchmark (37 held-out
families per proposer) measures the same ladder with more resolution and is running.

Documentation form (E25d, 04:42 UTC). The 8B-written MRCR library with four docstrings, untrained 4B caller,
64 held-out 2-needle rows each:

| Docstring | score | errors |
| --- | --- | --- |
| signature only | 0.184 | 35 |
| description + signature (as the 8B wrote it) | 0.592 | 23 |
| signature + usage line | 0.951 | 0 |
| description + signature + usage line | 0.953 | 1 |

The usage line alone carries the effect; a description helps somewhat when the usage line is missing and adds
nothing when it is present. With the reader fix (v3) as its library, the distilled 0.6B caller scores
0.611/0.433/0.475 on MuSiQue at 8k/32k/128k, +0.047 over v1 (paired se 0.020, n 90) at the same 53k tokens
per question, so the scripted gain carries through the caller.

Reader fix (E22d, 04:14 UTC). The same four arms run twice on the same server, the reader capped at 600 output
tokens as before, and at 1,500 with one short-answer follow-up when no ANSWER line appears. Answer F1, 90
held-out questions:

| Arm | 8k: 600 / fixed | 32k | 128k | paired gain (se) | reasoning fragments |
| --- | --- | --- | --- | --- | --- |
| BM25 top 60 | 0.495 / 0.498 | 0.362 / 0.443 | 0.353 / 0.417 | +0.050 (0.027) | 13 to 1 |
| Filter only | 0.383 / 0.415 | 0.224 / 0.231 | 0.473 / 0.534 | +0.033 (0.022) | 12 to 0 |
| Filter + BM25 | 0.469 / 0.563 | 0.369 / 0.380 | 0.457 / 0.535 | +0.061 (0.028) | 11 to 1 |
| Gold paragraphs | 0.680 / 0.668 | 0.634 / 0.627 | 0.672 / 0.660 | -0.010 (0.008) | 5 to 1 |

The fix removes nearly every truncated answer and gains about two standard errors on each retrieval arm; the
gold-paragraph arm, whose reasoning was short, does not move. With it the routine reaches 0.535 at 128k against
a reader ceiling of about 0.66. Saved as reference v3 (`skills/reference/musique_filter_read_v3.py`, registry
variant `musique_ref3`); the original stays unchanged because earlier results cite it.

Compute-matched null (E28, 03:42 UTC). Direct answering by the 4B sampled three times at temperature 0.7 and
combined by majority over normalized answers, on the same 30 held-out questions per length:

| MuSiQue | direct, T=0 | majority of 3, T=0.7 (tokens) | best of 3, chosen with gold (bound) | 4B + library (tokens) |
| --- | --- | --- | --- | --- |
| 8k | 0.515 | 0.499 (23k) | 0.603 | 0.549 (13k) |
| 32k | 0.248 | 0.217 (88k) | 0.258 | 0.353 (39k) |

At 32k the library beats majority-of-3 by +0.136 (paired se 0.084, n 30) at less than half the tokens, and
even choosing the best of three samples with the gold answer reaches only 0.258: the library's gain is not
bought by more compute spent reading. At 8k there is no difference.

Component ablation of filter-then-read (E22c, complete 03:36 UTC). Scripted, 90 held-out questions, the same
4B reader for every arm; answer F1, gold-support recall, paragraphs kept, helper tokens per question:

| Arm | 8k | 32k | 128k | recall at 128k | tokens at 128k |
| --- | --- | --- | --- | --- | --- |
| BM25 top 15 | 0.411 | 0.312 | 0.324 | 0.63 | 2k |
| BM25 top 60 | 0.502 | 0.341 | 0.365 | 0.84 | 8k |
| Helper filter only | 0.416 | 0.240 | 0.501 | 0.72 | 125k |
| Filter + BM25 top 15 (reference) | 0.543 | 0.346 | 0.413 | 0.89 | 126k |
| Gold supporting paragraphs only | 0.668 | 0.649 | 0.640 | 1.00 | 1k |
| 60 random paragraphs | 0.438 | 0.132 | 0.080 | 0.03 | 8k |

Paired over all 90: reference minus BM25-60 +0.031 (se 0.034), minus filter only +0.049 (se 0.038), minus
random +0.218 (se 0.050). Readings: (1) [withdrawn 05:15 UTC after E31: on a fresh question sample the filter alone is the
worst arm at 128k (0.309 vs 0.453 for the union), so the per-length ordering below was sample noise and does
not show a distractor-limited reader] at 128k the precise filter alone (12 paragraphs, recall 0.72) beat the
union (23 paragraphs, recall 0.89), 0.501 vs 0.413, on this sample; (2) BM25
top 60 is within noise of the full routine at a fifteenth of the tokens; (3) even with only the gold
paragraphs the reader reaches 0.65, and 12 to 19 percent of answers in every arm are fragments of
reasoning (the 600-token reader cap cut the reasoning before the ANSWER line, and the fallback took the last
line), scoring near zero: a fixable loss now tested in E22d (1,500-token cap plus a short-answer fallback,
paired against the 600 cap on the same server); (4) the position trim in the reference routine never fired
(the union never exceeded 60). Noise calibration: the two arms with identical inputs (reference and its
rank-trimmed twin) produced different outputs on 24 of 90 questions and different F1 on 6, from serving
nondeterminism at temperature 0; per-length cells of 30 can differ by 0.07 from identical inputs, so only
paired comparisons over all 90 are read.

Documentation decides whether an untrained caller uses a library (E25c, 02:45 UTC). The 8B-written MRCR
library omitted the contract's "Typical use: FINAL_TEXT(answer_mrcr(request))." line. Without it the untrained
4B often sent the whole input to `model_complete` (overflowing at 128k and above) or retyped the routine's
output by hand in a second cell. Swapping only that line, 4B caller, MRCR held-out (64 rows each):

| Library | without the usage line | with it |
| --- | --- | --- |
| 8B-written, 2 needles | 0.592 (23 errors) | 0.953 (1 error) |
| 8B-written, 4 needles | 0.576 (22 errors) | 0.875 (3 errors) |
| Hand-written reference, 2 needles | 0.215 (35 errors) | 0.952 (0 errors) |

The distilled 0.6B caller scores 1.000 without the line. The bootstrapper now inserts the line into accepted
libraries that omit it. The untrained 0.6B fails with or without the line (0.205 and 0.125; it overflows a sub-model
or never submits an answer), so it is distillation, not size, that makes the 0.6B caller immune. Separately, one thinking-off 8B bootstrap seed was confounded by a harness fault (an all-zero degenerate
16k-token first candidate stayed best and overflowed every repair prompt; fixed by tie-breaking toward
loadable short code and capping quoted code) and was rerun: thinking off failed on both clean seeds (best 0.214 and 0.154 in 24
candidates each), against success on the one thinking-on seed so far (two more running). GPU1 then had no
server for about 17 minutes (02:57 to 03:14 UTC): the re-split script started vLLM without the
environment's `bin` on PATH, the engine needed `ninja`, and the readiness loop waited instead of checking
the process (recorded in IDLE-LOG; the retry script fails fast).

Workflow (02:10 UTC). A job queue (`experiments/ops/queue.py`) with a detached dispatcher, a checked
launcher (`launch.py`: argument dry run, manifest, early health check) and one status view (`status.py`)
replace hand launches; the dispatcher's first version never reaped its wrappers and so never freed a
lane, fixed at 02:17. MIG slices report no utilization, so load is read from vLLM's running and waiting
counts: GPU1's 4B (port 8001) was saturated with 90 waiting while GPU0's replica (8006) ran 5, so new
helper-heavy jobs go to 8006.

Design lesson, stated once: a taught option that the warm start assigns near-zero probability cannot
be found by on-policy RL, so the teaching prior over alternatives has to be balanced before RL is
asked to choose between them.

## What this supports, and what it does not (draft at 15:00 UTC; seed 2 and 0.3-cost pending)

Supported by the evidence above:

1. A 4B controller can be taught the RLM decomposition routine from a few hundred native-replay
   trajectories and then executes it as well as the scripted expert in distribution, routing cheap
   single-helper lookups and chunked extraction correctly where direct answering and the untrained
   controller fail.
2. Two teaching-design facts matter more than data volume: the first turn must be family-agnostic
   because the controller sees only metadata before it inspects, and taught programs must be short
   and selected by the question, because a 3k-character program is not reproduced reliably.
3. On a real long-context benchmark (OOLONG-synth, held-out windows, 1k to 32k tokens), the taught
   controller reaches 0.567 and 30 updates of on-policy RL raise it to 0.652, within the expert's
   interval and three times direct answering at similar cost. The RL gain is a length-generalizing
   fix of chunk size at 32k tokens that was never seen in training, and it comes from the
   correctness signal; the 0.1 cost term is inert.

Not supported, or open:

- No evidence yet that a controller learns *when not to decompose*: on these task families
  decomposition is always right for aggregation and the lookup path is already in the teaching.
  The routing shown is between taught programs, not between recursion and direct answering.
- RL moved chunk size in opposite directions on the two families under the 0.1 cost term (both
  seeds) but not at 0 or 0.3; the mechanism is unresolved. At 1024 records every RL adapter is
  2 to 5 of 16 tasks below the warm start regardless of chunking, so RL may cost some
  out-of-range robustness; 16 tasks cannot separate that from noise.
- One model family and size, one benchmark, synthetic event logs; the OOLONG verifier is a
  format-tolerant exact match, not the official scorer; costs are endpoint-reported tokens with
  the harness prompt repeated every turn.

Ranked next comparisons (each a few GPU-hours on this setup):

1. A task family where direct answering beats decomposition for some inputs, so "do not decompose"
   is a rewarded choice; measure routing accuracy against an oracle router.
2. Replace the deterministic expert by diverse teachers (prompt and chunk variants are a start)
   and test whether RL from a higher-entropy warm start finds better programs than any teacher.
3. Scale the controller (Qwen3-8B) with the same recipe to see whether the SFT/expert gap and the
   RL gain change with capacity.
4. OOLONG at 64k to 128k tokens for the RL controller, where budget-safe chunking matters most.

## Operations notes (continued)

- The project volume reached its 1.1 TB quota at 06:46 UTC and an adapter save failed. The study's
  heavy outputs (adapters, RL runs, new evaluations) now live under
  `/home/atowell/research-runs/selective-recursion-20261019/` with symlinks at the original paths;
  the re-downloadable OOLONG parquet cache was removed (the exported JSONL and provenance remain).
  Nothing belonging to earlier sessions was deleted. The largest consumers are earlier replication
  artifacts: `runs/rlm-research-r4` (324 GB), `~/research-runs/rl-paper-reproduction-20261012`
  (291 GB), `~/research-runs/r1-zero-replication-20261007` (185 GB),
  `~/research-runs/spo-reproduction-20261008` (171 GB), the CURL replication runs (about 230 GB
  across eight directories) and the Hugging Face model cache (100 GB). Clearing any of these is the
  user's decision.
- SFT-003 (diversified but family-specific first cell) was stopped to give GPU 0 to SFT-004; it is a
  low-value ablation given the SFT-002 diagnosis and can be rerun later if wanted.

## Limits to keep in view

Synthetic templates and vocabulary; one model family and size; one training seed per stage
until a signal appears; cost is counted in tokens reported by the serving endpoint; the
programmed split is a strong task-specific baseline written with knowledge of the generator.

## Operations notes

- GPU 0 sat idle from 05:39 to 06:01 UTC (about 22 minutes) because the first pipeline
  serialized SFT-002 behind a slow zero-shot OOLONG evaluation that only needed GPU 1. The pipeline
  was restarted so training begins as soon as the card is free. Avoidable idle time, recorded as
  such.
- Two shell self-terminations happened when `pkill -f` patterns matched the invoking command line;
  later kills use bracketed patterns and PID loops.

