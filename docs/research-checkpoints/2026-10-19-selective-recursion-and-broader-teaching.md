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
`--oolong-train-lens`) is queued before the absolute-0.3 arm (`pipeline14_gpu0.log`).

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

