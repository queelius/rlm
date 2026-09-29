---
title: "What the current reinforcement-learning experiment actually shows"
date: 2026-09-29
status: exploratory_native_results_reviewed
evidence_cutoff_utc: "2026-09-29T14:02:51.479831Z"
latest_addendum_cutoff_utc: "2026-09-29T18:57:05.251012Z"
optimization_data: "Group A, eight official TRAIN goals, four attempts per collection"
diagnostic_data: "Group B, eight other official TRAIN goals, two fixed attempts per goal"
selection_rule: "Report every fixed endpoint; do not use B to select training or checkpoints"
---

# The latest RL results, without overstating them

> **Later addendum, September 29, 18:57 UTC:** Both third fixed RL readouts
> are complete. Raw-action success is now **4→4→5→4/16** and ingredient-assisted
> success **5→4→4→4/16**, including the SFT-trained starting points. The raw
> model lost its second-update gain on TRAIN1273, repeat 1; the other fifteen
> outcomes were unchanged. One-step extra-SFT controls remain 4/16 and 5/16;
> they are not matched to three RL steps. Both third readouts have all sixteen
> native outcomes, no unknown slots and zero transport failures. Raw used
> 898 calls / 27,481 generated tokens; assisted used 780 / 25,785. These are
> repeated diagnostic goals, not independent confirmation. Native paths and
> immutable source hashes are pinned in the updated
> [deck evidence](../../slides/2026-09-30-advisor-meeting/evidence.json).
>
> **Decision:** Do not describe the second-update gain as persistent transfer.
> Finish bounded accepted work, but prioritize a prospective teaching-history
> comparison over more unchanged RL checkpoints. The
> [paper assessment](publication-assessment.md) proposes that next study;
> it does not claim a queue change or a newly launched experiment.

## Earlier snapshot through 14:02 UTC

The first full comparison is complete. One RL update did not improve success
on the different-goal diagnostic panel. One additional supervised update also
left success unchanged. A second raw-action RL update has now gained one
successful attempt. The second assisted-action update remains at 4/16, one
below its starting model and unchanged from its first update.

| How the action is executed | Starting model | One RL update | Two RL updates | One extra supervised update |
| --- | ---: | ---: | ---: | ---: |
| The model writes the ingredient arguments. | 4/16 | 4/16 | 5/16 | 4/16 |
| Code writes ingredient arguments from a recipe already observed. | 5/16 | 4/16 | 4/16 | 5/16 |

These are the same eight diagnostic goals and two random seeds throughout,
not new independent samples at each checkpoint. The baseline is reused from
the earlier transfer comparison. Group B excludes the training target roots,
but shares a recipe world and some component recipes. It is not the official
test split. Extra supervised training matches the starting checkpoint, learning
rate, clipping and one optimizer step, not the data, supervised-token count,
likelihood temperature or total computation. At RL step two, it is no longer
even an equal-step comparison.

## The small new success is real, but not a broad transfer result

The raw-action second update gains TRAIN1273, repeat 1, with no losses on the
other fifteen attempts. The environment requests two `o1_i4` items. The starting
model finishes with none; the second-update model finishes with three, after
57 calls and 19 rejected actions. Its last steps make a missing prerequisite,
then craft the final item. This is a real task completion, not merely better
formatting or a reduced training loss. It is still only one newly successful
attempt on one repeatedly examined goal. We retain all endpoints and do not
choose this checkpoint because its score is the best so far.

Native trace paths, relative to the study root below:
`raw/readout-warm/nodes/t03-r1-flat-n0.json` and
`raw/readout-0002/nodes/t03-r1-flat-n0.json`.

The assisted second update solves the same four attempts as its first update.
It still loses the original TRAIN1796 success: it makes two of the three
requested items and then says it is finished. Total calls rise to 793 and
generated tokens to 25,909, versus 673 and 21,759 for its starting model.
There are 314 native action errors, four invalid actions and four context-limit
stops, with zero transport failures. Invalid formatting falls sharply from the
first update's 22 cases without restoring task success. This further separates
valid action syntax from useful task behavior; it does not identify a training
fix from this diagnostic panel.

## Extra SFT makes this batch cheaper, not more successful

The completed assisted extra-SFT control solves exactly the same five attempts
as its starting model. Calls fall from 673 to 633 and generated tokens from
21,759 to 19,552. All net savings are among the eleven failed attempts. The
five successful attempts use the same 81 calls and 2,155 tokens in both runs.
The raw-action extra-SFT control has the same pattern: unchanged four successes,
calls 778→677 and tokens 23,516→21,254, with all net savings among failures.

This distinguishes cheaper attempts from better solving. In the assisted
condition the only context-limit stop becomes an explicit unsuccessful finish;
all sixteen outcomes remain known. Neither interface has transport failures.
The first RL updates instead increase total calls and generated tokens.

## The training data give a sharper reason to change the next experiment

Our terminal-reward update compares four attempts at the same training goal.
If all four fail, all four receive zero *relative* learning signal. The same
is true if all four succeed. Shared parameters can still change these goals'
behavior through training on other goals.

| Training collection | Successful attempts | Goals with both success and failure, out of 8 | Generated tokens with nonzero relative credit |
| --- | ---: | ---: | ---: |
| Raw, first collection | 5/32 | 3 | 28.6% |
| Assisted, first collection | 12/32 | 3 | 25.0% |
| Raw, second collection | 7/32 | 2 | 21.5% |
| Assisted, second collection | 7/32 | 3 | 25.3% |

Four training goals (1921, 1051, 2338 and 2026) never succeed in either
interface in either round: **0/64 attempts across four goals**, not sixty-four
independent problems. They consume 4,231 calls and 137,331 generated tokens.
They supply no direct terminal-relative gradient in these collections. This
is the most decision-relevant new A-side observation: roughly three quarters
of sampled output tokens across the experiment receive zero relative credit,
and the same difficult goals remain outside the sampled success set.

Do not interpret 5→7 or 12→7 as controlled learning gains or losses. The random
seeds change between rounds, and after the first update the raw and assisted
actors have different weights. Only the first collection compares the same
unchanged actor across interfaces. Its 5→12/32 improvement is an execution
assistance effect, not an RL gain.

Both second updates are committed and usable. Their positive-action token
probabilities rise and negative-action probabilities fall on average; the
measured train/evaluation replay gap is zero. The optimizer is doing real
work. This does not establish that its update is large enough, well-directed
for future states, or sufficient to discover a missing procedure.

## Decision

Finish the already fixed four-update study and queued objective controls;
report every endpoint. Do not extend the same terminal-only recipe by default.
For the next intervention, prioritize **learning to recover from states the
model actually reaches**, against an ordinary-demonstration control on the same
A goals. The public-teacher recovery dataset is already CPU-qualified, but no
GPU training result exists. Evaluate its first decision on ordinary A starts,
not B. If it does not expose useful new competence, the next small comparison
is a fixed public-prefix curriculum with the prefix cost charged and only the
generated continuation trained. This decision follows the A-side lack of
reward contrast, not a search over B's errors or scores.

In parallel, finish the accepted second-model teaching-repair test and the
complete-task helper comparison. Those answer different, more important
publication questions than another slightly different endpoint on the same B
panel. Do not duplicate them: they remain in the existing follow-on queue.
This review changes the documented priority, not a sealed live job's inputs
or the external queue.

## Evidence and verification

[fresh-results.json](fresh-results.json) records all eight completed readout
cells, all paired outcomes, costs, four training collections/updates, endpoint
identities and source hashes. Checked 384 small native receipt bindings:
episode, root node and first returned call for every attempt. Verified the
common dataset, seed and runtime contracts, plan-bound responses, native
replay scores, and aggregate costs. Reused completed native replay audits;
did not rehash model tensors or regenerate results.

External study root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-fresh-rl-20260928-002`.
`COMPARISON-0001.json` and both second-step native summaries are complete.
The last assisted attempt finished at 14:02:51 UTC. Its 48 additional
episode/node/first-call hashes and common runtime, task and sampling contracts
were checked against the completed native replay audit. Earlier fixed-cutoff
explanations remain in the
[fresh RL report](../../experiments/selective_delegation/fresh_rl_results_20260929/README.md).
