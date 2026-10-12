# Resuming research on teaching RLM competencies

Research resumed on October 12 with a new single A100 allocation. The immediate
goal is to turn the tentative paper direction into controlled experiments:
**what preparation helps a model learn useful multi-step behavior, and does
that behavior carry to different tasks?**

The working lesson is not simply “use more RL.” A model must encounter useful
practice and feedback. Our next experiments distinguish better demonstrations,
learning to recover after mistakes, and changing the action interface. The
crafting studies below use flat tool policies; they inform RLM preparation but
are not themselves evidence of learned recursive decomposition.

## First new result: repair helps beyond the original goals

At the October 12, 14:16 UTC cutoff, the first of four prescribed new readouts
is complete. On panel02/world42, the fixed repaired policy solved 6 of 16
attempts. All three conditions below have complete native replay audits and
matched goals, world, starting inventories, rollout seeds and evaluation caps.

| Teaching policy | Whole-task successes | Model calls | Generated tokens |
|---|---:|---:|---:|
| Original-known teacher | 0/16 | 568 | 11,394 |
| Stable-visible repair | 6/16 | 348 | 8,021 |
| Discovery teacher | 8/16 | 576 | 17,772 |

The repair extends beyond its original evaluation goals, but does **not**
outperform discovery teaching here: paired against discovery, it gains one
attempt and loses three. Fewer calls are useful cost evidence, not a substitute
for completing the task. These are eight goals with two draws each, a single
training seed, and previously exposed baseline results. This is an exploratory
transfer readout of existing supervised policies, not new RL training or a
recursive-policy result. The other three prescribed readouts remain running
or queued; retain their results regardless of direction.

[Per-attempt evidence and native hashes](2026-10-12-teaching-panel02-world42.json).

## What the earlier completed runs now tell us

Several September experiments finished after their written summaries. The
[checked evidence](2026-10-12-teaching-resumption-evidence.json) records their
native report paths and hashes. These are recovered completed results, not new
October 12 training gains.

| Comparison | Completed result | Lesson |
|---|---|---|
| Demonstration repair on Phi | Original teacher 1/16; repaired 3/16; discovery teacher 4/16 | The large first-model repair effect does not carry over at the same magnitude. Eight shared goals are too few for a model-general claim. |
| Compact-action RL | 6/16 before, 9/16 after; three gains and no losses | A promising local learning result to replicate. The interface, warm-start training and input representation differ from the ordinary-interface study. |
| Masking arguments filled in by code | Warm 14/16; full-token RL 16/16; masked RL 16/16 | No observed advantage from the masking intervention; it used more calls than full-token RL. |
| Complete-task delegation | Flat 2/2; fixed routing one success and one unknown; adaptive routing 1/2 | A useful helper does not guarantee a useful overall delegation strategy. This is one exposed goal, not a general recursion comparison. |

The longer ordinary-interface RL sequence also needs all of its endpoints:

| Interface | Before RL | Update 1 | Update 2 | Update 3 | Update 4 | One extra SFT update |
|---|---:|---:|---:|---:|---:|---:|
| Model supplies action arguments | 4/16 | 4/16 | 5/16 | 4/16 | 4/16 | 4/16 |
| Code fills observed recipe arguments | 5/16 | 4/16 | 4/16 | 4/16 | Unavailable | 5/16 |

The last assisted update stopped before optimization at a numerical replay
check; it is not a completed bad score. The intermediate 5/16 result is not a
selected endpoint. These sixteen attempts cover eight different-goal TRAIN
problems, with shared recipes and repeated draws—not an untouched benchmark
or independent training replications.

## A concrete teaching problem

Four optimization-side goals never succeeded in 128 attempts across the two
ordinary interfaces and four collections. These are repeated attempts on four
problems, not 128 independent problems. With the terminal-reward, within-group
comparison used here, an all-failure group supplies no direct reward contrast.
Other RL objectives can use different feedback.

One characteristic failure is easy to understand: the model asks to craft an
item before making its ingredients, receives an error, then repeats the same
request. Filling in recipe arguments does not fix a missing prerequisite.
The compact policy sometimes escaped this pattern even before RL, suggesting
that how we present and teach the action sequence can change which successful
attempts are available to reinforce. That is a hypothesis to test, not a
visibility-only or interface-only causal conclusion.

## The next small experiments

1. **Additional-goal demonstration transfer — one of four new cells complete.**
   Evaluate both fixed repaired-teacher policies on the next numbered eight-goal panel, in two
   recipe worlds and with two fixed rollout seeds. Reuse the corresponding
   original-teacher and discovery-teacher controls. All four prescribed cells
   are retained; no checkpoint selection. These goals have prior baseline
   exposure, so this extends the exploratory study rather than creating an
   untouched confirmatory test.
2. **Recovery examples versus ordinary examples — data ready.** From the same
   warm checkpoint, compare one small supervised update on examples showing
   recovery after errors with one on ordinary examples. Start evaluation from
   the normal task state, not a helpful prefix. First ask whether the model can
   now succeed on the previously all-failure training goals; only then compare
   identical bounded RL. The two prepared arms have 32 rows and 823 target
   tokens each, but histories and eight labels differ. The trainer/readout
   adapter has not yet been implemented or launched.
3. **Names versus procedures — proposed.** Consistently rename items while
   leaving recipes and scoring unchanged. This checks whether the repair gain
   depends heavily on familiar names or copying opportunities. It would not,
   by itself, establish generalization to new problem structures.

The current audience deck already describes limited transfer and teaching
interventions as hypotheses. Its story is unchanged by this reconciliation;
no new general-RLM claim or slide headline is justified yet.

## Execution and resumption

The new allocation ends October 15 at 13:00 UTC. Four bounded readouts and
native audits are admitted; this is not a claim of three days of ready jobs.
The first readout completed all 16 attempts without transport failures and
passed its native replay audit. Automatic handoff to the random-visible
comparison succeeded; by 14:14 UTC it had 133 genuine returned calls and no
transport failures. Earlier weights, optimizer checkpoints and failed attempts
remain untouched. Old SPO, DrGRPO and CURL
campaigns remain stopped.

The live checkpoint is
`/project/alex_phd/runs/rlm-teaching-resume-20261012/SESSION_CHECKPOINT.md`;
the finite queue is `FOLLOWON-QUEUE.json` in the same directory. The new
event/20-minute review observer has been started, with a same-session delivery
probe admitted; its new delivery/acknowledgement cycle still needs to be
observed. Preserve 10% account allowance with 12% dispatch headroom. Native
artifacts and model weights are external; this GitHub update is not their backup.
