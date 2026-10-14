# Resuming research on teaching RLM competencies

## Latest: October 14, 22:18 UTC — panel03 complete; new-structure test running

The full next-panel comparison is now available. Across the same eight goal
identities, two worlds and two rollout draws:

| Teaching policy | Whole-task successes | Model calls | Generated tokens |
|---|---:|---:|---:|
| Original-known teacher | 3/32 | 1,171 | 24,594 |
| Stable-visible repair | 17/32 | 779 | 19,218 |
| Random-visible repair | 14/32 | 1,030 | 26,650 |
| Discovery teacher | 22/32 | 1,015 | 32,341 |

**The useful lesson is that better examples help against our weak original
teacher, but a strong teaching baseline changes the conclusion.** Neither
repair beats discovery on total successes. Stable repair uses fewer calls and
generated tokens than every other arm in this panel; this is a measured
success/cost trade-off, not evidence of universally superior teaching.

Stable repair gains fifteen paired attempts and loses one versus original;
random repair gains twelve and loses one. Against discovery, stable gains two
and loses seven, while random gains one and loses nine. Stable beats random
on four attempts and loses on one, with 27 ties. That small, single-fit
difference does not establish a reliable ordering advantage.

All **128 scheduled outcomes** are observed and match their saved native replay
scores. Matching task identities, starting inventories, seeds and inference
settings were checked, along with **8,300 unique audit-referenced file hashes**.
There are zero transport failures. One random-repair/world50 attempt reaches
the context limit and remains an observed failure, not an excluded run.
The [complete evidence receipt](2026-10-14-teaching-panel03.json) preserves
cell-level scores, costs, action errors, pairing checks and provenance. This
reconciliation reuses completed native replays; it is not another GPU run.

These remain previously exposed goals, repeated across worlds and draws,
using one fitted seed. They are not 32 independent tasks. Teaching histories
and prompt dose differ; we cannot isolate visibility alone. These are flat,
supervised tool policies—not new RL gains or learned recursive delegation.

A retrospective [behavior breakdown](2026-10-14-teaching-panel03-behavior.json)
helps locate the remaining problem. Across the four policies, 65 of the 72
failed attempts explicitly finish before producing the required inventory;
the other seven hit call/context limits. Discovery has the most native action
errors (432), but 387 occur in its ten failed attempts. Twelve of its 22
successful attempts also contain at least one native action error. Thus error
counts alone neither measure competence nor establish an ability to recover.
The breakdown checks all 256 episode/root-node file hashes and recomputes
target completion from the saved inventories. It suggests examining planning
and completion checking, without proving a causal mechanism. The prepared
recovery-versus-ordinary comparison, not this outcome-conditioned analysis,
will test whether the new teaching intervention helps.

After all eight predecessor jobs exited successfully and the entire queue and
its scoped GPU owners departed, the frozen **new-structure pilot launched at
22:17 UTC**. Its first real model response returned at 22:17:28; sixteen calls
had returned without transport failures by 22:17:50. The four prescribed
policies and all 64 planned attempts are unchanged from the prelaunch design
below. No new-structure score or completed comparison is claimed yet.

Next: finish this controlled transfer comparison, then the already prepared
recovery-versus-ordinary teaching comparison after the whole fresh queue
releases. Independent-fit replication and an evidence-backed decision about
matched RL remain required. The research goal is unfinished. This result stays
in supporting research evidence; it does not change the math-reproduction
deck's headline. The earlier dated sections below remain historical cutoffs.

## October 14 update: completed comparison and renewed research goal

The October 12 status below is retained as a historical checkpoint, not the
current queue. The math-RL reproduction has since completed; its result and
boundaries are documented in the [October 14 advisor deck](../../slides/2026-10-14-rl-reproduction/).
The user has now activated the goal of testing whether better teaching examples
enable useful RLM learning and publishing the findings.

All four prescribed panel02 repair readouts finished. Combining worlds42 and50:

| Teaching policy | Whole-task successes | Model calls | Generated tokens |
|---|---:|---:|---:|
| Original-known teacher | 1/32 | 1,102 | 22,740 |
| Stable-visible repair | 11/32 | 767 | 18,180 |
| Random-visible repair | 13/32 | 1,112 | 27,763 |
| Discovery teacher | 13/32 | 1,275 | 39,502 |

Each policy faced the same eight goal identities, two worlds and two rollout
draws. All 128 outcomes are observed, with zero transport failures. Rechecking
the existing native replay audits on October 14 matched every episode score,
paired goals/worlds/inventories/seeds/inference settings, and all 8,822 unique
files named by those audits. This was a receipt-integrity and pairing check,
not a new replay or model-weight rehash. Exact paths, hashes, costs and paired
outcomes are in [the evidence receipt](2026-10-14-teaching-panel02.json).

Stable repair gained eleven attempts and lost one versus the original teacher;
random repair gained thirteen and lost one. Against discovery, stable gained
two and lost four; random gained three and lost three. Thus repair helps the
weak original-teaching condition here, but neither repaired policy scores above
the stronger teacher. Equal aggregate scores do not establish equivalence.
Stable repair also uses fewer calls and generated tokens in this paired slice.

This remains an exploratory result on a baseline-exposed panel, using frozen
supervised checkpoints. Repeated draws and worlds are correlated, input history
and prompt dose change, and flat tool use is not recursive delegation. It does
not establish fresh-structure transfer, a visibility-only cause, or new RL gains.

The next numbered panel03 is now being evaluated under the existing source,
fixed checkpoint23 and seed2026092208, for both repairs and worlds42/50. Four
bounded science jobs each have a native replay follow-up. This extension must
not replace the main goal: fresh dependency-structure testing with original,
repair, discovery and longer-training controls, followed by a matched test of
whether RL adds value when useful starting behavior exists. Preserve negative
and incomplete outcomes; do not choose checkpoints on evaluation success.

Live queue and owner pointers are under
`/project/alex_phd/runs/rlm-teaching-goal-20261014/`. The old RLM observer remains
retired. No allocation renewal is assumed. This update belongs in supporting
research evidence; it does not change the math-reproduction deck's main claim.

## October 14, 21:28 UTC: a harder transfer check is ready

The first repaired-teacher readout on panel03/world42 completed: **7/16**
whole-task successes, compared with the original teacher's **1/16** and the
discovery teacher's **13/16**. All three have sixteen observed outcomes and no
transport failures. Paired against the original, repair gains seven attempts
and loses one; against discovery, it gains one and loses seven. The repair
uses 376 calls and 9,244 generated tokens. These results reinforce the need
for a strong teacher control: improving a weak starting point does not mean
we have found the best way to teach the model.

The [partial evidence receipt](2026-10-14-teaching-panel03-w42-partial.json)
checks matching goals, starting inventories, seeds and inference settings,
reconciles all native replay scores, and rehashes 2,755 unique audit-referenced
files. The remaining three prescribed repair/world cells are still in progress;
this is not a complete panel03 result or an untouched transfer test.

To test transfer more directly, we froze **eight new crafting problems before
any model evaluation**. Their prerequisite structures differ from all 4,477
distinct structures in our recorded catalog of recipe worlds42–52. This is
an exact rooted-graph comparison that ignores item names and quantities while
preserving shared prerequisites—not merely a renaming exercise. We also
exclude 127 previously exposed goal names. Four selected goals have dependency
depth three and four have depth four; each needs at most twelve recipe products.
The selection rule was fixed after a CPU-only feasibility probe, without new
model scores. A native constructive solution succeeds for all eight; this
checks that the tasks are solvable, not that a learned policy can solve them.

The prospective comparison has four fixed policies:

- Original teaching after 23 updates.
- Repaired teaching after 23 updates.
- Discovery teaching after 23 updates.
- Original teaching after 69 updates, to check whether more training suffices.

Each gets the same eight problems and two rollout seeds: **64 planned attempts**.
Checkpoint choices, task order, starting stock, sampling and inference caps are
fixed. All four CPU preflights and eight focused tests passed; no fresh-task
model calls have run at this cutoff. The source, actor identities, task hashes,
qualification details and environment lock are recorded in the
[prelaunch receipt](2026-10-14-teaching-fresh-prelaunch.json). The ready queue
must wait until the entire existing queue and its scoped owners release.

This is still a pilot: one fitted seed, one synthetic recipe world, familiar
vocabulary and shared intermediate ingredients. One selected goal is even a
prerequisite of another. Initial stock is conservatively sufficient and has
no distractors, unlike the earlier panels. We will not treat the eight goals
as independent domains, compare headline percentages across different task
sets as a treatment effect, or describe flat tool use as learned recursion.

The next decision is whether improved teaching survives this new-structure
check and warrants broader replication. Testing whether matched RL adds value
remains part of the goal; this inference-only pilot does not answer that question.
This checkpoint belongs in supporting research evidence, with no new advisor
deck headline until the comparison finishes. Native checkpoints remain external;
publishing their hashes does not back up their contents.

## October 14, 21:43 UTC: second result and recovery-teaching readiness

Random-order repair completed **6/16** successes on panel03/world42. Stable
repair scored **7/16**, original teaching **1/16**, and discovery teaching
**13/16**. Random and stable repair differ on only three paired attempts:
random wins one and stable wins two. This small slice does not establish an
ordering advantage. Random repair used 511 calls and 12,754 generated tokens,
with zero transport failures but 127 native action errors. All sixteen scores
match native replay; 1,086 audit-referenced file hashes and the paired initial
inventories were checked. The world50 comparisons are still running.

The next queued experiment asks a different question: **does teaching the
model how to recover from its own mistakes help more than more ordinary
examples?** Both arms begin at the same discovery-teacher checkpoint. Each
gets one supervised optimizer update on 32 frozen examples from the same
eight training goals, with 823 supervised tokens. We then test whole tasks
from their normal starting states, alongside the unchanged starting model.
Doing better at predicting the teaching answers is only an optimization check;
the useful outcome is completing more tasks without a helpful error prefix.

Both real datasets passed full token re-encoding checks. Ten focused CPU tests
pass, including a small actual optimizer update and the native prepare-only
entrypoint. A separate code review found no scientific-contract defect but
required an outer watchdog: the prepared five-job queue now supplies hard
20-minute fit and 90-minute readout limits, with a 310-minute whole budget.
Its 33 unique input/source pins were checked. These are launch preparations,
**not GPU training results**. The full preceding fresh-task queue must release
before admission, and remaining allocation time must be checked again.

Recovery and ordinary examples differ in history, eight target labels, and
prompt length (80,120 versus 65,044 tokens); this is not a pure history-only
intervention. The evaluation is on training-side goals, not untouched transfer.
It is supervised teaching, not RL, and one null update cannot rule out a
larger-dose benefit. The original exploratory advance thresholds and prohibition
on B-driven selection are preserved in the queue. Exact plans, review findings,
checks, costs and raw-result pointers are in the
[evidence receipt](2026-10-14-teaching-recovery-prelaunch.json).

The research order remains: finish panel03, test the frozen new structures,
then test recovery versus ordinary teaching. Broader replication and matched
RL added value remain unresolved. No new advisor-deck headline is warranted
by this partial, previously exposed panel.

## October 14: reconcile the earlier RL endpoint before designing another run

The older September 29 narrative ended at the third update. The native records
contain a fourth raw-action update and completed evaluation. Reconciliation now
covers all **176 observed diagnostic outcomes and 256 training trajectories**,
checking 1,296 episode/root-node/first-call hashes, native replay scores, paired
task/inventory/runtime contracts, and saved probability-replay diagnostics.
These are existing runs, not new GPU experiments or independent replications.

| Policy interface | Warm | RL step1 | Step2 | Step3 | Step4 | One extra SFT step |
|---|---:|---:|---:|---:|---:|---:|
| Model writes all action arguments | 4/16 | 4/16 | 5/16 | 4/16 | 4/16 | 4/16 |
| Code supplies observed recipe arguments | 5/16 | 4/16 | 4/16 | 4/16 | Not evaluated | 5/16 |

The final raw model gains TRAIN1796 on one seed but loses TRAIN1847 on that
seed. For the gained attempt it makes four requested items against a target of
three; for the lost attempt it finishes without making the requested product.
Equal totals therefore do not mean an unchanged policy. Calls rise from 778 to
905 and generated tokens from 23,516 to 27,727. This fixed endpoint supplies
neither a net task-success gain nor a cost saving on the reused eight-goal B
diagnostic. The temporary step2 gain was not sustained. No best checkpoint is
selected, and the one-step SFT control is not a four-step dose match.

The assisted fourth fit stopped before a new optimizer update; its evaluation
was explicitly skipped, not scored as zero. One of 51,588 sampled-token
log-probabilities exceeded the fixed maximum generation/replay-gap gate:
0.2500000596 against 0.25. The mean gap was 0.0005573, below its 0.025 gate.
That reconstructs the exact stop condition, not the numerical cause of the
underlying discrepancy. We preserve the failed attempt and the threshold;
we have not resumed it or claimed a fix.

Most importantly for the next experiment, the same four A training goals
(1051, 1921, 2026 and 2338) had **zero successes in 128 attempts** across both
interfaces and all four collections. These are four shared-world goals sampled
repeatedly, not 128 independent problems. Their all-failure groups supply no
direct relative terminal-reward gradient, although updates from other goals
can change their behavior. This supports testing whether recovery teaching
creates useful ordinary-start successes before repeating the unchanged RL
recipe. It does not prove that sparse reward is the sole cause of failure.

The [reconciliation receipt](2026-10-14-teaching-prior-rl-reconciliation.json)
preserves all cells, collection signal, endpoint checks, missingness and the
next-decision boundaries. Current GPU jobs and frozen follow-on queues remain
unchanged. A future RL comparison must use fresh on-policy data, fixed taught
endpoints, no-RL controls and explicit dose accounting. Diagnostic B outcomes
must not choose training inputs or favorable checkpoints. Broader transfer,
independent fits and RL added value remain open questions; the full research
goal is not complete.

## October 14, 21:57 UTC: stable repair completes the next panel

Stable-order repair has now completed both worlds on panel03:

| Teaching policy | Whole-task successes | Calls | Generated tokens |
|---|---:|---:|---:|
| Original known-recipe teacher | 3/32 | 1,171 | 24,594 |
| Stable-order repair | 17/32 | 779 | 19,218 |
| Discovery teacher | 22/32 | 1,015 | 32,341 |

Repair gains fifteen paired attempts and loses one against original teaching;
against discovery it gains two and loses seven. Its world50 score of 10/16
is slightly above discovery's 9/16, but the combined comparison still favors
discovery. Repair uses fewer calls and generated tokens in this slice; that
is not a claim of equal accuracy or general cost dominance.

All 96 outcomes in this table match completed native replay, with zero transport
failures. The check reconciled matching task/seed/inventory/runtime contracts
and 6,170 unique audit-referenced hashes. The
[evidence receipt](2026-10-14-teaching-panel03-stable.json) retains every outcome.
Eight previously exposed goal identities are repeated across worlds and draws;
these are not 32 independent new tasks. The final random-order/world50 cell is
still running and will be retained, not replaced by the more favorable ordering.

This strengthens the narrow repair-versus-weak-teacher result, not fresh transfer
or superiority to discovery teaching. The frozen four-policy fresh-structure
pilot remains next, followed by recovery versus ordinary teaching. The completed
second-fit original/repair/discovery checkpoints are available for later reuse;
a matched longer-training control for that seed still needs preparation.
No new advisor-deck headline is justified before the fresh comparison.

## Historical October 12 checkpoint

**Priority correction, October 12:** the advisor's primary objective is a
completed reproduction of an RL paper's training result on our A100. That
objective remains unfinished. The already-admitted readouts below can finish;
new RLM teaching fits and exploratory extensions are deferred. The RLM-only
review observer is stopped so it cannot supersede this human instruction.

Research initially resumed on October 12 with a new single A100 allocation
and teaching experiments addressing the secondary research question:
**what preparation helps a model learn useful multi-step behavior, and does
that behavior carry to different tasks?**

The working lesson is not simply “use more RL.” A model must encounter useful
practice and feedback. Our next experiments distinguish better demonstrations,
learning to recover after mistakes, and changing the action interface. The
crafting studies below use flat tool policies; they inform RLM preparation but
are not themselves evidence of learned recursive decomposition.

## New results show a limited teaching repair benefit

At the October 12, 14:28 UTC cutoff, two of four prescribed new readouts
are complete. On panel02/world42, the stable and random-order repaired policies
solved 6 and 7 of 16 attempts respectively. All four conditions below have
complete native replay audits and matched goals, world, starting inventories,
rollout seeds and evaluation caps.

| Teaching policy | Whole-task successes | Model calls | Generated tokens |
|---|---:|---:|---:|
| Original-known teacher | 0/16 | 568 | 11,394 |
| Stable-visible repair | 6/16 | 348 | 8,021 |
| Random-visible repair | 7/16 | 457 | 11,358 |
| Discovery teacher | 8/16 | 576 | 17,772 |

Repair extends beyond its original evaluation goals, but neither repaired
policy outperforms discovery teaching here. Paired against discovery, stable
repair gains one attempt and loses three; random repair gains two and loses
three. Random order is one success above stable order, so this cell does not
support a stable-order advantage. Fewer calls are useful cost evidence, not a substitute
for completing the task. These are eight goals with two draws each, a single
training seed, and previously exposed baseline results. This is an exploratory
transfer readout of existing supervised policies, not new RL training or a
recursive-policy result. The other two prescribed readouts remain running
or queued; retain their results regardless of direction.

[First-result paired evidence](2026-10-12-teaching-panel02-world42.json) and
[updated native review](2026-10-12-teaching-review.json).

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

## Secondary teaching experiments

1. **Additional-goal demonstration transfer — two of four new cells complete.**
   Evaluate both fixed repaired-teacher policies on the next numbered eight-goal panel, in two
   recipe worlds and with two fixed rollout seeds. Reuse the corresponding
   original-teacher and discovery-teacher controls. All four prescribed cells
   are retained; no checkpoint selection. These goals have prior baseline
   exposure, so this extends the exploratory study rather than creating an
   untouched confirmatory test.
2. **Recovery examples versus ordinary examples — data ready, now deferred.** From the same
   warm checkpoint, compare one small supervised update on examples showing
   recovery after errors with one on ordinary examples. Start evaluation from
   the normal task state, not a helpful prefix. First ask whether the model can
   now succeed on the previously all-failure training goals; only then compare
   identical bounded RL. The two prepared arms have 32 rows and 823 target
   tokens each, but histories and eight labels differ. The trainer/readout
   adapter has not yet been implemented or launched.
3. **Names versus procedures — proposed, now deferred.** Consistently rename items while
   leaving recipes and scoring unchanged. This checks whether the repair gain
   depends heavily on familiar names or copying opportunities. It would not,
   by itself, establish generalization to new problem structures.

The current audience deck already describes limited transfer and teaching
interventions as hypotheses. Its story is unchanged by this reconciliation;
no new general-RLM claim or slide headline is justified yet.

## Execution and resumption

The new allocation ends October 15 at 13:00 UTC. Four bounded readouts and
native audits are admitted; this is not a claim of three days of ready jobs.
Both world42 readouts completed all 16 attempts without transport failures and
passed native replay. Automatic handoff to stable-visible/world50 succeeded;
the live check found 85 genuine returned calls and no transport failures,
including an actual crafting request. Earlier weights, optimizer checkpoints and failed attempts
remain untouched. No new math GPU job has yet been launched; the paper
reproduction handoff prioritizes diagnosing the existing SPO failure.

The live checkpoint is
`/project/alex_phd/runs/rlm-teaching-resume-20261012/SESSION_CHECKPOINT.md`;
the finite queue is `FOLLOWON-QUEUE.json` in the same directory. Its delivery
probe and first numbered review reached the session; the RLM-only observer
has since been stopped because paper reproduction takes priority. Primary
handoff: `/project/alex_phd/runs/rl-paper-reproduction-20261012/SESSION_CHECKPOINT.md`.
Preserve 10% account allowance with 12% dispatch headroom. Native
artifacts and model weights are external; this GitHub update is not their backup.
