---
date: 2026-10-04
status: cartpole_complete_two_walker_pairs_complete_third_running
priority: replication_before_mechanism
---

# What the next comparisons would establish

## Current decision, October 5 at 05:43 UTC

The early paired advantage was +184.64 on average; at 500k it is +5.22, with
differences -24.20, +55.18 and -15.31. Every pair's gap shrank, but the later
sign differs across seeds. This is neither a consistent late advantage nor
evidence of equivalence. First-pair CURL's extra recovery cost remains explicit.

1. **Run the already-declared walker scope check.** It launched at 00:43:32 UTC
   after the cartpole queue ended. All three pairs, fresh weights, fixed 100k
   endpoint and unchanged settings in [WALKER_PROTOCOL.md](WALKER_PROTOCOL.md).
   The first two pairs finished at 482.83 versus 241.28 and 385.23 versus 360.96.
   Both favor CURL, but their differences are +241.55 and +24.27. The third
   CURL seed is running, with its control queued. Finish the cohort; two
   favorable signs with unequal gaps do not establish a stable benefit.
2. **Check sensitivity to the test starts with every frozen final model.**
   After all six runs finish, a separately declared panel of 50 fresh starts
   could be applied to all six fixed 100k checkpoints. Keep the original
   ten-start primary result unchanged. This probes evaluation variability,
   not variability from training new models. The second control's ten returns
   include 46.98 while the other nine are 290.65--503.13; retain all of them.
   The small +24.27 gap deserves a check on a fresh panel, without dropping
   difficult starts or selecting checkpoints. CPU inspection estimates 45--60
   minutes including loading, with only small output files. A proposed panel
   is starts 20000--20049, distinct from the original 10000--10009. Do not choose
   later checkpoints or training seeds because this panel favors them. An evaluation-only
   entrypoint is needed; this follow-up is prepared conceptually, not launched.
3. **Decide on more training seeds or a longer walker horizon from the cohort.**
   The first control was improving near its endpoint, so a longer horizon could
   distinguish delayed learning from a persistent gap if the pattern repeats.
   A mixed cohort would instead motivate more seeds. Do not stop after a
   favorable partial cohort or change
   the declared endpoints in response to interim rewards.
   CPU inspection finds all-six 100k-to-200k continuations feasible in roughly
   7--9 hours, versus 28--32 hours for direct 500k continuations. Both would use
   about 127 GB of additional space including one temporary checkpoint, because
   replay reaches capacity by 200k. Also reserve 19 GB for the unfinished
   third pair's parent checkpoints; those were not yet included in the 05:42
   free-space reading. Recheck shared project quota before any launch.
   Admission and analysis are still specific
   to cartpole and need a small explicit walker extension; the learner does
   not. Do not launch against unmodified cartpole admission. Resume semantics
   reset simulator state at natural episode boundaries; these would not be
   bit-identical uninterrupted runs. Neither continuation has been launched.
4. **Keep the shared-checkpoint matching-on/off intervention conditional.** It
   asks whether continued extra learning helps after an early foundation. It
   should not be framed as repairing proven late harm. Removing the update also
   removes optimizer work; correspondence information needs a separate control.

This remains a learning reproduction, not a novelty claim or a result about
language-model agents. The complete cartpole guide and native records are the
checkpoint; GPU work now answers the second-task question.

## Earlier decision after two longer pairs, October 4 at 22:48 UTC

The second pair did not repeat the first reversal: CURL scored 866.14 versus
810.96, a +55.18 difference, while the first pair was -24.20. Both late
differences are smaller than their early differences, but point in opposite
directions. Two pairs are not enough to estimate a stable later effect.

1. Finish the third prespecified pair without changing endpoints or settings.
2. Prepare a second task, walker/walk, to ask whether the early benefit extends
   beyond cartpole. The [task protocol](WALKER_PROTOCOL.md) records the paper's
   task-specific settings, three declared seed pairs and separate evaluation
   cost. The queue's small task-forwarding change passed nine focused tests
   and independent review. CPU preparation overlaps active third-pair work;
   no new task has launched yet.
3. Retain the shared-checkpoint matching-on/off experiment below as a direct
   test of continued value. Do not motivate it as a remedy for established
   general late harm. Gradient diagnostics remain secondary to causal training
   comparisons and scope checks.

This revises the emphasis, not the running queue. The first reversal was useful
evidence; it is not a conclusion to defend when another seed differs. Detailed
second-task settings will be frozen before its results are observed.

## Earlier decision after the first longer pair, 20:50 UTC

For seed 123, CURL led 678.02 versus 454.47 at 100k, but finished 842.74 versus
866.94 at 500k. The other two longer pairs are still being run. Finish them
before treating this reversal as a repeatable pattern. The first pair has extra
CURL recovery cost, so it cannot establish efficiency at equal physical cost.

If the reversal repeats, the most direct next question is **whether continuing
the image-matching exercise helps or hurts a model that already learned with it**.
This is different from asking whether the no-matching learner eventually catches up.

1. **Fork the same saved 100k CURL state into two declared training branches.**
   One keeps the image-matching update; the other disables it after 100k. Restore
   the same policy, critic, targets, replay, optimizers and random state. Keep
   random crops, positive-crop sampling and the RL encoder update. Compare fixed
   500k scores and a declared curve summary; do not choose the best checkpoint.
   If disabling matching wins repeatedly, continuing that whole extra update has
   negative incremental value in this setting. A tie while the scratch control
   catches up favors a history/catch-up explanation, but does not prove no harm.
   Removing matching also removes optimizer work: this is not a pure test of
   correspondence information. Preserve the reference's two encoder optimizer
   steps in the on branch. Approximate budget: two A100-hours per matched fork
   pair, six hours for three prespecified parents. These are planning estimates.

2. **Use a small gradient diagnostic only to help explain that comparison.**
   On prespecified saved experience batches, compare the directions and sizes
   of the RL and image-matching gradients, without updating weights. Opposing
   directions would motivate a conflict hypothesis, not prove poor control.
   Adam history and overlapping optimizers complicate the link. Expected cost:
   minutes after checkpoint loading, no new training interactions. It should
   not replace the actual on/off training comparison.

3. **A second task answers a different question: does the early pattern transfer?**
   A matched walker/walk pair at 100k, action repeat 2 and 50,000 decisions, would
   test scope beyond cartpole. Freeze the task-specific paper settings first.
   Budget roughly two to three A100-hours per pair, then measure and replicate.
   This cannot by itself explain why the first cartpole ordering reversed.

These are conditional proposals, not launches or novelty claims. The
[pinned implementation](https://github.com/MishaLaskin/curl/blob/8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc/curl_sac.py#L421)
defines the preserved update, the [paper](https://proceedings.mlr.press/v119/laskin20a/laskin20a.pdf)
describes shared representation learning, and the
[supplement](https://proceedings.mlr.press/v119/laskin20a/laskin20a-supp.pdf)
supplies task-specific settings. Augmentation-only prior work is discussed below.

The current resume validator rejects changing an arm's scientific configuration.
Any switch-off study must explicitly record a new schedule intervention and its
ancestry; do not relabel a CURL checkpoint as a scratch no-CURL run or weaken
the current continuation checks. Both fork branches come from one training seed,
not two independent seeds. The clean new on/off pair avoids making the recovered
seed-123 continuation the sole mechanistic reference.

## Earlier plan and source context

The prespecified three-seed CURL/no-CPC comparison is complete. All three pairs
favor CURL at 100k. The 500k continuation queue launched at 18:36:58 UTC; other
comparisons below remain proposals. These are replication and interpretation
checks, not novelty claims.

At 19:37 the repaired owner resumed CURL from 307k after a save-format failure
at 409k. The control saved cleanly at 198k and is next. All six original models
remain included. The immediate decision is still to finish the longer-horizon
comparison, not add a learning variant in response to a serialization failure.
Retained history and physical interaction cost now differ for the recovered
CURL seed; report both. A clean fresh-seed paired run would be needed before
making an equal-physical-budget claim about this recovery comparison.

The [CURL paper](https://proceedings.mlr.press/v119/laskin20a/laskin20a.pdf)
combines random crops, instance contrastive learning and SAC, and benchmarks the
combined method across tasks. Its reported cartpole scores use ten training
seeds; our three seeds can establish a local pattern, not reproduce that precision
or the full benchmark claim. The [supplement](https://proceedings.mlr.press/v119/laskin20a/laskin20a-supp.pdf)
also makes task-specific action repeats explicit. One task cannot establish
general representation quality or transfer.

The paper's Table 1 reports cartpole CURL at **582 ± 146 after 100k steps**
and **841 ± 45 after 500k steps** (mean and standard deviation across ten
training seeds). This motivates checking the later endpoint: the early score
is not the method's final level of performance. These published numbers are
context, not pass/fail thresholds for our three-seed modern-stack comparison.
The paper's Pixel SAC baseline is not our same-crops control.

The augmentation-only question already has substantial precedent.
[RAD, especially §I](https://arxiv.org/pdf/2004.14990), trains the RL objective
on augmented observations without an auxiliary objective; its
[official update](https://github.com/MishaLaskin/rad/blob/18d079e677398c70ff2eefefcc81d5a99662103d/curl_sac.py#L432)
samples augmented replay and leaves the contrastive call disabled.
[DrQ, §3 and Appendix F](https://arxiv.org/pdf/2004.13649), adds augmentation
averaging for target and current Q estimates. Its
[official critic update](https://github.com/denisyarats/drq/blob/dd040f144ed8f696c7db35087cd80c87372edb64/drq.py#L202)
averages two targets and fits both current views. Removing CPC from our fixed
CURL configuration is therefore a useful matched control, but does not reproduce
every RAD or DrQ configuration. Source review date: October 4, 2026; code links
pin the inspected revisions.

## Different controls answer different questions

| Comparison | Question and interpretation limit |
|---|---|
| Queued no-CPC vs CURL | Does the whole additional contrastive update improve return beyond these same crops? Retains positive-crop sampling. Removes encoder/W optimization work too, so does not isolate correspondence learning from additional updates. |
| Shuffled positives vs matched CPC | Does correct correspondence matter when both arms perform the same forward/backward passes and optimizer steps? Freshly derange keys without relabeling targets. Incorrect labels can harm learning; this is not a neutral compute-only control. |
| One query-encoder optimizer vs upstream two | Is performance sensitive to stepping the query encoder twice on its contrastive gradient? Keep critic updates and W training. Removing the duplicate step also changes auxiliary update magnitude and Adam history; this is a sensitivity test, not an established bug fix. |
| Longer budget / second task | Does an early effect persist, and does it extend beyond cartpole? These test learning horizon and task scope, respectively, rather than isolating a loss mechanism. |

The optimizer overlap is visible in the
[pinned CURL constructor and update](https://github.com/MishaLaskin/curl/blob/8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc/curl_sac.py#L299).
Preserve it in the reference. A later single-step variant should explicitly
state which optimizer retains the encoder and which retains W.

## Ranked next work after the valid 100k pair

1. **Extend both arms and all three seeds to 500k simulator steps.** This is the
   nearest additional published endpoint. Resume natural-boundary 100k states
   without another warm-up; retain the original 100k results. A shrinking gap
   suggests an early learning advantage; a persistent gap supports a longer
   horizon effect; reversed ordering revises the 100k story. If neither arm
   learns, investigate observations and updates before buying more interaction.

2. **If CURL has a repeatable advantage, add shuffled-positive CPC at 100k for
   the same three seeds.** CURL above both no-CPC and shuffled CPC would support
   useful correspondence information. CURL similar to shuffled CPC would weaken
   that explanation. Shuffled below no-CPC chiefly demonstrates harm from false
   targets. Track duplicate replay indices; deranged positions can still contain
   identical observations. Use a separate permutation RNG so sampling changes
   do not introduce another avoidable difference. If the initial effect is
   unresolved, prioritize a second published task, such as walker/walk, instead;
   freeze its action repeat 2 and count decisions separately.

Both use one A100 40 GB, sequentially, batch 128 and the existing pixel network.
The first reference currently suggests roughly 15 minutes per 100k run with ten
evaluation episodes every 500 decisions; this is provisional, not final timing.
Budget 15–25 minutes per shuffled run and approximately 60–90 additional minutes
per 100k-to-500k extension, plus checkpoint I/O. A walker 100k run needs 50,000
decisions at repeat 2: allow roughly 60–90 minutes, then measure. These are
planning extrapolations. Keep seed-level curves, fixed endpoints and failed runs
visible; evaluation episodes are not additional training replicates.

## Preparation decision after the second reference run

The second CURL seed learned but finished at 446, versus 678 for the first.
Its control was still running at the 18:02 UTC review cutoff. This favors
finishing the planned pairs before adding a loss variant. We can prepare a
longer-training comparison without deciding its outcome in advance.

The continuation batch will use a **separate sibling output directory**, but
the **same GPU lock** as the initial batch. This keeps the original 100k
results unchanged and avoids concurrent jobs on the reserved GPU. Preparation
is not a launch: all six original runs must first be checked for valid completion.

The narrow changes needed are:

1. Let a new queue snapshot explicitly pass each parent's checkpoint as
   `--resume`, recording its path, checksum and matching configuration. Merely
   increasing the step budget in today's queue would start fresh training.
2. Use a total budget of 62,500 decisions at action repeat 8: 500,000 simulator
   steps. Restore the parent's 12,500 decisions and 11,500 updates, without
   another warm-up. Require a natural episode boundary, not a budget-truncated
   episode. The terminal update count should be 61,500. Preserve the original
   parent checkpoint in place.
3. Allow an explicit shared lock path when the output root is different.
   Keep the original sealed queue unchanged. Make the review observer aware
   of both output roots, or explicitly retain periodic review of the new root.
4. Analyze the extension with an explicit 500k endpoint and parent links. A
   continuation is the same training seed, not an additional independent run.
   Do not point the old 100k summary at both phases: it deliberately rejects
   repeated arm/seed identities and would then exclude valid original results.

The child will not repeat the evaluation already saved at 100k. Its curve must
link that parent point explicitly. Keeping the same evaluation schedule adds
100 evaluation batches and one million separate evaluation simulator steps to
each extension; none of those interactions is used for training.

Retaining six 100k parents and six 500k children is expected to require roughly
85 GB, plus temporary space during an atomic checkpoint save. That is a planning
estimate, not a quota measurement. Host memory is ample for one sequential run;
check project quota before admitting the batch. Keep the two-hour invocation
caps and 15-minute checkpoint target, subject to the remaining allocation.

The checkpoint restores learning state and random generators at an episode
boundary. It does not serialize simulator physics or frame history. We will
describe these as documented resumed continuations, not promise bit-for-bit
identity to an uninterrupted GPU trajectory.
