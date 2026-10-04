---
date: 2026-10-04
status: three_pairs_complete_500k_technical_recovery_running
priority: replication_before_mechanism
---

# What the next comparisons would establish

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
