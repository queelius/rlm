---
date: 2026-10-04
status: proposed_followups_not_launched
priority: replication_before_mechanism
---

# What the next comparisons would establish

Finish the prespecified three-seed CURL/no-CPC comparison before choosing another
arm. These are replication and interpretation checks, not novelty claims.

The [CURL paper](https://proceedings.mlr.press/v119/laskin20a/laskin20a.pdf)
combines random crops, instance contrastive learning and SAC, and benchmarks the
combined method across tasks. Its reported cartpole scores use ten training
seeds; our three seeds can establish a local pattern, not reproduce that precision
or the full benchmark claim. The [supplement](https://proceedings.mlr.press/v119/laskin20a/laskin20a-supp.pdf)
also makes task-specific action repeats explicit. One task cannot establish
general representation quality or transfer.

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
