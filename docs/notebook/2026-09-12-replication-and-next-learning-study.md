# A small RL gain did not survive repetition

Evidence cutoff: September 12, 2026, 13:06 UTC. These are preliminary results,
not a claim of reliable RL improvement. GPU work continues beyond this snapshot.

## What we learned

We trained the helper on 128 new news articles and tested it on 256 separate
articles. One training run corrected one test answer. Repeating the training
with different random samples did not reproduce that gain.

| Model | Correct answers out of 256 |
|---|---:|
| Starting model | 211 |
| First one-update RL run | 212 |
| Repeated one-update RL run | 211 |

The repeat's predictions matched the starting model on every article. The
training updates really changed weights, and the reward and replay checks passed.
Thus this is not evidence of a broken optimizer, but neither is it useful learning.
We are retiring the one-answer positive signal rather than presenting it as a gain.

The training answers help explain why learning may be difficult. Only five or
six of 32 prompts produced answers with different scores across their four
samples. Most prompts offered no better-versus-worse comparison. Merely splitting
the score into four item scores would not create additional contrast in these
saved samples. That does not rule out a genuinely different way to assign credit.

## The next comparison

Before reading the repetition result, we froze 1,024 additional training articles
and 512 separate test articles. We will compare eight RL updates with eight
updates learned from correct example answers. Both start from the same model and
use the same articles and groups. Their losses, answer samples and token counts
differ, so this is a practical learning comparison, not an isolated test of the
training objective.

The supervised run completed all eight updates in 224 seconds inside training,
or 236 seconds including its owning process's preparation. Every update has a
saved checkpoint. The fixed final test has not run yet; training completion and
lower training loss do not establish better answers. Only the preselected eighth
checkpoint is the endpoint. We will not choose a checkpoint using test results.

## Why controller learning remains important

The larger goal is to teach a model how to inspect outside information, ask
useful smaller questions, and combine their answers. Helper classification alone
does not test that goal. A separate conversation-search pilot is being repaired
to find useful reward variation in the controller's actions before training it.
Its eight questions share one underlying conversation, so it cannot establish
transfer to new conversations. Three attempts failed during container, task or
harness setup before any model call; those failures are not model results. The
broader helper RL run is using the GPU while that separate setup is repaired.

Another small diagnostic showed why final accuracy alone is insufficient: one
correct answer came from an incorrect calculation. Our analyses distinguish a
correct answer from faithfully executing the requested computation.

The [plain-language research notebook](https://github.com/queelius/rlm-research/blob/main/research/r4/analyses/NOW.md)
links the detailed evidence, corrections, queue and limitations. Published source
and reports are not a backup of external datasets, raw traces or model weights.
