# What the latest controls changed

Evidence cutoff: September 12, 2026, 18:23 UTC. This follows the
[earlier helper-training report](2026-09-12-broader-rl-learning-signal.md).
These are exploratory findings, not a claim that we have solved RLM training.

## The large training-data effect did not survive the next panel

We compared eight RL updates using different articles each time with eight
updates repeating the same 128 articles. Both made the same number of sampled
label decisions, and every training block had the same category balance.

| Model | Earlier panel: correct / 512 | Fresh official-test panel: correct / 512 |
|---|---:|---:|
| Starting helper | 422 | 422 |
| Supervised training | 427 | 426 |
| RL with varied articles | 437 | 427 |
| RL with varied articles, second seed | 436 | 429 |
| RL repeating 128 articles | 417 | 425 |

The varied-data advantage over repetition shrank from 20 answers to two in the
first-seed comparison. Repetition even changed from below the starting model to
three answers above it. All five fixed models returned every answer. We did not
choose checkpoints after seeing these scores.

This weakens the earlier interpretation that data variety explained a large,
reliable gain. The initial effect depends strongly on which examples we test.
The category mix does not explain it, but differences in example difficulty,
the particular repeated block, and learned category preferences remain possible.
We should not invest in another long run of this recipe without a sharper test.

A different task gives the same caution. On 224 encyclopedia descriptions, the
starting helper got 209 right, supervised training 208, and the two varied-data
RL models 209 and 210. That is little evidence of useful cross-task improvement.

## Knowing the input format is not the same as understanding the task

The controller had to retrieve a particular assistant reply from a conversation.
It often treated the message list as a dictionary with imagined field names.
We added a short description of the real structure, without revealing message
content, the answer, or a retrieval procedure.

Across eight conversations with two trials per condition, Python traceback
observations fell from 18 to zero. Exact answers did not improve: one of 16
without the description, zero of 16 with it. Every outcome was available.

For example, given this simplified conversation:

```text
User: Write a social media post about physics.
Assistant: Here is a post about gravity ...
```

The model might return “Write a social media post about physics” when asked
for the post. It can now read the message list, but still chooses the request
instead of the reply. Removing Python errors alone does not teach that distinction.

## Procedural training needs more than four updates

We supplied 32 demonstrations that find the matching user request and return
the following assistant reply. Each demonstration was independently checked
against the public conversation and the correct answer. The model received
the same initial prompts during training and evaluation.

Four supervised updates lowered the training loss, but the model still invented
dictionary fields rather than following the demonstrated retrieval procedure.
Its training-panel readout returned no exact answers: 28 outcomes were available
and four were unavailable. Three requests exceeded the model's context limit
after large debugging outputs; one encountered an IPython broker timeout.
We did not replace missing results with wrong answers or run the conditional
held-out evaluation after this failed learning check.

The training itself took 2.7 minutes. Its readout took another seven minutes.
Lower loss is not yet a working procedure. The next planned test continues from
the saved weights, optimizer and random state to a fixed total of 32 updates,
keeping the examples and instructions unchanged. That isolates training amount
before changing what the model is taught.

## RL now makes measurable updates; answer improvement is still being tested

The first controller RL attempt made no update. Numerical differences between
the generation and training engines made its whole-trajectory probability
correction too uneven for the predeclared gate.

A new experiment uses a standard token-level correction with an explicit bias
limitation. It starts two branches from identical weights and random state,
uses the same saved gradient, and changes only the learning rate. One update
moves the adapter ten times farther than the other, as intended.

On the saved training attempts, the larger update increases the likelihood of
both rewarded trajectories and reduces that of five of six penalized ones.
This shows the update changes relevant behavior probabilities. It does not
show that new answers improve. Only two training question groups supplied a
learning contrast, and their near-successes relied on broadly printing the
conversation. The current three-arm evaluation keeps the starting model and
both fixed doses visible on the same 16 separate conversations.

## The next decomposition test asks what information to request

Our first test allowed zero, one, or two levels of delegation on questions
requiring linked facts. Across 112 physical model calls, none called a helper.
The depth conditions therefore did not test the benefit of actual recursion.
Several programs also searched dictionary keys instead of paragraph text.

The next screen supplies the paragraph text directly to two helpers, each
receiving half of it. All report-based conditions reuse their first reports.
We compare stopping there, asking generally for more information, and asking
focused follow-up questions. A fourth condition reads the full original source.
This isolates information targeting from the earlier Python-access failures.
It is a fixed experimental call graph, not yet learned delegation or a new
architecture. Its first launch failed before asking any question; a narrow
service-startup repair is being prepared while the RL evaluation runs.

## What looks most promising now

The useful research direction is to separate three abilities: reading the
environment correctly, requesting the right information, and faithfully using
that information in the answer. Our tests already show that improving one
component need not improve the complete RLM. Earlier, a better helper corrected
five more of 128 article labels, yet the full system still answered only three
of 16 questions correctly with either helper.

The strongest next evidence would be a targeted intervention that fixes one
identified failure, improves complete answers, and repeats on new examples or
another task. We do not yet have that complete result. The larger initial
classification gain, by itself, is not the strongest publication claim.

## Evidence and preservation

The [research notebook](https://github.com/queelius/rlm-research) contains the
methods, source hashes, limitations and resume pointers. Key external folders
under `/project/alex_phd/runs/rlm-research-r4/analyses/` are:

- `helper-agnews-repeat128-official-transfer-2026-09-12`: all five fixed endpoints and paired raw-response audit.
- `helper-agnews-training-breadth-mechanism-2026-09-12`: category balance, first-step pairing and training contrasts.
- `mrcr-structural-preview-findings-2026-09-12`: access versus retrieval comparison.
- `openai-mrcr-procedural-sft-first-readout-2026-09-12`: teacher checks, generated-program analysis and missingness.
- `openai-mrcr-token-tis-movement-2026-09-12`: two-dose likelihood movement, not an answer-quality result.
- `musique-depth-findings-2026-09-12`: the depth manipulation was not exercised.

An additional released-base reference returned 423/512 news labels and 204/224
encyclopedia labels. Its original qualification failed because it required an
optional log warning that eager execution did not emit. A separate CPU audit
supports the intended configuration, but no historical in-worker dispatch
receipt exists. We retain it only as an explicitly qualified exploratory
reference; the original failure and unavailable primary metric are unchanged.
It does not strengthen the claim of broad RL improvement.

GitHub preserves the selected source and reports, not model weights or the full
external research store. Every training checkpoint remains separately stored
with its optimizer and random-state metadata.
