# A small, direct replication of language-model RL

## Why we are doing this

Before proposing more RL methods for recursive language models, we want to run
an established language-model RL recipe correctly. The immediate question is:
**Can we reproduce a working reward-to-weight-update-to-evaluation pipeline
using a paper's own code?** The next question is whether its short training run
improves answers to problems excluded from training.

We use [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
and its [official implementation](https://github.com/sail-sg/understand-r1-zero).
This is a reduced-scale reproduction, not a claim to reproduce its published
large-run scores. CURL was a visual-control detour, not LLM RL evidence. No new
CURL runs are part of this work.

## Fixed starting protocol

- Official source: `dfca49dd460ee7cc8e4a5a162c876a7fd6993b87`, MIT package license;
  individual training files have Apache-2.0 headers. Retrieved October 7, 2026.
- Model: Qwen/Qwen2.5-Math-1.5B, revision
  `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`.
- Train from the base model with the authors' Dr. GRPO implementation. Use
  their Qwen-Math prompt, reward checker, eight sampled answers per question,
  temperature 1, learning rate 1e-6, and 3,000-token response limit.
- One A100 40GB. Full-model updates, not LoRA. Gradient accumulation and
  alternating generation/training keep the model within memory.
- Initial operational pilot: 64 training questions, one epoch. Follow-up:
  512 questions, starting again from the base model. Random subset seed 42.
- Evaluation: 64 monitoring questions and 128 disjoint final questions from
  the bundled MATH500 set, selected with seed 142. Same prompt, token limit,
  greedy decoding, and checker before and after. No best-checkpoint selection.

The bundled training collection includes some questions from the original MATH
test split. It is therefore important to use the authors' separate evaluation
collection rather than an arbitrary MATH test download. We checked for exact
whitespace-normalized question overlaps and found none between the bundled
training collection and MATH500. This check does not establish absence from
the model's pretraining data or rule out paraphrases.

## What counts as a result

First report whether answers are generated and graded, nonzero learning signals
occur, optimizer updates are finite, saved weights change, and updated weights
are used for subsequent generation. These are operational checks, not evidence
of improved reasoning.

Then report how many of the same held-out questions are answered correctly before
and after training, including newly correct and newly incorrect answers. A small
one-seed difference is preliminary. We will not label a short null result as a
failure of the paper, or a math improvement as proof that recursive tool use works.

## Storage and current status

External runs and data: `/project/alex_phd/runs/r1-zero-replication-20261007`.
Official clone: `/project/alex_phd/research-cache/repos/understand-r1-zero`.
Isolated Python 3.10 environment: `/project/alex_phd/envs/r1-zero-dfca49d`.

The pilot is running. Its starting-model evaluation scored **19/64** on the
fixed monitoring set using the training program's checker. Actual saved answers
and scores were inspected. This is a baseline, not an RL improvement; the
separate 128-question final evaluation has not been run yet.

No training improvement has been measured yet. The first package installation
failed because FlashAttention's build script requires Torch at build time;
we installed its official prebuilt wheel matching Torch 2.6 and Python 3.10.
We also pinned fsspec 2023.6.0 for the old dataset library and recreated the
unchanged data subsets in that library's format (`data-oat/`). The first pilot
attempt failed before optimization because the environment's `ninja` executable
was not on PATH. The second attempt fixes PATH and uses unchanged author code.

The official actor independently chooses a time-based generation seed, even
when learner seed 42 is fixed. We retain that behavior for this pilot and will
record it as a reproducibility limitation, not claim fully deterministic RL.

The initial 16-question rollout batch is smaller than the paper's example
configuration. Eight sampled answers per question give 128 responses and one
accumulated optimizer update per rollout. Four rollouts mean four updates in the
pilot; the 512-question follow-on has 32 updates. This differs from the paper's
training scale and rollout/update schedule.

Follow-on order: short operational pilot, bounded 512-question training run,
fixed-checkpoint evaluation. Broader RLM-paper comparisons wait until this
straightforward reference works.
