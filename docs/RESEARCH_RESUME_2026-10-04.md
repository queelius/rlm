---
date: 2026-10-04
status: deferred_for_user_priority
reviewed_research_cutoff: 2026-09-29T18:57:05Z
next_priority: curl_single_gpu_replication
replication_paper_selected: CURL_ICML_2020
---

# RLM research checkpoint and replication options

The user has a higher-priority assignment and asked to preserve the RLM work
for later resumption. The new discussion concerns reproducing published results
to improve our practice of supervised fine-tuning, reinforcement learning, and
evaluation. The user subsequently approved proceeding on one A100, with CURL
(ICML 2020) selected as the first RL plus self-supervised learning target.
Update at 17:13 UTC: execution access is restored. The real GPU pilot completed,
and a fresh 100k-environment-step reference run is active. The earlier device
and shell-network access restrictions were specific to the managed context.

The new protocol and live handoff are in
`experiments/curl_replication_20261004/README.md`. The explanatory LaTeX document
is `docs/curl-replication-2026-10-04/learning-guide.tex`. This is a reference
learning exercise, not a claim that CURL reproduces LLM SFT or our old RLM runs.

## Where the saved work lives

The research worktree is
`/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921`, on
`research/selective-delegation-20260921`. Its current HEAD is `eb53938`, with
uncommitted later analysis and slide changes. Preserve those changes.

Start with these paths inside that worktree:

- `docs/research-audit-2026-09-29/README.md`: retrospective findings and limits.
- `docs/research-audit-2026-09-29/publication-assessment.md`: the teaching-history
  paper candidate and proposed stronger comparisons.
- `docs/research-audit-2026-09-29/current-rl.md`: includes the later September 29
  18:57 addendum, which supersedes its earlier table.
- `slides/2026-09-30-advisor-meeting/READING_GUIDE.md`: self-contained explanation.
- `slides/2026-09-30-advisor-meeting/evidence.json`: source identities and cutoffs.
- `slides/2026-09-30-advisor-meeting/research-update.pdf`: ten-slide audience PDF.

The PDF SHA256 is
`0e710f910be4b084eeb7e5d1f6d5798ed80a04684fc07d570e74758467ed35ca`.
The advisor package was published by the user. Remote main was observed at
`6acbb87a6e24944ec33e62eb6a16a88cb2831259` on October 4. Read the
[advisor package](https://github.com/queelius/rlm/tree/main/slides/2026-09-30-advisor-meeting)
and [historical audit](https://github.com/queelius/rlm/tree/main/docs/research-audit-2026-09-29).
This new checkpoint is saved locally; its presence is not a claim of a new push.

Native outputs and model checkpoints remain under
`/project/alex_phd/runs/rlm-research-r4`, especially
`sidecars/selective-delegation-20260921`. Git publication does not back up those
weights or raw outputs. No artifacts have been deleted or relocated.

## Findings and interpretations to preserve

The strongest focused lead is repairing training examples. Reordering existing
recipe lookups and rebuilding observations before the same taught actions
improved crafting success from 1 to 14, 2 to 13, and 3 to 17 out of 32 attempts.
The three comparisons contain 16 distinct goals in total, not 96 independent
problems. Input lengths and computing work changed. More training partly
rescues the original examples, and a teacher that discovers recipes in order
performs similarly to the repair. This is exploratory evidence, not an
established general or novel method.

Separately, code filling ingredient arguments from already-observed recipes
increased success from 323 to 381 of 768 attempts over 48 goals, with unchanged
weights within each pair. It did not choose items, quantities, or the plan.
The user correctly challenged whether code could do that too. A programmed
recipe planner is a strong baseline, and programmed teachers already solve
selected training tasks. Do not claim that an RLM is necessary or generalizes
better than code. New item names alone do not distinguish it from a generic
recipe planner. Crafting is currently a controlled learning environment.

Recent raw-action RL scores were 4, 4, 5, 4 out of 16 across the SFT-trained
starting point and three fixed updates. Assisted scores were 5, 4, 4, 4.
Extra-SFT controls gave 4 and 5 respectively; one SFT update is not an
equal-dose control for three RL updates. These are reused diagnostic goals
from official TRAIN with shared components, not an untouched official test.
Updates were verified, but useful persistent behavioral improvement was not.
Do not use diagnostic B for training choices or favorable checkpoint selection.

Do not describe all decomposition as learned. Fixed document chunks and
candidate-by-candidate helper calls were programmed. MuSiQue question lists
were learned, but the complete system did not reliably beat direct answering
and was not an adaptive recursive tree. Most crafting studies trained one model
to choose successive tool actions.

The broader campaign found no consistent benefit from fixed splitting and
summarization across five datasets. Executing model-chosen arithmetic in code
improved strict FinQA numerical matches from 51 to 108 of 508 for one model,
and 52 to 102 for another. No extra training was used. Prior exposure, rounding,
and reference errors limit this score; this is not a new arithmetic method.

## Ideas clarified in the final discussion

A demonstration can solve a task yet omit information a learner needs. In our
illustration, the model is taught to ask about a frame before any observation
has introduced one. Moving the lantern recipe lookup earlier gives that action
context. Easier name copying remains an alternative to better planning as an
explanation of the gain.

The final slide combines three questions: does the benefit carry over to new
problems, does it beat stronger alternatives, and why does it help? Deliberately
new dependency patterns can test compositional or out-of-distribution
generalization. Held-out examples are not automatically OOD. Another model tests
portability of the training method; another task is needed for a cross-task claim.

Slide 4 was rewritten and visually verified. The later discussion of the
code-only baseline and the meaning of generalization was not incorporated into
slides 6 and 10. Keep those points in mind for a future deck revision.

## Operational state and first steps on resumption

On October 4, one A100 40 GB was visible with 1 MiB allocated and no reported
compute processes; utilization was unavailable. No experiment was launched
because the user reprioritized the work and requested discussion. Old queue
text is not fresh launch authority.

The external `SESSION_CHECKPOINT.md` and `RESEARCH_QUEUE.md` still begin with
September 29 08:57 records, earlier than the latest reviewed findings. The
observer's last recorded status is October 1 09:40:26 UTC, with 138 events and
an unacknowledged review. Event counts are not analyzed results. Later outputs
may exist beyond the deck cutoff; they were not scientifically reviewed here.
The old allocation deadline has passed.

Before choosing new RLM work, reconcile completed native results after the
cutoff, especially later RL controls, Phi teaching repair, and whole-task
helper trials. Do not assume an old proposal remains unrun. Preserve sealed
sources, checkpoint identities, and diagnostic-data restrictions. Do not revive
an old queue or observer without checking current ownership and allocation.

During the earlier discussion, the managed context made Git metadata and the
external run store read-only. That restriction was resolved on October 4 before
the CURL pilot. This retrospective section does not claim that old RLM artifacts
were changed or an old active owner stopped. No old process was killed.

## Replication approach proposed for discussion

Start with authors' code, then independently implement one scientific component.
This separates algorithmic misunderstandings from implementation bugs and
hardware changes. Read the paper and write down the intended comparison before
treating the code as authoritative; disagreements should be documented.

1. Choose one reported comparison with accessible code, model, data, scorer,
   and an affordable complete run. Pin the paper version and source revision.
2. Reproduce evaluation of released starting and trained checkpoints first.
   Check prompts, extraction, decoding, stopping, and denominators.
3. Run a faithful training comparison. A smoke test is not a reproduction;
   changing model, data, LoRA versus full tuning, or budget must be labeled.
4. Reimplement the central objective independently. Compare the same saved
   batch's rewards, advantages, token masks, loss, gradients, and updates before
   comparing long-run scores. Similar final scores alone do not prove agreement.
5. Add one ablation or extension only after the reference behavior is understood.
   Keep development and final evaluation separate and measure seed variation.

The user suggests a paper with a single-A100 experiment, potentially in
addition to a larger reproduction. Prefer a verified single-device target
before waiting for two GPUs. Two devices can run independent seeds or controls,
or separate training from generation. Two 40 GB cards do not automatically act
as one 80 GB device. Actual memory and runtime must be checked from the chosen
configuration, not inferred from the model size alone.

## Candidate papers and their limits

- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) and
  [official code](https://github.com/sail-sg/understand-r1-zero) study prompt and
  objective effects, including GRPO versus Dr. GRPO. Good LLM-specific learning
  target, but even the documented 1.5B command is tested on eight A100 40 GB
  devices. A one-GPU adaptation is not an exact published reproduction.
- [Tulu 3](https://arxiv.org/abs/2411.15124) is an end-to-end SFT and RLVR
  reference. Its [reproduction documentation](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)
  identifies removed historical scripts and changed verifiers. Pin paper-era
  code; do not assume latest main reproduces the paper. Full scale is large.
- [FastTD3](https://arxiv.org/abs/2505.22642) and
  [official code](https://github.com/younggyoseo/FastTD3) offer a genuine
  single-A100 RL target in simulated humanoid control. The paper reports runs
  on one A100 and 16 CPU cores, with several tasks solved in under three hours.
  This teaches RL fundamentals, not LLM SFT. Exact 40 GB fit is not verified here;
  the GPU-resident replay buffer matters.
- [CURL](https://proceedings.mlr.press/v119/laskin20a.html) and
  [official code](https://github.com/MishaLaskin/curl) combine contrastive
  self-supervised representation learning with RL from images. The official
  single-GPU cartpole example is described as about an hour, depending on GPU.
  It is not an A100-specific timing claim. This is attractive if SSL means
  self-supervised learning; the older environment stack needs checking.
- [Q-RAG](https://arxiv.org/abs/2511.07328) and
  [official code](https://github.com/griver/Q-RAG) train an embedding-based
  retrieval policy while freezing the answering LLM. It is close to our question
  of which evidence to inspect next. Authors report one A100 **80 GB**, within
  12 hours per model. Our visible device is 40 GB. Training the retriever also
  does not reproduce LLM weight-training mechanics; evaluation cost is separate.

SSL is still ambiguous. If self-training on generated explanations is intended,
[STaR](https://arxiv.org/abs/2203.14465) is conceptually useful, but uses known
answers and an older TPU-oriented [official implementation](https://github.com/ezelikman/STaR).
It is not unsupervised learning without answer information. If the user meant
SFT, keep the replication focused on language-model post-training instead.

All sources were inspected on October 4. No repository was cloned or executed
for the initial discussion. These were candidates at that point; CURL was
subsequently selected. A successful
reproduction would provide a trusted reference, but would not retroactively
validate our old RLM experiments without checking the relevant implementation.
