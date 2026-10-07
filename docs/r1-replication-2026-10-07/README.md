# A small, direct replication of language-model RL

**Research resumed; current attempt began on 7 October at 15:24 UTC.** A longer fresh-base training run
is active: roughly 4,000 questions and 250 weight updates, followed by both
final MATH500 evaluations. The native loader reduces the originally intended
4,096/256 budget to 4,080 questions and 255 updates with the current small collections; see the
[accepted plan and accounting correction](longer-training-plan.md).
The earlier presentation batch is complete, but the research has not stopped.
The first larger attempt hit GPU memory exhaustion before completing a
collection; a memory-allocation retry then failed during model initialization.
The current attempt uses the smaller collection size that worked in our short runs.
No larger-run score is available yet. The [failure record](longer-attempt1-failure.json)
is retained alongside the successful short runs.
The authors' released model scored 73.2% in our evaluator, close to their
reported 74.2%. That is an evaluation check of their weights, not our own
training improvement; see the [reference-model receipt](author-reference-receipt.json).

**Early training check at 15:31 UTC:** The current run has completed learning
updates with finite, nonzero reported gradients. Of the first 32 questions,
23 had a mixture of correct and incorrect sampled answers. All 256 checked
rewards agreed with independent rescoring using the same full checker and
length-limit rule. This supports continuing the run, not a claim of improved
test performance; see the [startup check](longer-attempt3-startup-receipt.json).

Read the [five-slide PDF](../../slides/2026-10-07-r1-replication/research-update.pdf)
and [slide-by-slide guide](../../slides/2026-10-07-r1-replication/speaker-guide.md).
The [learning guide](learning-guide.pdf) explains the learning step,
the evaluation, and how to interpret the results.
Three short Dr. GRPO training runs have finished: one with chat-style input,
and two trained with the question alone. Each started from the original model
and used the same 512 training questions and 32 weight updates.

## What we have learned

We can run the authors' language-model RL recipe, verify real weight updates,
and test the saved models. The clearest finding is that **the starting input
format matters enormously**. With no additional training, the model answered
154 of 500 questions correctly in chat style, but 305 when given only the
question. Chat-style training raised the chat score to 308. That large gain
does not, by itself, establish newly learned mathematical abilities.

Training on questions alone produced scores of **314 and 306 in two separate
runs**, compared with 305 before training. The repeat had 18 newly correct
answers and 17 newly incorrect answers. We have not yet established a reliable,
substantial improvement beyond the stronger starting setup. This is a useful
limitation to discover before making claims about RL for recursive models.
See the [repeat's checked results](question-only-repeat-scoring-receipt.json)
and [training audit](question-only-repeat-training-receipt.json).

**Evaluation checks finished at 12:48 UTC:** Five unchanged-model evaluations
were repeated, and each produced exactly the same saved answers as its first
evaluation. These checks reproduced 305, 314, and 306 with question-only input,
and 154 and 308 for the original chat-style comparison. They strengthen our
confidence in this evaluation setup; they are not five more training runs.
They also do not remove earlier variation observed when the evaluation batch
changed. The [repeatability receipt](evaluation-repeatability-receipt.json)
records the files, hashes, and checks.

**Additional result at 13:48 UTC:** A fourth training run used the authors'
standard GRPO option instead of Dr. GRPO, with the same starting model, 512
questions, and 32 updates. On the question-only test it scored **318/500**,
versus **305/500** before training: 24 answers improved and 11 regressed.
That is a modest gain in one exploratory run, not a demonstrated advantage
over Dr. GRPO's two question-only results of 314 and 306. The switch changes
two parts of the learning rule together, and training randomness is not fully
matched. With chat-style input the same final model scored **169/500**, versus
154 before training (22 gains, 7 losses). The large gap between its 318
question-only and 169 chat-style scores shows that input format still matters.
Both final tests are complete and checked. See the
[checked result](grpo-final-scoring-receipt.json),
[chat-format result](grpo-chat-scoring-receipt.json),
[training audit](grpo-training-receipt.json), and
[pre-result plan](grpo-comparison-plan.md).

**Final check at 13:51 UTC:** Repeating the GRPO question-only evaluation with
unchanged weights reproduced all 500 saved answers byte-for-byte, including
318 correct. This is an evaluation check, not another training run. See the
[repeatability receipt](grpo-repeatability-receipt.json). The bounded batch is
complete; the [handoff](HANDOFF.md) records what to retain and what to try next.

The five-slide deck now has a 15:25 cutoff. It retains the central prompt-format
comparison and adds the authors' released-model check and pending longer run.
The small GRPO comparison remains in the supporting account, not a claim that
we have found the best algorithm.

Dr. GRPO comparison, verified on 7 October at 12:35 UTC:

| Model tested | Chat-style input | Question alone |
| --- | ---: | ---: |
| Starting model |154/500|305/500|
| Trained with chat style |308/500|317/500|
| Trained with questions alone |168/500|314/500|
| Repeated question-only training |166/500|306/500|

Read down a column to compare training while keeping the input format fixed.
Read across a row to compare input formats while keeping the model fixed.
Each trained row uses its prescribed final checkpoint; no best-run or
best-checkpoint selection is used.

These are small-scale reproductions and exploratory follow-ups, not a new
algorithm or a reproduction of the paper's full benchmark scores. The original
results below remain available rather than being replaced with the best run.

## Current results, 7 October

**Prompt comparison at 10:36 UTC:** The apparent gain depends strongly on how
we ask the question. The same final trained model was used throughout:

| Test input | Starting model | After chat-style RL | Change |
| --- | ---: | ---: | ---: |
| Chat-style conversation and instructions | 154/500 | 308/500 | +154 |
| Only the original question | 305/500 | 317/500 | +12 |

With the question alone, 18 answers became correct and six became incorrect.
Thus the starting model already performed much better under another input
format. Training helped under both formats, but the large chat-style gain is
not enough to establish new mathematical abilities. Nor does this prove that
training learned nothing. It is one training run, and the smaller effect needs
replication. See the [prompt-control receipt](prompt-control-scoring-receipt.json).

**Follow-up at 11:25 UTC:** Training with the question alone completed all
32 updates and saved the final model. Its final monitoring score is 39/64,
versus 38/64 before training. That small change is not the final test result.
The [training receipt](question-only-training-receipt.json) records all 4,096
attempts, including one empty response, and the verified checkpoint.

**First final result at 11:32 UTC:** With questions alone, the new model scored
**314/500**, compared with **305/500** before training. Twenty-four answers
became correct and fifteen became incorrect. This is a small positive change
from one training run, not yet a reliable improvement. The
[final-answer receipt](question-only-final-scoring-receipt.json) records the
full rescore, question checks, and separate monitoring/test counts.

**Other input format, 11:34 UTC:** That same new model scored **168/500** with
chat-style input, compared with 154 before training. It did not show the large
chat-format improvement produced by chat-style training. The complete table is:

| Test input | Starting model | Trained with chat style | Trained with questions alone |
| --- | ---: | ---: | ---: |
| Chat style |154|308|168|
| Question alone |305|317|314|

Each trained column is a separate run starting from the original model, not
another stage of the same run. All three models use the same 500 questions.
The new chat comparison has 21 gains and seven losses. Its capped-response count is
201, versus 205 for the base; question-only training did not fix that pattern.
We cannot yet identify every cause of the contrast between training formats.

**Repeatability check, 11:41 UTC:** Both first-run question-only evaluations produced
exactly the same saved answers when repeated: 305 for the starting model and
314 for the final model. These are repeats of evaluation, not independent
training runs. They do not erase the variation observed earlier when batch
composition changed. A fresh training repeat started at 11:41 with the same
settings and questions, a different learner seed, and its own data cache.
That fresh training run subsequently scored 306/500 with question-only input,
as summarized above. Unlike rerunning an evaluation, this repeated the learning
process from the original weights. Its chat-style test scored166/500, with
25 gains and13 losses against the original154. Both input formats are retained.

**Expanded chat-style check at 10:27 UTC:** On all 500 MATH500 questions, the same models
scored **154 before training and 308 afterward**. On the 308 additional
questions outside the earlier monitoring and test samples, scores rose from
92 to 184. The [expanded-test receipt](full500-scoring-receipt.json) separates
those data roles. This confirms the direction on a broader set, but it remains
one training run. The original 128-question result below is retained unchanged.
All 500 were excluded from weight training, but 64 were used to monitor progress.

On the same **128 questions excluded from training and progress checks**, the
model improved from **41 correct before training to 80 afterward**. Forty-four
answers became correct and five became incorrect. Both sides used identical
prompts, response limits, and the authors' full answer checker. We used the
prescribed final checkpoint, not a checkpoint selected for its score.

This is a substantial improvement in this one short run. It is not by itself
evidence of newly learned mathematical methods. For 36 of the 44 gains, the
original response had hit the response-length limit. For example, “Write 3/20
as a decimal” changed from repetitive noncompletion to an explanation ending
in 0.15. The [fixed-test receipt](fixed128-scoring-receipt.json) records paired
counts, saved-answer hashes, and these behavior measurements.

The larger run completed all 32 weight updates and saved its final model. On
the 64 questions used to monitor training, the final score increased from
**20 to 36 correct** under the authors' full answer checker. Eighteen answers
became correct and two became incorrect. An earlier evaluation immediately
after the last update scored 39; we retain the final evaluation, not the higher
score. There was no additional optimizer update between these evaluations.

The monitoring result above is separate from the 128-question test. The completed
prompt controls support the paper's warning that input format can conceal
existing ability and inflate the apparent size of a training improvement.
We have not isolated the mechanism behind each changed answer.

The [completed-training receipt](subset512-training-receipt.json) records the
final checkpoint hash, monitoring scores, and audit of all 4,096 training
responses. The slide deck and learning guide include the prompt comparison;
the original separate test remains in the supporting evidence.

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

The pilot completed successfully. With the paper's standalone checker applied
to both sets of saved answers, the fixed monitoring score was **20/64 before
training and 22/64 at the final evaluation**. Three answers became correct and
one became incorrect. Four nonzero, finite-gradient optimizer updates completed;
sampled saved weight tensors changed, and the updated model was broadcast to
the generator after each update.

This is an operational success and a small, uncertain score increase, not a
confirmed improvement. The first post-training check gave 23/64; a second check
of the same weights gave 22/64, with eight answer texts changing. We retain the
final result rather than selecting the better score. Greedy decoding did not
make this GPU run exactly repeatable. The training program's faster checker
scored 19/64 before and 21/64 at the final evaluation; those are not mixed with
the standalone-checker counts above.

The 512-question run started at 09:19 UTC and finished by 10:09 UTC. The separate
128-question final evaluation scored 41 before and 80 after. Training outputs are in
`/home/atowell/research-runs/r1-zero-replication-20261007/subset512-attempt1`:
the project allocation was almost full, whereas the home allocation had room.
The completed pilot was copied to the same home research store under
`archive/pilot-attempt2`, verified byte-for-byte and hashed. Its original path
remains a symlink, so earlier evidence links still work. Only the verified
duplicate was removed from project storage; all pilot artifacts are retained.

An answer-level audit of the pilot found two recoveries from repetitive
noncompletion, one corrected final answer, and one regression into repetition.
For example, a fuel-saving problem went from repeating the question to finishing
the correct calculation, 12,000/15 - 12,000/48 = 550 gallons. This does not show
that the model learned a new mathematical method. See the
[scoring receipt](pilot-scoring-receipt.json) for exact saved-answer hashes.

The first package installation
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

## Extension before the presentation deadline

The user requests an expanded evaluation if the initial result is ready early.
We will retain the original 128-question comparison and then evaluate the same
base and final models on all 500 questions in the authors' MATH500 collection.
This extension was chosen before the 512-question run's final result was known.
Both sides will be generated afresh with identical settings. Scores from
different batches will not be spliced together.

The full set includes the 64 monitoring questions, the original 128 final
questions, and 308 additional questions. We will identify these roles in the
analysis. All were excluded from weight training, but the monitoring questions
were repeatedly inspected. More evaluation questions improve coverage; they
do not substitute for independent training runs. The completed prompt control
changed the next priority, as documented below.

The presentation remains limited to five slides and is due by 13:00 UTC;
extra experiments must not delay a usable published version.

### Next experiment: train with the question alone (10:35 UTC)

The stronger untrained prompt changes the most informative next comparison.
We launched `run_subset_raw_attempt2.sh`, using the authors' existing `--prompt_template no`
option. It starts again from the original base weights, uses the same 512
questions and 32 updates, and has a 90-minute cap. All other scientific training
settings stay fixed. Checkpoints and optimizer state are saved every eight
updates, retaining the latest, plus the prescribed final save. Outputs live at
`/home/atowell/research-runs/r1-zero-replication-20261007/subset512-raw-attempt2`.

The first attempt was stopped after its saved inputs revealed stale chat-style
questions despite the question-only setting. The dataset library had reused
an earlier transformed cache. A focused CPU reproduction confirmed the cause:
an unhashable transform fell back to a random cache key that repeated under
the same seed. The second attempt uses a separate dataset directory with the
same questions and references, verified identical, and no old transformed cache.
We preserved the failed attempt and changed no author algorithm code. This
is why we inspect real training inputs rather than trust configuration alone.

The primary comparison will be the final model versus the original 305/500
question-only baseline. We will also test the final model with the chat-style
prompt. We will report both, regardless of which is better. This follow-up is
exploratory because the earlier evaluation motivated it; it is not a new,
untouched confirmatory test. If the stronger-prompt gain is promising, repeat
that training condition independently. If it is flat or worse, report that
limitation at this training budget rather than claiming the paper failed.

### Independent chat-style repeat prepared at 10:27 UTC, now second priority

This prepared, not yet launched, run would restart from the same base weights on the same 512
training questions, with learner seed 43 instead of 42. The authors' actor still
uses a time-based generation seed. This is a fresh training realization, not a
fully deterministic seed-controlled comparison. All other scientific settings
stay fixed: 32 updates, eight attempts per question, and the same prompt/reward.
It has a 90-minute cap and saves every eight updates plus the prescribed final
checkpoint, under the separate home-store directory `subset512-repeat43`.

The question is whether the observed gain survives another short training run.
The planned final checkpoint will be evaluated with the same 128/500-question
protocol, not chosen from monitoring scores. Do not replace or average away
the first run. The raw-question experiment now takes priority. Reassess this
repeat after its results; do not delay the usable deck or compete for the GPU.

## Automated continuation

The lightweight observer in `experiments/r1_replication_20261007/review_monitor.py`
queues reviews into this existing session when evaluations finish or the active
training process exits, with a periodic fallback. It does not launch GPU jobs
or interpret results itself. Its current configuration and status are in the
external experiment store under `reviews/`. It reuses the earlier campaign's
queue engine at the explicit `ORIGIN` path in the script; this is a cluster-local
dependency, not a standalone laptop service. The earlier CURL observer remains
stopped. Dispatch requires more than 12% account allowance to preserve 10%.
