# Preliminary RLM training and experiment analysis

This document is the detailed companion to `rlm-preliminary-results.tex`. The slides are meant
to communicate the big picture in a short meeting. This document preserves the fuller record:
what was tested, what worked, what failed, what remains uncertain, and why the experiments took
substantially longer than any one training job.

The experiments are preliminary and exploratory. They establish that supervised fine-tuning can
teach an 8-billion-parameter model several useful RLM work patterns. They do not yet establish
that a trained RLM can reliably solve unfamiliar, open-ended problems by recursive decomposition.

## Executive summary

The work produced five main findings.

1. **The complete training loop works.** We can generate checked examples of RLM behavior, train
   a LoRA adapter, save intermediate versions, load an exact version into vLLM, run it inside the
   RLM, and verify both its answer and its record of work.
2. **A model can learn narrow RLM routines very reliably.** The first causally valid experiment
   solved 29 of 30 held-back tasks, compared with 11 of 30 for the model answering directly.
3. **Training-data design is part of the scientific question.** One broader experiment, V2A,
   accidentally gave the same visible controller state different hidden task-specific targets.
   Its apparently positive sealed score reflected shortcut learning, not a task-conditioned
   policy. The experiment is valuable negative evidence and motivated stronger causal audits.
4. **A causally valid fixed dispatcher can transfer perfectly within its intended schema.** V2B
   solved 192 of 192 sealed tasks. This is a real result, but the model was reproducing one fixed
   six-branch Python program. It is evidence that SFT can install a dependable RLM mechanism, not
   evidence of broad reasoning or learned decomposition.
5. **Broader multi-step behavior is promising but incomplete.** In the V3A exploratory
   development pilot, the trained controller solved all 24 tasks in four local workflow families:
   one-turn dispatch, inspect-then-compute, persistent-state verification, and repair after a
   visible fault. It solved none of the 12 tasks in the two model-delegation families. Thus local
   Python-mediated work was learned more reliably than handing work to another model.

No RLVR optimizer update occurred. The RLVR probes were intentionally designed to stop when the
rollouts were invalid or when every rollout received the same reward. Those stops taught us that
useful reward-based training requires both a robust input contract and tasks that produce valid,
policy-dependent reward variation.

V3B subsequently completed four matched LoRA training runs testing whether varied but equivalent
input layouts improve robustness. Its held-out behavioral evaluation has not yet run, so its
effectiveness is unknown. V3C is a reviewed experimental design, not a completed training result.

## The research arc

| Stage | Question | Status | Most defensible conclusion |
| --- | --- | --- | --- |
| Initial smoke | Can Qwen3-8B run through the local RLM and vLLM stack? | Completed | The untrained model and harness connected, but the controller did not yet finish the test tasks. |
| V1 narrow ABI SFT | Can SFT teach a fixed two-turn RLM routine? | Completed and audited | Yes, on a narrow shared-schema task family: 29/30 held back. |
| V2A broader symbolic SFT | Does a more varied curriculum learn counterfactual predicates? | Completed and sealed | The measured score rose, but hidden target information caused shortcut learning. This is negative evidence. |
| V2B causal dispatcher | Can SFT install one dispatcher whose action is valid before the task is inspected? | Completed and sealed | Yes: 192/192 within the explicit payload schema, using one repeated program. |
| V3A broader causal SFT | Can SFT teach several multi-step controller workflows and subcall patterns? | Exploratory development pilot completed | Four local workflows transferred; the two delegation workflows did not. Result: 24/36, not sealed or replicated. |
| RLVR V0--V6 | Can we capture valid stochastic RLM trajectories and perform one justified reward-based update? | Diagnostic probes completed | Token/log-probability capture worked, but no probe supplied both valid trajectories and useful reward variance. No update occurred. |
| V3B input-contract study | Does training on six equivalent request layouts improve robustness over one layout? | Four paired training runs completed | Training succeeded; efficacy is withheld until held-out paired evaluation. |
| V3C compact-target study | Can shorter post-inspection targets reduce cost without losing correctness? | Designed and CPU-audited | Not yet trained or evaluated. |
| `rlm-bootstrap` self-training | Does practice inside the RLM teach more than ordinary direct practice? | Designed | Future comparison; no training or evaluation result yet. |

## Common model and training architecture

The main experiments used the pinned `Qwen/Qwen3-8B` base revision
`b968826d9c46dd6066d109eabc6255188de91218`. Keeping the model fixed allowed the experiments to
change the curriculum and controller behavior without also changing the model family, tokenizer,
chat template, or serving behavior.

Training used LoRA rather than changing every base-model weight. The base model remained frozen,
while approximately 43.6 million LoRA parameters were trained in the attention and feed-forward
linear layers. The resulting adapter weights are about 175 MB. This is less than one percent of an
8-billion-parameter model.

During SFT, one process loaded the frozen base and mutable LoRA parameters. Gradients and optimizer
state updated only the LoRA parameters. Checkpoints contained the adapter, optimizer, scheduler,
random-number state, tokenizer information, and content identities. After training, the adapter
was published as an immutable artifact. A separate vLLM process then loaded the same base model
plus a named adapter for evaluation. The serving model was never silently updated in place.

This separation also anticipates RLVR. A future trainer will own mutable policy version `k+1`,
while a rollout server uses an immutable copy of policy version `k`. A rollout batch must retain
the generated token IDs, action masks, old-policy log probabilities, sampling settings, policy
identity, RLM trace, and verifier reward. Only after the batch is complete should the trainer
update the adapter and publish a new version.

## Initial untrained smoke

The first Qwen3-8B probe compared direct answers with an inspect-then-act RLM on three small tasks.
The direct answers were not accepted as correct, and all three RLM conditions exceeded the
four-turn limit without submitting a final answer. This was not a training result. It showed that
the model, server, proxy, executor, and trace machinery could be exercised end to end, while also
showing that the untrained model did not naturally follow the controller interface reliably.

Primary artifact:

- `/project/alex_phd/runs/rlm-v2-spikes/inspect-then-act-qwen3/summary-qwen3-8b.json`

## V1: narrow RLM-interface learning

### Question and data

V1 asked whether SFT could teach a narrow two-turn controller routine. It used 80 symbolic
training tasks, exported as 160 controller-turn examples, and 30 held-back test tasks. The test
tasks had new surface instances, but the operation families, program patterns, and atomic values
overlapped with training. It was intentionally a mechanism test rather than a broad
generalization test.

### Training and result

Qwen3-8B received a rank-16, alpha-32 all-linear LoRA adapter. Training ran for 120 optimizer
steps. The retained training manifest reports 269.4 seconds, or about 4.5 minutes.

The predeclared endpoint produced:

| Condition | Held-back tasks solved |
| --- | ---: |
| Model answering directly | 11/30 (36.7%) |
| Untrained model inside the RLM | 0/30 |
| Trained RLM controller | 29/30 (96.7%) |

The paired improvement over direct answering was 60 percentage points. The preregistered paired
bootstrap 95% interval was 43.3 to 76.7 points. This comparison was not equal-compute: direct
answering used one model call per task, while the trained controller used two.

### What it taught us

V1 established that the model could learn the RLM interface and a reliable inspect-then-compute
routine. It also exposed a specific overfitting problem. Starting at step 40, the model repeatedly
dropped an `even` predicate in one program and summed every square. That single stable mistake
motivated later counterfactual pairs such as even versus odd, divisible versus not divisible, and
strict versus non-strict thresholds.

Primary sources:

- `/project/alex_phd/runs/rlm-v2-spikes/abi-curriculum-sft/RESULTS.md`
- `/project/alex_phd/runs/rlm-v2-spikes/abi-curriculum-sft/curriculum/manifest.json`
- `/project/alex_phd/runs/rlm-v2-spikes/abi-curriculum-sft/adapter-qwen3-abi-v1/manifest.json`

## V2A: a broader experiment that revealed shortcut learning

### Intended question and data

V2A broadened the task inventory to six paired operations:

- sum the squares of even or odd values;
- count values divisible or not divisible by a given divisor; and
- sum values strictly above or at least as large as a threshold.

There were 160 training groups, 32 development groups, and 32 sealed-test groups, with six tasks
per group. This produced 960 training tasks, 192 development tasks, 192 sealed tasks, and 1,280
supervised controller turns. Development and sealed data used new value ranges, divisors,
thresholds, and renderers.

### Training and measured result

The rank-16 LoRA run completed 320 optimizer steps in 691.8 seconds, or about 11.5 minutes. Four
checkpoints were evaluated on a preregistered 48-task development screen. The final checkpoint was
selected and then evaluated on the remaining development tasks and the sealed set.

On the sealed set:

| Condition | Tasks solved |
| --- | ---: |
| Model answering directly | 12/192 (6.25%) |
| Untrained model inside the RLM | 0/192 |
| Selected V2A controller | 43/192 (22.4%) |

The numerical comparison is reproducible, and the sealed evaluation order and scoring were valid.
The positive interpretation was not.

### The causal problem

At the first controller turn, the model did not see the task contents. The task existed only in a
runtime Python variable named `request`. Nevertheless, the generator used private task information
to choose one of six operation-specific target programs. The same controller-visible prompt was
therefore paired with incompatible targets.

The model could not infer which target was appropriate from what it saw. Different checkpoints
instead learned different fixed favorite programs. On the selection screen, step 80 succeeded
only on strict-threshold sums, step 160 only on even-square sums, and step 240 only on divisible
counts. On the sealed set, all 43 successes were concentrated in three of six operations; the
other three scored zero.

This is not a sealed-data leak or a scoring mistake. It is a training-label defect: the desired
action was not causally determined by the model's visible state. V2A is therefore strong negative
evidence about shortcut learning, not evidence that the model learned a broad task-conditioned
controller.

### What changed afterward

Later admission checks require one target policy for each controller-visible state. A specialized
action is allowed only after an observation has exposed the relevant task. Audits preserve
predicate constants and reject hidden operation or trajectory choices.

Primary sources:

- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft/EXECUTION_PLAN.md`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft/SESSION_CHECKPOINT.md`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft/v2a-curriculum-160g/curriculum/manifest.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft/evaluation-v2a/screen/summary.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft/evaluation-v2a/confirmatory-development/summary.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft/evaluation-v2a/sealed-test/summary.json`

## V2B: a causally valid fixed dispatcher

### Corrective design

V2B asked a narrower but cleaner question: can SFT install one controller action that is valid
before any hidden task choice is known?

Every one of the 576 training examples used the same Python dispatcher. At runtime, that program
read `request["input"]`, parsed an explicit `PAYLOAD_JSON=` record, branched over the same six
operations, computed the result, and submitted one final JSON answer. It made no model subcalls.

The training, development, and sealed sets used 96, 32, and 32 whole groups respectively. Their
values and surface renderers were split-disjoint. The task payload always explicitly named the
operation.

### Training and result

Training ran for one pass, 144 optimizer steps, and 332.3 seconds, or about 5.5 minutes. All four
saved checkpoints solved all 48 tasks on the development screen. The preregistered tiebreak chose
the earliest checkpoint, step 36. It then produced:

| Phase or condition | Tasks solved |
| --- | ---: |
| Remaining development tasks | 144/144 |
| Sealed model answering directly | 9/192 (4.7%) |
| Sealed untrained RLM | 0/192 |
| Sealed selected V2B controller | 192/192 (100%) |

### What 192/192 means

This is a real sealed result. The model reliably emitted a causally valid dispatcher on new
values and renderers, and that dispatcher correctly handled every explicit operation.

It is also deliberately narrow. All 576 targets were byte-identical. The training set contained
only 13 tokenized sequence variants and 563 duplicate sequences. Evaluation tested whether the
model would reproduce the learned dispatcher, not whether it would invent a new strategy or
choose among decompositions. V2B did not train repair, natural-language transfer, recursive calls,
or model delegation.

The right conclusion is: **SFT can install a dependable RLM mechanism.** The wrong conclusion is:
**the model learned general RLM reasoning.**

Primary sources:

- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v2b/EXECUTION_PLAN.md`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v2b/v2b-curriculum-96g/curriculum/manifest.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v2b/v2b-training.log`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v2b/evaluation-v2b-development-screen/screen/summary.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v2b/evaluation-v2b-development-screen/confirmatory-development/summary.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v2b/evaluation-v2b-development-screen/sealed-test/summary.json`

## V3A: broader multi-step controller behavior

### Design

V3A expanded from six arithmetic predicates to 24 operations across lists, strings, records, and
nested JSON. It trained six workflow families:

1. emit one generic 24-operation dispatcher;
2. inspect the request and then compute;
3. inspect, compute, store an audit value, and verify persistent state;
4. inspect and delegate to a public model;
5. inspect and use single or batched model subcalls; and
6. repair after an injected visible fault.

The training curriculum contained 1,656 episodes and 2,592 supervised controller turns, balanced
at 432 examples per family. A fresh Qwen3-8B LoRA adapter trained for one 648-step pass. The saved
checkpoint timestamps span approximately 38 minutes; unlike the other principal SFT runs, the
retained final manifest does not contain a trainer-reported runtime.

### Exploratory development result

The completed evaluation was a smaller development pilot, not the planned full evaluation. It
used six tasks per family for checkpoint selection and six disjoint tasks per family for
confirmation. The selected step-648 checkpoint solved 24 of 36 confirmatory tasks:

| Workflow family | Tasks solved |
| --- | ---: |
| One-turn dispatcher | 6/6 |
| Inspect, then compute | 6/6 |
| Persistent-state verification | 6/6 |
| Repair after a visible fault | 6/6 |
| Delegate to another model | 0/6 |
| Use single or batched subcalls | 0/6 |

The successful local families demonstrated multi-step work inside the RLM: inspecting hidden
input, computing after an observation, carrying state forward, verifying it, and recovering from
a controlled visible fault. This is a form of local decomposition into controller steps.

It did not demonstrate reliable recursive decomposition through model calls. The delegation
family followed the expected trace shape but returned wrong public-model answers. The batch family
attempted subcalls, but generated long or incomplete model output and exhausted its turn budget
before submitting a valid answer.

### Limits

The pilot is labelled exploratory and development-only. It used one training seed, only 36
confirmatory tasks, and no sealed evaluation. The tasks had new latent values but the same
operation algebra and canonical payload format. The result is a promising systems signal, not a
replicated claim about broad generalization.

Primary sources:

- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a/EXECUTION_PLAN.md`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a-curriculum-production-20260825T033650Z/curriculum/manifest.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a-adapter-final-20260825T041500Z/manifest.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a-development-pilot-20260825T050000Z/screen/summary.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a-development-pilot-20260825T050000Z/confirm/summary.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a-posthoc-audit-20260825T080000Z/pilot-posthoc-audit-2c367338a11ac7ef61d91e58c06bdc005f92386367c87dca90be6ccde2ba1fac.json`

## RLVR readiness probes: valuable stops, but no training update

RLVR requires more than a correct final answer. For a valid policy update, the system must retain
the action token IDs, action mask, sampling settings, and old-policy log probabilities for every
controller turn. It must also produce groups in which the reward differs across otherwise matched
attempts. If every attempt receives reward 1, the within-group training advantage is zero.

The probes deliberately stopped instead of converting invalid attempts into reward zero.

| Probe | Outcome | Lesson |
| --- | --- | --- |
| V0 | Four stochastic attempts were all correct and effectively identical. | Reward variance was zero; no update was justified. |
| V1 | Raising temperature led the controller to attempt a forbidden subcall. | An invalid run is not a legitimate reward-zero trajectory. |
| V2 | Strict replay found tuple/list serialization disagreement before stochastic exposure. | Evidence must reconstruct exactly, not merely compare equal after normalization. |
| V3 | Four valid stochastic attempts all earned reward 1. | Token and log-probability capture worked, but reward variance was again zero. |
| V4 | A two-task batch design was implemented and CPU-tested. | Infrastructure improved, but no live attempt or optimizer step occurred. |
| V5 | vLLM failed before readiness because the NVML user library and kernel driver disagreed. | The attempt produced zero sends and was retained as terminal-invalid. |
| V6 | The corrected server became available, but the controller failed on a multiline input contract. | Four sends occurred, but no episode was admitted and no reward evidence was produced. |
| Posthoc diagnostic | Allowing one more controller turn produced 0/16 completed episodes. | A larger turn ceiling alone did not repair the input-format weakness. |

Across all these probes, **the optimizer never started**. This is not RLVR efficacy evidence. It is
evidence that the pipeline can refuse unjustified updates and that task/input design must be
improved before spending compute on policy optimization.

Primary sources:

- `/project/alex_phd/runs/rlm-v2-spikes/rlvr-one-step-smoke-v2-artifacts/V2_TERMINALIZATION.json`
- `/project/alex_phd/runs/rlm-v2-spikes/rlvr-one-step-smoke-v3-artifacts/V3_TERMINALIZATION.json`
- `/project/alex_phd/runs/rlm-v2-spikes/rlvr-two-task-smoke-v4/README.md`
- `/project/alex_phd/runs/rlm-v2-spikes/rlvr-variance-probe-v5-attempt-001/attempt-journal.jsonl`
- `/project/alex_phd/runs/rlm-v2-spikes/rlvr-variance-probe-v6-attempt-001/attempt-journal.jsonl`
- `/project/alex_phd/runs/rlm-v2-spikes/v6-posthoc-repair-probe-run-001/RESULT.md`

## V3B: four completed matched training runs, evaluation pending

V3B was motivated by the input-format weakness. It asks whether exposing the controller to six
semantically equivalent request layouts during SFT improves robustness compared with training on
one canonical layout.

The control and treatment arms used the same 1,656 latent training episodes, 2,592 supervised
turns, 24 operations, action targets, references, and training budget. Only the visible request
layout changed. The treatment balanced six layouts that varied header order, blank lines,
irrelevant metadata, and JSON key order.

Two paired training seeds were run with a GPU crossover. For one seed, control used GPU 0 and
treatment GPU 1; for the other, the assignments were reversed. All four runs reached 648 steps and
published valid adapters.

| Seed | Arm | Final training loss | Runtime |
| --- | --- | ---: | ---: |
| 27082026 | Control | 0.07389 | 3,750 s |
| 27082026 | Varied layouts | 0.07502 | 3,792 s |
| 27082027 | Control | 0.07114 | 3,769 s |
| 27082027 | Varied layouts | 0.07205 | 3,747 s |

The mean treatment-minus-control loss difference was +0.00102. That is an optimization
diagnostic, not a behavioral result. The experiment record explicitly says that efficacy is
withheld: the paired held-out evaluation and sealed test have not run. The public-worker
qualification needed for that evaluation also remains review-locked.

The presentation-safe statement is: **four matched SFT runs completed successfully; whether varied
input layouts improve behavior remains to be measured.**

Primary sources:

- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3b-input-contract/README.md`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3b-input-contract/docs/superpowers/specs/2026-08-25-v3b-input-contract-design.md`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3b-ops-20260825T235900Z/TRAINING_RESULT.json`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3b-ops-20260825T235900Z/RUN_LOG.md`

## V3C and `rlm-bootstrap`: designed future work

V3C asks whether shorter operation-specific code after the controller has inspected the task can
reduce generated tokens and latency without reducing correctness. Its audit found that one
universal 1,431-token dispatcher accounts for most supervised target tokens, but that the
dispatcher cannot safely be replaced at the first turn because the operation is not yet visible.
The candidate therefore compacts only actions after an observation has exposed the task. It is
CPU-audited but review-locked; no V3C GPU training or evaluation has occurred.

The sibling `rlm-bootstrap` project addresses a larger question. It plans to compare a model
trained on ordinary direct solutions with the same initial model trained on successful work
performed inside the RLM. Every checkpoint should then be evaluated both directly and under a
fixed RLM harness. This comparison is designed to ask whether RLM practice teaches useful
decomposition rather than merely providing more calls at test time. No `rlm-bootstrap`
self-training run has occurred yet.

Primary sources:

- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3c-compact-design/DECISION_MEMO.md`
- `/project/alex_phd/repos/rlm-bootstrap/rlm_self_training_experiment_handoff.md`
- `/project/alex_phd/repos/rlm-bootstrap/docs/superpowers/specs/2026-08-24-clean-causal-self-sft-design.md`

## Why the GPUs were occupied for hours

The often-mentioned 691.8 seconds, or 11.5 minutes, is only the V2A LoRA training job shown in an
earlier version of the slides. It is not the duration of the research campaign.

### Training time

- V1 training: 269.4 seconds.
- V2A training: 691.8 seconds.
- V2B training: 332.3 seconds.
- V3A training: approximately 38 minutes from retained checkpoint timestamps.
- V3B training: four jobs of 3,747--3,792 seconds each.

The four V3B jobs alone consumed about 4.18 GPU-hours. They ran as two concurrent pairs, so they
occupied both A100s for roughly 2.1 hours of wall-clock time. Adding the earlier SFT jobs and V3A
puts identifiable training above five GPU-hours.

### Evaluation and serving time

Evaluation was often longer than training because tasks were run serially through the controller,
executor, verifier, and trace checks. Summing the elapsed time stored in completed episode records
gives approximately:

- 28 minutes for the principal V1 evaluation;
- 72 minutes across the V2A screen, confirmation, and sealed phases;
- 154 minutes across the V2B screen, confirmation, and sealed phases; and
- 169 minutes across the V3A development screen and confirmation.

These principal evaluation records total roughly seven hours of episode elapsed time. That number
is not the same as seven hours of continuous GPU arithmetic: it includes controller orchestration,
Python execution, time limits, validation, and failed episodes while a model server remained
loaded.

Additional time went to model loading, vLLM kernel compilation, availability canaries, startup
failures, RLVR probes, restart checks, artifact audits, and clean shutdowns. vLLM reserves model
weights and key/value-cache memory even while it is waiting between requests. Therefore a GPU can
look occupied in memory while utilization is low.

No continuous GPU-utilization log was recorded, so an exact active-compute total cannot be
reconstructed. The artifacts do, however, explain why the reservation lasted hours: the campaign
contained multiple training runs and thousands of evaluated or diagnostic episodes, not one
11-minute job.

## Overall assessment

### What worked

- The train, checkpoint, serve, RLM, trace, and exact-verification loop works reproducibly.
- LoRA made repeated Qwen3-8B experiments practical on one A100 while another served evaluation.
- Narrow, causally valid RLM routines were learned reliably.
- A fixed dispatcher transferred perfectly to split-disjoint values and renderers within its
  explicit schema.
- Broader SFT learned several multi-step local workflows, including inspection, computation,
  state persistence, verification, and controlled repair.
- The evaluation and RLVR gates preserved negative evidence instead of silently treating invalid
  executions as ordinary wrong answers.

### What failed or remains unresolved

- V2A used targets that were not determined by the model-visible state and produced shortcut
  specialization.
- V2B's perfect score came from one repeated program and says little about flexible policy choice.
- V3A did not solve either model-delegation family and has no sealed or replicated result.
- The RLVR probes never produced a valid, informative rollout group followed by an optimizer
  update.
- Input-format variation exposed brittle parsing and recovery behavior.
- V3B has trained adapters but no behavioral comparison yet.
- No completed experiment demonstrates open-ended recursive decomposition or compositional
  generalization on unfamiliar problem families.

### Most important scientific lesson so far

The harness and the training curriculum jointly determine what can be learned. A target must be
recoverable from what the controller has actually observed. A reward must separate better and
worse valid behavior. A held-back set must vary the aspect for which a generalization claim is
being made. Strong provenance cannot rescue a weak causal comparison, but it can reveal exactly
where the comparison is weak and prevent an invalid run from becoming a misleading result.

## What the short slide deck should claim

The deck should communicate only three preliminary findings:

1. We built and exercised the complete RLM training and evaluation loop.
2. Supervised training reliably installed narrow RLM behavior and then expanded to several
   multi-step local workflows.
3. The larger question—whether this produces reliable recursive decomposition and compositional
   generalization—remains open and motivates the next experiments.

The deck should not use V2A as its main positive result, describe V2B as broad reasoning, describe
V3A as sealed or replicated, claim reliable model delegation, or imply that RLVR training has
already occurred.

## Likely advisor questions

### What model did you train?

We used Qwen3-8B at a pinned model revision. It is small enough for LoRA training on one 40 GB A100
while another A100 can serve an immutable evaluation copy.

### Was this full fine-tuning?

No. The base model stayed frozen. We trained an approximately 43.6-million-parameter LoRA adapter,
then loaded the saved adapter alongside the base model for evaluation.

### What dataset did you use?

The data were generated locally rather than downloaded from Hugging Face. Deterministic programs
created symbolic list, string, record, and nested-JSON tasks; scripted expert policies performed
the RLM work; and exact private programs verified the results. Dataset sizes and behaviors changed
across V1, V2A, V2B, and V3A, as described above.

### Why use synthetic tasks?

They let us know the exact answer, construct controlled counterfactuals, isolate whole task groups
between splits, and determine whether a failure came from the answer or from the RLM work record.
They are useful for mechanism research, but success on them is not evidence of broad real-world
reasoning.

### Did the test data differ from training?

Yes, but by different amounts in different experiments. V2B used new values and renderers under
the same explicit payload schema. V3A used new latent instances under the same operation algebra
and canonical renderer. No completed experiment yet tests a broadly new natural-language problem
family.

### Why is the 192/192 V2B result not the main conclusion?

Because every training target was the same fixed Python dispatcher. The result shows that SFT can
install and reproduce that dispatcher reliably. It does not show that the model learned when or
how to invent different decompositions.

### Did the model learn decomposition?

It learned several local multi-step patterns: inspect the hidden request, compute after observing
it, carry state forward, verify the state, and repair a controlled visible fault. We have not yet
shown reliable recursive decomposition through additional model calls.

### Did you train with recursive model calls?

V3A included delegation and batch-subcall examples. The trained model attempted the expected
interfaces but did not solve those evaluation families. V1, V2A, and V2B do not provide evidence
of learned recursive calls.

### Did RLVR improve the model?

No. The probes stopped before any optimizer update. Some valid rollout groups were all correct and
therefore had no reward variation; other attempts were invalid. The next RLVR experiment needs
tasks that generate both valid successes and valid failures under a frozen policy.

### Why not use a larger model across both GPUs?

The first scientific question was about the effect of training and the harness, not maximum model
size. Qwen3-8B fits on one A100, leaving the other for an independent rollout or evaluation copy.
This makes model identity and update boundaries much clearer. V3B used both GPUs concurrently for
independent matched training runs rather than splitting one model across them.

### Are the comparisons equal-compute?

No. Direct answering and RLM execution can differ in prompt length, output length, Python work,
and number of calls. V2A and V2B each used one model call per task in their sealed comparisons,
but the RLM generated much longer outputs and executed Python. V1 used two controller calls versus
one direct call. Future comparisons should report both accuracy and complete resource use.

### What comes next?

The immediate unfinished work is the held-out paired evaluation of V3B. The `rlm-bootstrap`
experiment then compares direct self-training with RLM-trajectory self-training from the same
initial model. Reward-based training should resume only after a frozen probe demonstrates valid
trajectories with useful reward variation. Longer-term tests should use new task structures where
decomposition quality, not just arithmetic correctness, can be measured clearly.
