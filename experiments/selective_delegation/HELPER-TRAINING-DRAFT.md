# Helper-only training: a bounded decision draft

Status: design accepted for CPU preparation. The immutable 570-example package is
now exported to `helper-sft-inputs-001`, with sealed preparation sources and native
token lengths. `train_planner.py --role helper` is prepared and CPU-tested; no
optimizer or GPU work has been launched here. Main retains launch authority.

## Question and alternatives

Can a small helper adapter, trained to answer annotated subquestions in strict
JSON, improve end-to-end accuracy with the **same frozen SFT planner and base
final**? Distinguish two possible effects: fewer protocol failures versus better
answers among already-valid outputs. Either can be useful, but they are not the
same scientific result.

The RL training panel had 256 valid roots but 29 invalid-helper outcomes. One
parent failed helper parsing on all 16 attempts. This is a plausible helper
target, not evidence that helpers explain every failure: inspected examples also
include an unsupported death-place question, a terse relation question answered
with a date rather than location, and correct helper answers overturned by final
synthesis. Four always-EM-zero parents also had zero F1 throughout; replacing EM
with F1 supplied no new mixed-reward groups in that panel.

Three bounded options, not a request to launch all three:

1. **Cheap interface diagnostic first, if the completed aggregation warrants it.**
   Freeze sampled plans and compare raw relation shorthand with deterministic
   natural-language rendering: a generated `X >> relation` can become
   `What is the relation of X?`. Transform only an explicitly recognized single
   relation form; otherwise retain the original question. Bind references to
   actual preceding predictions, never annotations. Preserve both raw and rendered
   questions in receipts. This is a prompt/interface change, not training or an
   answer fallback; awkward or ambiguous renderings remain a limitation. It must
   be a separate arm, not silently combined with helper training.
2. **Recommended minimal training comparison:** fresh 4B helper LoRA, one fixed
   epoch on the 570 train-only step examples below, versus the exact frozen 4B
   helper contract without that adapter. This tests adaptation to terse questions
   and strict answer formatting without changing planner/final capacity.
3. **Frozen 8B helper diagnostic:** useful if helper content remains limiting after
   interface checks. It asks whether changing helper capability helps, not whether
   4B helper SFT works or parameter count alone is causal. No training is needed,
   but two resident models and different model vintages add implementation/cost.

## Supervision, binding, and split boundaries

Use only the existing 256 `split=train` parents in `inputs-001/cases.jsonl`:
198 two-hop and 58 three-hop parents, yielding **570 annotated steps**. All step
answers are nonempty strings. The step fields are `question`, `answer`, `id`, and
`paragraph_support_idx`; only the question and answer participate in supervision.

For step i, resolve each `#k` using the earlier **annotated** step-k answer, through
the existing one-pass strict dependency binder. Reject forward or unresolved
references; do not substitute recursively inside answer text. The CPU audit found
zero binding errors. There are 263 questions without references, 300 with one,
and seven with two references.

The prompt must be exactly `isolated_helper_prompt(case, resolved_question)`:
current question plus all public document `id`, `title`, `text` fields, using the
existing short-span instruction. Do not inject the original composed question,
full plan, step IDs, support indices, source IDs, component IDs, hop count, final
gold, or future answers. The target is only serialized `{"answer": step_answer}`
plus native EOS. Do not include rationale or support IDs in the target. Gold
entities appearing naturally in document evidence are not removed.

Host-only example fields can retain parent ID, step index, split and provenance;
model-facing fields remain prompt/target strings. Sort train parent IDs, shuffle
parents with fixed seed 2026092111, retain within-parent step order, and freeze the
example order before training. Exact repeated examples, if any, must be counted
and disclosed rather than silently reweighted.

This is **privileged teacher-forced supervision**. Both annotation questions and
gold previous answers make training easier than deployment. At inference, the
frozen planner supplies its own questions and references bind only actual prior
helper predictions. Training on gold bindings does not teach recovery from wrong
earlier predictions, unsupported questions, or malformed model-generated plans.
Do not bind a wrong predicted entity to a training question and retain its old
gold answer as if that were a valid supervised pair. Predicted-history adaptation
would need separate trustworthy labels and is outside this small experiment.

The immutable inputs contain 64 validation parents from official dev (2/3-hop)
and 64 transfer parents from official dev (4-hop), while training comes from
official train. The CPU metadata audit found **zero shared atomic component IDs
between any pair of these splits**. Within-split overlap still matters. Never
randomly split individual steps across train/validation: siblings and components
must remain together. All 64 validation parents are research-development material
once used for the current adaptive decisions; do not rename them a fresh test set.
Keep transfer outputs unopened until helper architecture, dose and checkpoint
choice are frozen. Final transfer scoring is a separate four-hop robustness test,
not an excuse to keep tuning on transfer failures.

## Smallest viable SFT recipe and measured context budget

Reuse `train_planner.py`'s proven optimizer/checkpoint mechanics, **not its data
projection, length caps, or trained adapter initialization**:

- Same pinned released Qwen3-4B-Instruct-2507 BF16 base; fresh rank-8, alpha-16,
  dropout-zero FP32 helper LoRA on q/k/v/o/gate/up/down projections.
- AdamW LR 1e-4, weight decay zero, clip norm one; microbatch one, 16 examples per
  update, prompt loss masked, CE normalized by target tokens in the batch.
- One epoch over all 570 examples: **36 updates**, last batch ten examples.
  Fixed final checkpoint is primary; save every eight updates and epoch end.
  No checkpoint search or automatic extension to three epochs; the helper-role
  implementation rejects more than one epoch or more than 45 minutes.
- Native one-user chat, thinking disabled, generation prompt, separately encoded
  answer JSON plus one EOS; cache off, nonreentrant activation checkpointing.
- First accepted run capped at 45 minutes and lease minus 600 seconds, with normal
  exclusive ownership. Measure the first two updates' time/peak allocation and
  project remaining duration; if the full epoch cannot fit, preserve the partial
  checkpoint and mark the dose incomplete. This is a compute cap, not a measured
  runtime forecast. Long prompts make the earlier 297-second planner-SFT timing
  an invalid estimate for this helper run.

Actual train-only CPU tokenization with the existing 4B tokenizer:

| Quantity | Result |
|---|---:|
| Helper examples | 570 |
| Prompt tokens, minimum / median / maximum | 1,548 / 2,757 / 4,819 |
| Target JSON + EOS tokens, median / maximum | 8 / 24 |
| Maximum prompt + target | 4,826 |
| Exceed current planner prompt cap 512 | 570 |
| Exceed total context 2,048 / 4,096 / 6,144 | 538 / 13 / 0 |

Use an explicit **6,144-token training total cap**, no truncation, and target cap
48. Every observed target fits even the smallest existing inference helper budget
`floor(384/8)=48`. Training does not change inference caps: use total helper output
384, divided by actual predicted plan length, and final output 128. Longer prompt
gradients still require a measured GPU memory check; CPU token counts do not prove
training memory feasibility.

## Exactly matched baseline and adapter roles

Freeze root to **planner SFT checkpoint 48**; do not jointly change it to RL4 while
testing helper adaptation. Reuse identical sampled root-plan receipts for the
base-helper and trained-helper conditions, with the existing title-index-only
observation and no provisional answer. Each helper condition then runs its own
sequential predictions and binds subsequent questions to its own answers. Final
always uses the same released base model, full source, fixed root plan and that
condition's actual helper trace. No gold, reference plan or annotated answer enters
either deployed condition.

| Role | Control | Helper-SFT condition |
|---|---|---|
| Root | Frozen planner SFT48 adapter | Same adapter and same sampled plan |
| Helper | All adapters disabled | Fresh trained helper adapter only |
| Final | All adapters disabled | All adapters disabled |

One 4B base can carry two named PEFT adapters. Select exactly one by role; never
compose, merge, or leave the helper adapter active for root/final. Calls remain
sequential because adapter selection is process-global. Eval mode and frozen
parameters are mandatory; check named active adapters, native model/adapter hashes,
and role receipts. This role-routing change needs a focused tiny-model test; the
current evaluator's single-trained-root rule cannot simply enable its adapter on
helpers without loading a distinct helper adapter. The design does not require
another model server or a general multi-model framework.

Use T=.5, top_p=1, top_k=0, native token accounting, identical per-parent/repeat/step
seeds, strict JSON parsers, and no repair/retry/fallback in both conditions. Invalid
helper responses end the episode with zero downstream reward. Count all planned
parents/repeats, including malformed roots; common frozen roots fail identically.
Charge shared root acquisition once physically, but one root per hypothetical
deployed attempt. Count actual helper/final tokens, latency, and aborted attempts;
do not count a fixed 384-token allocation as tokens actually generated.

Primary metrics: end-to-end official alias-max EM/F1 with paired parent intervals,
and per-episode protocol outcomes. Separately report helper JSON validity, helper
answer lengths, and matched episode changes where both arms completed valid
finals versus changes involving a protocol failure. Annotated helper EM/F1 is
valid for a separate gold-question/gold-binding diagnostic, but not automatically
for arbitrary model-generated subquestions. Do not equate every model helper's
last answer with the composed question's gold answer.

## Frozen 8B alternative: plausible fit, not a size-only comparison

The already-cached model is `Qwen/Qwen3-8B`, revision
`b968826d9c46dd6066d109eabc6255188de91218`; its manifest records Apache-2.0 and
verified cache files. It is **not** Qwen3-8B-Instruct-2507. Compared with the current
4B-Instruct-2507 helper, version/training recipe and chat behavior change alongside
size. Describe any result as a heterogeneous frozen-helper substitution.

Safetensors index totals are 16,381,470,720 bytes for 8B and 8,045,591,552 for 4B:
24.427 GB decimal (about 22.75 GiB) of tensors together. A co-resident frozen 4B
root/final plus 8B helper is plausible on the 40GB A100, but KV caches, prefill
workspace, activations, adapters and allocator reservations still need a brief
actual-load/real-response check. No load or fit claim was made in this draft.

Keep 4B root SFT48 and 4B base final unchanged. Route only helper calls to frozen
8B, using its own native tokenizer/chat template with thinking explicitly disabled.
Override sampling defaults: the cached 8B configuration defaults to T=.6,
top_p=.95, top_k=20, so inheriting it would not be matched. Use T=.5/top_p=1/top_k=0,
the same strict answer JSON and helper output cap; count each model's actual native
tokens and time. Sampling seeds can match, but tokens and random trajectories are
not identical across models. Never attempt to attach the 4B adapter to 8B.

## Promotion, revision and stopping rules

Main first decides whether the completed final-synthesis results leave enough
helper-limited headroom to justify this work. Do not launch helper training merely
because a flat reward group exists: one inspected flat group already had correct
helpers and a wrong final.

Before launch, freeze the development panel, repeats, final checkpoint and triage
criteria. A practical small-panel promotion rule is either (a) at least two net
additional correct episodes with at least one improvement where both finals are
valid, or (b) at least 50% fewer helper-protocol failures without worse total EM.
These are exploratory budget-allocation rules, **not significance tests**; publish
paired parent intervals and overlapping-component caveats regardless of outcome.
Label success under (b) protocol learning, not semantic improvement.

If only teacher-bound step accuracy improves but model-plan end-to-end outcomes do
not, revise the exposure/interface hypothesis rather than extending epochs. If
JSON failures fall but content errors persist, consider the frozen 8B helper probe.
If both helper arms produce correct intermediate answers that finals overturn,
pivot to the already-motivated aggregation question. If a deterministic shorthand
rendering matches the training benefit, keep that cheaper explanation/control in
view. A negative small-panel result is underpowered, but does not justify an
unbounded dose or prompt search. Freeze one chosen design before any untouched
transfer evaluation.

## Inputs and provenance

Campaign root: `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Cases: `inputs-001/cases.jsonl`, SHA256
`0aebb983cf5c91b38f8bbee93e6fbf1ebe86588fd59048f54c140ae17bf95ef8`.
Root checkpoint: `planner-sft-001/checkpoint-0048`; helper-failure motivation:
`rl-planner-001`, not a held-out helper-training result.

4B cache:
`/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554`.
8B cache:
`/project/alex_phd/research-cache/models/Qwen--Qwen3-8B--b968826d9c46dd6066d109eabc6255188de91218`.
Existing Python:
`/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python`.

CPU audit used the real `isolated_helper_prompt`, `bind_question`, and native
tokenizer with teacher-bound train annotations. Any later preparation must seal
the exact source, examples, model/tokenizer manifest and seed before optimizer
work. New adapters, training rows and receipts stay in a fresh external run
directory; this draft does not authorize changing an active sealed owner.

Prepared examples SHA256:
`9ed43882b914774fb9ba704608d555cb643f06cf5bcfb876e6fd4e90c1557faa`.
There are zero duplicate prompt/target pairs. Rows contain host-only `id`,
`parent_id`, zero-based `step_index`, and `split`, plus model-facing string
`prompt` and JSON-string `target`. Dataset order is parent-shuffled then step
ordered; the reused trainer applies its separately recorded seeded epoch shuffle
over these independent supervised examples. The last of 36 updates has ten rows.

Launch syntax, for main only after sealing the final trainer source:

```text
TRAIN_PYTHON train_planner.py --role helper --prepared R/helper-sft-inputs-001 \
  --output R/helper-sft-001 --epochs 1 --seed 2026092111 --hours 0.75
```

The helper role validates prepared-example and sealed-source hashes, counts 256
parents separately from 570 step examples, records the base manifest, preserves
fresh helper initialization, and enforces the cumulative owner time cap across
explicit committed-checkpoint resumes. The original planner role retains its
512/256/2048 caps and effective three-epoch/seed-20260921 defaults. Checkpoint,
target-loss, and epoch-order functions shared by the RL runner are unchanged.
