---
status: running_first_update
updated_utc: 2026-09-22T12:57:00Z
question: Can terminal success improve a model already taught public information gathering?
depends_on: complete_textcraft_train_readiness_001
maximum_optimizer_updates: 2
---

# TextCraft terminal-reward RL readiness

The corrected pilot and its controls were accepted at 11:26 UTC, after the
complete TRAIN audit and exact replay through the trainer. Queue008 started
`textcraft-terminal-rl-002` at 12:51 UTC after the fresh-goal comparison released
the GPU. Its first real response returned in 2.03 seconds. Saved pre-update
likelihoods and the longest-call backward qualification are complete; the
qualification recorded 10.9 GiB peak allocated memory and no base-model gradients.
The first training gradient pass is in progress, with no committed optimizer
update yet. These are execution checks, not evidence of RL improvement.

The [fresh-goal comparison](TEXTCRAFT-FRESH-FINDINGS.md) completed at 1/32 versus
15/32 for the earlier and revised demonstrations; the RL job remains the
previously accepted objective, tasks and caps. The earlier preparation notes
below remain historical context; the dated readiness result and acceptance
section give the current source and run pointers.

## Recommendation

If063 has usable mixed-success groups at affordable trajectory lengths, try at most **two actual on-policy RLOO updates** from the fixed public056 checkpoint23. Retain all eight preselected TRAIN tasks and four samples per task, including flat groups and paths containing invalid actions. Do not train only the mixed groups or convert unavailable episodes to zero. If variation occurs in only one task, state that the effective learning signal is one training task; a two-update result cannot establish generalization.

The older trainers were not drop-in implementations of this objective. A small
dedicated [terminal-RLOO trainer](rl_textcraft_terminal.py) is now prepared, with
explicit multi-turn token credit and collection/training mode switching. It is
not an SFT step or the older turn-normalized objective.

The conditional [trajectory-loss component](textcraft_trajectory_loss.py) now has
four focused CPU tests, also run by the main agent. They check complete groups,
token-summed credit, causal alignment and an actual saved request. This is only
loss arithmetic and input validation: no optimizer step, model-mode transition,
GPU memory qualification or learning improvement has been demonstrated.

The dedicated trainer adds five CPU tests using a tiny actual Qwen/PEFT model.
They cover generation/replay agreement, mode changes, Adam/checkpoint boundaries
and rejection of an excessive probability discrepancy before updating. All nine
trainer/loss tests passed in the sealed source and on the main agent's rerun;
the main rerun also passed three independent watcher tests. These tests do not
establish memory feasibility or learning on the actual4B model.

Sealed implementation: `R/source-065-textcraft-terminal-rl`, source manifest
SHA256 `45ee12e56bb6548c510cc5e1b3ebf01145a885beff7e3f1f7f7ec8df4adafe74`.
The proposed command and pins are in `R/TEXTCRAFT-TERMINAL-RL-PROPOSED-001.json`;
future run PLAN SHA256 is
`77dbe538bda51f4957d4bbd7170b579ea43bfcd7b1cdd0133faf70a8bf2b26a6`.
Before GPU admission, replay the completed063 batch through the trainer's native
audit on CPUs. The matched extra-SFT control is now CPU-qualified in
`R/source-textcraft-matched-sft-001`, with its frozen outcome-independent row
order and focused tests described below. It remains proposed, not GPU-accepted
or a completed scientific comparison.

## Exact objective and masks

For task g and its four independently seeded episodes i, let R(g,i) be the **native root finish success**,0 or1, under the unchanged original flat interface and96-call/8192-output-token/256-per-call/8192-context limits. Returned schema errors, rejected commands and failed crafts remain actions in that episode; later recovery can still earn1. No intermediate recipe, inventory or action-validity shaping is added. No explicit successful finish means0 when the episode is observably terminated by the harness. Inference exceptions or interrupted/unrecorded episodes are missing, not0; do not optimize an incomplete32-episode batch.

Use A(g,i)=R(g,i)−sum(other three rewards)/3 and

`loss = − sum_g,i A(g,i) * sum_calls sum_emitted_tokens log πθ(token | exact_saved_prefix) /32`.

Every actor-emitted token receives the episode advantage: JSON punctuation, item names, quantities, notes, invalid output, and emitted EOS. Never add a synthetic EOS to a cap-truncated response. Each call is replayed separately using its actual saved chat-token prefix plus `emitted[:-1]`; select the final `len(emitted)` logits against `emitted`. Tool feedback, public inventory, instructions and previous answers appearing inside the next prompt have zero **target-loss** mask. Do not concatenate the whole episode into a new artificial chat history, or count earlier actions again merely because they appear in a later prefix.

No division by number of turns or trajectory tokens; no advantage standard-deviation normalization, PPO epochs, KL penalty or importance clipping initially. The token sum is the score-function gradient for the sampled bounded episode policy. Mean-per-turn or mean-per-token losses introduce trajectory-length-dependent weighting and are not interchangeable with it. Long unsuccessful paths can have large gradient contributions; record lengths, gradient norm and clipping, not silently normalize them away. Global norm clipping1 is an explicit additional update transformation.

## Reusable code and actual seams

Paths below are under E=`experiments/selective_delegation` unless absolute. Hashes identify inspected versions, not permission to modify a live source.

| Component | Reuse / limitation | SHA256 |
|---|---|---|
| `R/source-063-textcraft-train-readiness/eval_textcraft.py` | `episode`, public prompts, strict parser, budgets, native receipts; `NativeClient` requires frozen parameters and does **not** save generation log-probabilities | `ddc7d49f5749258e620fc651de1cc84d96a581e44ee958965ad4f0e03fad1223` |
| `R/analysis-source-textcraft-train-readiness-001/analyze_textcraft.py` | Exact native call/episode replay and terminal grading | `6c7f5c4625ceb8bd85d0e6b6fe1a9cff3277e2aba2d138b6441de76be41bf3ca` |
| `rl_planner.py` | Four-sample `rloo`, per-response causal alignment, generation-score capture, LoRA parameter checks; existing temperature0.8 and denominator64 **must not** carry over | `404857539b5b6c0c84c2d8aeca255ec089fce1fee8eb2ac0d2f71bf7e8aec977` |
| `rl_sufficiency.py` | Distinct sampled/update cursors, zero-advantage skip and committed boundaries; existing16-parent/eight-block/four-zero constants are not this pilot's contract | `377228d03fa255c11cfdc56c22b2d34eced6460a1c6ddf41dc0a38f63c2b60e3` |
| `train_planner.py` | Atomic adapter+Adam+RNG+STATE+COMMIT checkpoint writer | `9ce66e6f2bc8774bf5f4a5ae0e4934fbad5fd04da8886cb799231b8582f562f8` |
| `train_textcraft_sft.py` | Known-working BF16 base/FP32 rank8 LoRA, target-only suffix loss, non-reentrant checkpointing | `274daede3a32aab96f3ad7914ae22b00cb9c4301efb39f457eeb8c786e6ec439` |
| `R/../root-rlvr-campaign-v1/campaign_train.py` | Existing multi-turn terminal-credit implementation and Adam identity checks, **not the desired loss**: averages turns within each episode | `38fe3087b8deb94ee8e0fa2e0330ca34a822f84192031c389ff32acf00d79d1f` |
| `R/../single-gpu-rlvr-v2/source/single_gpu_rlvr_tis_v3.py` | The older trainer's token-mean TIS loss; inspect masking, do not import its normalization/correction or mixed-only admission | `eff526cf496d3896e220d5aa0694bed3b703114e4a19d6f7c9e4f73de6240bd9` |

R is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

The minimum new seams are: a frozen rollout mode for the trainable actor; per-call behavior-log-probability receipts at T0.5; a loop applying one terminal advantage to all that episode's action-token sums; and a small sampled-boundary cursor. No new environment, server, retriever, model role, or hierarchy is needed.

## On-policy identity and replay

Warm adapter SHA256`f029d36967eb1cef30e102fe82e52f22f90d5eacdaf91e82937b60218ba5b00f`, COMMIT SHA256`616bab7a57a7845cb21f7d32ffa76b8deb280bc7604459d171d5ea69648685cf`. Load that exact056 checkpoint, not a fresh LoRA or privileged048 weights. Start a declared new optimizer phase for **both** RL and its SFT control: AdamW LR2e−5, weight decay0; retain rank8/alpha16/dropout0. Resetting Adam moments is an explicit design choice, not continuation of056's SFT optimizer state.

Freeze every parameter during collection, eval mode/cache enabled. Only after the entire batch is recorded and independently regraded, enable the same saved FP32 LoRA parameter list for one gradient pass; base stays frozen. Restore eval/cache mode before the next batch. Every call binds the current adapter COMMIT. Use T0.5,p1,k0 for sampling **and** log-softmax; no changing weights between calls within a batch. Save generation processed log-probabilities for fresh rollouts and independently teacher-force the same emitted IDs under the same weights before updating. Compare the two and record BF16 train/eval replay discrepancies. Save before/after per-token likelihoods, gradient norm, adapter delta and full reward/advantage inventory.

Optional cost-saving first batch: reuse **all** complete063 trajectories, since they were sampled from exact056 with the same sampling contract. Their old log-probabilities must be recomputed under unchanged056;063 has no generation-score receipts, so do not claim those were measured during sampling. A tiny native generation/replay check at T0.5 must qualify the new path before an optimizer step. Starting only after observing mixed063 rewards makes this an explicitly readiness-conditioned exploratory update, not an unbiased unconditional learning-effect experiment. If that provenance is inconvenient, collect a fresh complete32 under056 instead; never reuse063 after updating weights.

## Bounded schedule, memory and controls

The trainer intentionally does not automatically promote a partial endpoint after a cap or failure. Already committed checkpoints retain adapter weights, Adam state, RNG and cursor: they are not lost. A later explicit manual-resume or declared partial-readout decision can use an audited committed checkpoint; the current automatic successful-endpoint contract requires normal completion. Before any optimizer step, save the actual train/eval token-log-probability discrepancy and reject max absolute error above0.25 or mean above0.025, using the existing gradient-pass forwards (no extra forwards).

Keep eight tasks × four samples, interleaved across tasks, with fresh predeclared seeds per sampled batch. Maximum two actual updates, four sampled batches, or three hours, whichever comes first. A complete all-flat batch writes a zero-update boundary and advances to the next fresh seeds; it does not abort at the first flat batch. Stop after two consecutive all-flat batches, preserving all groups. This avoids unlimited sampling for favorable batches. Checkpoint each sampled boundary, including skips, using separate sampled-cursor directories so an unchanged optimizer step cannot reuse an old STATE. An interrupted batch is preserved without an optimizer step; no implicit retry or partial-group update.

OneA10040GB is plausible with one BF16 base, FP32 LoRA/Adam, SDPA, microbatch **one call**, immediate backward/gradient accumulation, non-reentrant activation checkpointing and no training KV cache. The local model weight index is8,045,591,552 bytes (~7.49GiB); inference KV at8192 tokens is about1.125GiB from36layers×8KVheads×128head dimension. Output-only256×151,936 FP32 logits are ~148MiB; projecting all8192 positions would create ~4.64GiB per FP32 logits tensor and is unnecessary. These are component estimates, **not a measured40GB peak guarantee**. Qualify a longest observed063 call backward before the first update; preserve any OOM as an observed seam rather than changing context/truncation silently.

Completed057 used871 calls/32 episodes and2,131 native service seconds; this is a rough cost reference, not a forecast for TRAIN. Replaying long prefixes several times adds substantial compute. If063 itself approaches its one-hour cap, a full fresh two-update pilot is not a cheap job. Reusing its complete initial batch is the cheaper scientifically explicit option; otherwise stop at readiness rather than introduce a framework.

Matched extra-SFT control: initialize from the same056 weights and fresh Adam recipe, use the existing public055 demonstrations for these **same eight tasks**, and take exactly the same number of actual optimizer updates (zero if RL has zero). Deterministically select/order the supervision before any RL outcome; report token counts and FLOPs. Match per-update supervised-token budget to credited RL action-token count as closely as whole rows allow, with any excess declared, or use an explicitly update-matched fixed row schedule. Neither is simultaneously identical in states, target information, token dose and gradient scale. SFT is a useful “more supervised optimization” control, not proof that any difference is caused solely by RL. Frozen056 remains the primary no-update baseline.

Afterward evaluate only the last committed endpoint, never the best TRAIN reward checkpoint. Use a separately fixed held panel for base056/RL/extra-SFT; the existing exposed eight VAL tasks can provide a pilot comparison but not fresh confirmation. No depth, gold trajectory, private recipe graph or verifier status enters the policy prompt. Native world state is confined to trusted tool execution/reward, while discovered recipes become ordinary public observations. If063 is all-success there is no terminal-reward gradient; if all-zero or only transport failures, do not implement this RL pilot yet. Resolve competence/reward availability or retire the branch first.

## Prospective token-and-update-matched extra-SFT control (CPU preparation)

Fixed inventory, inspected2026-09-22 before065 training: the eight063 TRAIN tasks
have86 existing public055 demonstration rows:39 query,39 craft,8 finish.
Counts by task suffix are1680:9,1791:5,2040:7,2294:9,404:11,429:19,465:11,860:15.
Native tokenizer audit gives2074 target tokens **including EOS**,83666 prompt
tokens, maximum target51 and maximum prompt+target2005. No VAL rows or new gold
trajectories enter. The full055 row-file SHA is
`dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a`;
the063 PLAN SHA is
`e96a7ac7e9ded7695217378cd1668607c68cc5c0bbe244d40a88842b4c691444`.

Sort rows by(task_id,step), shuffle once with seed2026092243, freeze the exact
row-ID cycle before RL outcomes, then continue this cycle across updates.
For each **committed actual065 optimizer update**, independently recompute
four-sample RLOO credit from its BATCH episodes, and sum emitted native token
counts of every action with nonzero advantage. Include invalid actions and any
actually emitted EOS; do not count flat-group tokens, qualification forwards,
or sampled batches without an optimizer step. Cross-check the result against
the committed STATE.update replay-token count and nonzero-action count.

Consume whole teacher rows until reaching or exceeding that update's credited
token budget; no subset search or target truncation. Report each overshoot
(strictly less than the last row's target length, at most50 here), repeated
rows, prompt tokens, and totals. This one-sided rule matches dose closely while
never silently dropping the final example or using target content to optimize
the match. Initialize from exact public056 checkpoint23, fresh AdamW2e-5/wd0,
clip1, existing rank8/alpha16/dropout0. Microbatch one call; sum teacher target
cross-entropy at temperature1 and divide by **actual selected target tokens**.
Checkpoint step0 and every actual update using the qualified existing writer.

The control requires normal065 terminal completion, authenticated owner release,
and its exact usable committed endpoint. Cap/error/unusable endpoint means no
automatic partial-dose training; zero actual steps means explicit no-training.
Proposed control cap30minutes, endpoint only after the full declared dose.
Different state histories, privileged supervised targets, prompt compute,
temperature1 SFT versus0.5 RL, gradient scale/clipping, and fresh Adam mean this
is token/update matched—not FLOP-, information-, or objective-matched. No GPU
acceptance or launch is implied. Small new control files may reuse existing
tokenization/loss/checkpoint helpers; all shared and sealed sources stay fixed.

CPU implementation is now in `train_textcraft_matched_sft.py`, with four focused
tests in `test_train_textcraft_matched_sft.py`: exact credited tokens including
invalid actions, cyclic whole-row overshoot, endpoint dose rejection, and a tiny
native Qwen/LoRA target-loss backward/Adam/checkpoint fixture. Existing helpers
provide tokenization, summed suffix loss and checkpoint writing. Four tests pass;
the synthetic tiny model emits PEFT's expected missing-config-path warning when
saving. No actual065 terminal exists yet, so completed-run receipt integration
and4B training/memory/time feasibility remain unqualified. No implicit resume.

The immutable order receipt is
`R/TEXTCRAFT-EXTRA-SFT-ORDER-001.json`, SHA256
`f4488f75ce7c316b3e429590d1b5490cc0472ca6c4895df615caa201281fb58b`.
After065 completes, CPU `--prepare-only` derives and pins the full schedule;
the same output PLAN can then be used only by a separately accepted launch:

```text
TRAINPY train_textcraft_matched_sft.py \
  --frozen R/TEXTCRAFT-EXTRA-SFT-ORDER-001.json \
  --rl-output R/<accepted065output> --output R/<newmatchedSFToutput> \
  --hours 0.5 --prepare-only
```

The row-order freeze contains
no RL or VAL outcomes; actual update lengths are determined only by the declared
committed RL dose. This is prepared code, not an accepted GPU control or a claim
of successful learning.

## Completed TRAIN readiness and order-replay repair — 2026-09-22 UTC

The completed063 run has **25 successes out of32 observed episodes**, with no
missing or unavailable outcomes. All32 issued explicit finish actions; the seven
failures are native goal failures, not transport failures or unfinished episodes.
The fixed eight TRAIN tasks and all four samples remain in the analysis:

| Official TRAIN suffix | Rewards, repeats0–3 | Group |
|---|---|---|
| 2040 | 1,1,1,1 | All success |
| 1791 | 1,1,1,1 | All success |
| 2294 | 1,1,1,1 | All success |
| 1680 | 1,1,0,1 | Mixed |
| 429 | 0,0,0,0 | All failure |
| 404 | 1,1,1,0 | Mixed |
| 465 | 1,1,1,1 | All success |
| 860 | 0,1,1,1 | Mixed |

Three mixed groups yield12 nonzero trajectory advantages; the other20 episodes
remain in the denominator with zero advantages. This establishes usable sampled
terminal-reward variation on TRAIN, not learning or held-out transfer. Main's
separate trainer integration replay reports388 credited calls and12,074 emitted
tokens; these are gradient-dose counts, not additional model calls.

Total cost:770 native calls,1,636,197 input tokens,24,009 output tokens and
1,899.39 native-service seconds (31.66minutes). There were zero transport errors
and zero schema errors. Public native feedback contains313 action errors across22
episodes. Counting each error by its unique literal native message category gives
115 insufficient-inventory errors,98 extra-ingredient errors,57 missing-ingredient
errors,33 wrong-quantity errors and10 nondivisible output-count errors. These are
diagnostics, never exclusions from terminal reward.

The useful RL question is now whether terminal credit strengthens correct
recipe/batch execution and finishing only after the goal is met, beyond another
matched supervised update—not whether it teaches JSON syntax. For example,
TRAIN404 succeeds in repeats0–2 but fails repeat3. In the failed trace, the public
recipe for `a2_i3_22` requires two `a2_i2_12` and one `a3_i1_23` per output;
attempted recipes repeatedly violate those quantities, then finish while all
three requested `a2_i4` remain missing. Repeat0 instead crafts four intermediates
with correctly scaled ingredients (eight `a2_i2_12`, four `a3_i1_23`), crafts the
root, and succeeds. TRAIN1680 and860 likewise each have one failed finish and
three successful samples. These illustrate execution differences, not localized
causal credit: whole-trajectory RLOO also reinforces unrelated earlier tokens.
TRAIN429 has no successful sample and supplies zero within-group gradient in
this first batch; nothing here establishes global task collapse or irreparability.

The initial sealed001 native analyzer failed before producing a report. The
collector's `freeze_inputs` wrote sorted-key task JSON but returned original055
in-memory dictionaries, whereas the auditor loaded the sorted copy. The first
request differed only in inventory-key order, which still changes exact prompts
and token IDs. The additive002 analyzer reconstructs original dictionary order
from hash-bound055 tasks and checks exact task IDs, list order and semantic values
against frozen063 inputs. Exact prompt, input/output IDs and request digests are
still audited. A real063 regression confirms sorted-order failure, restored-order
success and rejection of changed inventory quantities; all three focused tests
pass. Fresh007 loads its prepared JSON directly and has no such split.

Authoritative report: `R/analysis-textcraft-train-readiness-002.json`, SHA256
`b0f97a7b98ce572e87d583bb738010417006f61d669af84b05724ad74d2d3c9d`.
Analyzer seal: `R/analysis-source-textcraft-train-readiness-002/SOURCE.json`, SHA256
`47180eda3516b48c0aca1bbd488a28244b34fafac9b1c0774e59f5aa3c93ca05`.
The original failed source, watcher log and absent001 report remain preserved.

The corrected prospective RL pointer is now
`R/source-065-textcraft-terminal-rl-v2` (SOURCE SHA256
`b9de4ecb3e4f90cc5750b30f04d80db20f2d1eed86959e22011239b2541bea47`),
using `readiness.runtime_tasks(original_plan)` for reused and fresh batches.
New output `R/textcraft-terminal-rl-002/PLAN.json` has SHA256
`36dae80595f1b9decfeb446ef4abd4e5542926624d462a7511b383a4fb140c70`.
Original065 source/output001 are untouched and never used a GPU. The completed
TRAIN variation supports the bounded pilot's readiness criterion; actual GPU
admission and queue priority remain the main agent's decision, not this report.

## Accepted follow-up — September 22, 11:26 UTC

`R/INDEPENDENT-TRAINING-QUEUE-008.json` records four bounded jobs: corrected
terminal RL (`textcraft-terminal-rl-002`, at most 3 hours), matched extra SFT
(at most 30 minutes), then fixed-final RL and SFT readouts on the existing
16-goal panel (at most one hour each). The latter three reject failed or unusable
training endpoints before loading a model. There is no best-checkpoint choice
or automatic promotion of a partial run. This remains exploratory and uses
one training seed. The panel is reused, not a new untouched confirmation set.

Receipt SHA256:
`3194d991caed7746cd39e80a8fa026dc4760606c24ccb99df75ad3361121bc49`.
Main's sealed integration check (`TEXTCRAFT-TERMINAL-RL-ACTUAL063-SEALED-003.json`)
replayed all 32 episodes and 770 calls before admission. The queue supervisor
is PID159369/create1790076396.65 and waits for the entire authenticated queue007,
including release of every extant GPU owner. The current GPU run is unaffected.
