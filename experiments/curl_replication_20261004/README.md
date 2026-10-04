---
date: 2026-10-04
status: first_reference_completed_control_running
question: Can we reproduce one published CURL learning result and isolate the contribution of its contrastive objective?
stage: first_reference_and_control
completed_pilots: 1
completed_primary_runs: 1
primary_hardware: one_A100_40GB
---

# CURL reproduction: start here

We are temporarily setting aside new RLM experiments to learn from a published
RL implementation. CURL learns to control a simulated system from images,
using both task rewards and an image-matching learning objective. It is not a
language model, SFT recipe, or recursive agent. The intended benefit is a small,
auditable training and evaluation reference before returning to those systems.

Read the [learning guide PDF](../../docs/curl-replication-2026-10-04/learning-guide.pdf)
first; its source and build instructions are adjacent. The old research is preserved in
[the RLM checkpoint](../../docs/RESEARCH_RESUME_2026-10-04.md).

## Actual progress at 17:28 UTC

Execution access is restored. A real GPU pilot completed 1,200 decisions,
9,600 training simulator steps, 200 updates, and six evaluation episodes in
about 30 seconds, excluding startup and its final checkpoint write. It saved a
366 MB full-state checkpoint. This checks execution, not reproduction of the
paper's learning result. The fresh reference then completed 100k steps, improving
its ten-episode mean return from 8.44 to 678.02. One trained seed does not establish
the published average or a CURL advantage. The control is now running.

The reference and control use the official agent unchanged, with our thin
simulator, evaluation and checkpoint adapters. The runtime is Python 3.12.12,
Torch 2.13.0+cu130, dm-control 1.0.47 and MuJoCo 3.14.0. This is a modern-stack
compatibility reproduction, not an exact reconstruction of the 2020 stack.
See [manifest.json](manifest.json), [requirements.lock.txt](requirements.lock.txt),
and the [running findings](../../docs/curl-replication-2026-10-04/FINDINGS.md).

Historical access issue, now resolved:

At 16:03 UTC on October 4, this managed shell had no `/dev/nvidia0` or
`/dev/nvidiactl`. `nvidia-smi` could not communicate with the driver. Shell
GitHub access failed through the configured proxy. The accessible RLM Python
3.11.14 environment had no PyTorch. A supported GitHub read connector did work
for reviewing the upstream source. This is not evidence of an allocation-wide
GPU failure or proof that the host GPU is idle.

That earlier context also restricted external writes and Git operations. The
user restored execution access; the new environment is isolated, and unrelated
environments/processes remain untouched. The live account quota at 17:12 UTC
reported 98% remaining. Preserve the user's 10% reserve.

## The first scientific comparison

**Reference target:** CURL on DeepMind Control Suite `cartpole/swingup`, from
pixel observations. Reproduce the 100,000 environment-step endpoint first;
500,000 is a later extension. Table 1 reports 582 ± 146 and 841 ± 45 respectively,
as mean and standard deviation over ten training seeds. Those are the authors'
results, not ours, and the standard deviation is not an acceptance threshold.

**Mechanism question:** Does adding CURL's contrastive objective improve learning
beyond keeping its random-crop input augmentation? The control retains the
same architecture, replay sampling, crops, SAC updates, task and budget, but
skips `update_cpc`. This is OUR matched ablation. It is not automatically the
paper's differently configured Pixel SAC baseline.

Use authors' code first, then independently implement the bilinear contrastive
loss and verify fixed-batch logits, loss, gradients and one update. Final reward
alone is too weak a test of implementation agreement.

## Ordered queue

The pilot and reference seed 123 are complete. The matched control is active,
followed by the additional prespecified seeds. Native artifacts live in
`/project/alex_phd/runs/curl-replication-20261004`. The queue launcher records
completion/failure separately; it is not a substitute for interpreting results.

1. **Runtime pilot, seed 123:** render real observations, collect through 1,200
   agent decisions, verify finite rewards, nonzero parameter updates, evaluation,
   and save/reload. Cap at 30 minutes. It is a smoke test, excluded from seed means.
   Check real transitions promptly; after warm-up check loss and parameter changes.
   Completed pilot used two evaluation episodes every 600 decisions to reduce
   its cost and deliberately exercise mid-episode evaluation isolation.
2. **Reference run, seed 123:** a fresh start to 100k environment steps. Cap at
   two hours excluding one-time installation. If incomplete, retain the checkpoint
   and label the run interrupted, not a bad score or a completed reproduction.
3. **Matched no-contrastive control, seed 123:** same budget and two-hour cap.
4. **Repeat both arms, seeds 456 and 789:** only after the first pair is valid.
   Report all three seeds; do not stop because an attractive difference appeared.
5. **Extend both arms to 500k:** conditional on available allocation and throughput,
   not the sign of the first result. First establish that the existing comparison
   is interpretable. A second GPU is useful for another seed/control, not required
   to fit this first task. More tasks come after this reference works.

Before execution, read the actual allocation deadline. Do not assume the historical
48-hour lease or an unconfirmed two-GPU request is active. Start only jobs that can
checkpoint within the remaining allocation. Measure speed in the pilot; the
two-hour cap is a planning limit, not an A100 runtime prediction.

## Freeze these choices before the first learning run

The full numerical starting configuration is in `manifest.json`. In particular:

- Count simulator steps and model decisions separately. At action repeat 8,
  100k environment steps correspond to 12,500 decisions; 500k to 62,500. Count
  actual executed repeats if an episode ends early. Evaluation steps are separate.
- Keep the upstream 1,000-decision random-action warm-up, one SAC update call
  per subsequent decision, batch 128, and three-frame image stack. The warm-up
  is included in the training interaction budget.
- Evaluate at step zero, every 500 training decisions and the exact terminal
  endpoint. Use ten episodes per checkpoint, center crops, deterministic actions,
  no parameter updates, and a separate evaluation environment. Record the actual
  evaluation seeds. Proposed fixed seeds: 10000 through 10009.
- Evaluation must not consume the training RNG stream. Preserve and restore
  Python, NumPy, Torch and CUDA RNG states around evaluation. This and the separate
  environment are documented evaluation-isolation changes from upstream.
- Score each episode by summing raw rewards, including rewards accumulated across
  repeated actions. Then average the ten episode returns. Do not divide the score
  by action repeat. Check the installed wrapper's reward accumulation and episode
  length against the pinned source before interpreting a paper comparison.
- Each training seed is one replicate. Ten evaluations of one policy are not ten
  independently trained policies. Plot individual seed curves and an across-seed
  summary. Use the fixed endpoint, not the best evaluation encountered.
- Keep the control's unused positive-crop sampling to avoid an unnecessary RNG
  difference. Same seeds do not guarantee identical trajectories after policies
  diverge. This is an equal-interaction-budget comparison, not equal wall time.

## Small implementation issues that matter scientifically

**Resume is not provided by `--save_model`.** The training loop invokes
`save_curl`, which writes the contrastive module, not a complete acting policy
and training state. Upstream `save()` separately writes actor and critic weights,
but still omits optimizers, temperature, replay state and random generators.
The replay saver uses a contiguous slice and needs care across buffer wraparound.

Add full checkpoints without altering the learning update: policy, critics and
targets, CURL parameters, temperature, all five optimizers, replay arrays and
indices, random states, counters and configuration. Preserve shared parameter
ties. Test that a saved fixed-batch next update matches the uninterrupted update.
For a true trajectory continuation, simulator state and frame history must also
be restored. Otherwise explicitly resume at an episode boundary and label that
limitation. Avoid silently restarting an unfinished episode.

Checkpoint about every 15 minutes at an episode boundary, plus terminal and
pre-deadline checkpoints. Saving only filled replay entries reduces early-run
cost. Full-capacity pixel arrays alone occupy about 18 GB of host memory/storage
(two arrays of 100000 × 9 × 100 × 100 bytes); measure checkpoint I/O before
choosing cadence. Keep checkpoints outside Git in the external run store once
that store is writable. Never treat weights alone as a complete resumption.

**Evaluation shares the training environment upstream.** The loop evaluates
before its episode-reset branch. At mid-episode evaluation this can leave the
saved observation inconsistent with the environment. Our proposed 500-decision
schedule may align with normal cartpole boundaries, but a separate environment
is the clearer safeguard. Record the patch rather than claiming byte-identical
execution of the historical script. This is a source-level risk, not an observed
local simulator failure.

**Two optimizers own the query encoder in the contrastive update.**
`encoder_optimizer` covers `critic.encoder`; `cpc_optimizer` covers the entire
`CURL` module, which contains that same encoder. Both step after the same backward
pass. Preserve this in the first upstream-code reference, measure parameter
identity/update behavior on a real Torch batch, and document it. Do not silently
replace it with a textbook single optimizer or claim a novel bug from static
inspection alone. The actor and critic share convolutional weights, not their
entire encoder heads; the actor's detach flag stops gradients at convolutional
features. This matters for an independent implementation.

**The stack is old.** Upstream specifies Python 3.6/CUDA 9.2 and floating Git
dependencies. A modern simulator or PyTorch port is a compatibility adaptation.
Pin and record exact versions, renderer, wrapper commit and every patch. The
README's million-decision quickstart is not the 100k-step paper protocol; its
shell script also selects GPU 5. Use the allocation's assigned visible device.

## Minimal artifacts and analysis after each run

Store immutable config and source hashes, dependency lock, hardware/allocation,
start/end times, exit reason, simulator counters, training seed and evaluation
seeds. Keep JSONL evaluation records with one row per episode, update diagnostics,
checkpoint index and restart history. Archive a few representative videos, not
every frame. Record failures separately from scientific scores.

After each completed pair: state what changed, one competing explanation, and
the smallest follow-up that could distinguish it. Update the guide with actual
curves only after tracing each point to its episode records. Missing runs stay
missing, not zero. No favorable checkpoint selection on a reused evaluation set.

## Execution and resumption

Read the external `SESSION_CHECKPOINT.md` before launching anything. Check the
current owner and native `metrics.jsonl` results. Never launch a second process
into an existing live run. Each job uses a sealed source copy and the campaign's
`GPU.lock`. The allocation ends October 7 at 15:36:06 UTC.

`train_reference.py --help` lists the driver options. Set `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl`. The dependency capture includes Torch's CUDA 13 wheel;
rebuilding needs the PyTorch cu130 index, not just the default Python index.
The original clone is commit `8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc`.

`--max-seconds` limits each invocation, not the cumulative time across resumes.
Natural episode-boundary checkpoints retain task and action-space random state;
a budget-truncated episode restarts on resume, which is explicitly recorded.

The serial queue is running in tmux `curl-20261004-queue`; its external config
is `QUEUE.json` in the artifact root. Inspect `queue.jsonl` and per-run native
metrics, not just process presence. Create `STOP` in that root to ask its owner
to finish/checkpoint the current episode and stop. Do not edit its sealed code.

A separate `review_monitor.py` observer uses the installed `codex queue`
command to request review in this exact session. It neither trains models nor
interprets scores itself. It coalesces completed-run events for five minutes,
otherwise requests review after twenty minutes, and waits for an evidence-backed
acknowledgment before requesting another. It reads the live account quota and
requires more than 12% remaining to protect the 10% reserve. Its own `reviews/STOP`
stops review requests without stopping the scientific queue. Admission is not
proof of delivery; `reviews/STATUS.json` records that distinction. Both processes
are bounded by the current allocation deadline.

The pedagogical document distinguishes execution checks from learning results.
Do not treat a completed pilot as evidence that CURL beats the control.
