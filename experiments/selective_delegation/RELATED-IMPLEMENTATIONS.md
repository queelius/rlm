# MetaAgent-X: code-level implications for our frozen-helper experiment

Read-only inspection, 2026-09-21; not an implementation reproduction or a new GPU
job. This supplements, rather than replaces, [LITERATURE.md](LITERATURE.md).

## Provenance

[Paper v1](https://arxiv.org/html/2605.14212v1) links the
[official repository](https://github.com/pettingllms-ai/PettingLLMs).
Inspected commit: `a054fb18b83f4dc7c1d49aba3b71669d1f93ca71`
(commit date 2026-05-14). Retrieved by shallow clone at
2026-09-21T12:25:49Z into
`/project/alex_phd/research-cache/repos/PettingLLMs-inspect-20260921`.
Root license: MIT, copyright 2025 PettingLLMs-AI. No installation, model download,
submodule initialization, LFS execution, or downloaded-code execution occurred.
Despite a `.gitmodules` entry, `verl/` is a tracked tree at this commit, not a
gitlink; the worker/GRPO files inspected below came in that same checkout.

## 1. Repeated execution is implemented, but its grouping and denominator matter

The actual [tree rollout](https://github.com/pettingllms-ai/PettingLLMs/blob/a054fb18b83f4dc7c1d49aba3b71669d1f93ca71/pettingllms/trainer/multi_agents_execution_engine_autoevol.py#L1237)
samples designs once, executes each several times, and emits **one designer
trajectory per design** with its mean execution reward—not one designer copy per
execution. Workflow response trajectories each receive their execution's reward.
The [UID assignment](https://github.com/pettingllms-ai/PettingLLMs/blob/a054fb18b83f4dc7c1d49aba3b71669d1f93ca71/pettingllms/trainer/multi_agents_ppo_trainer.py#L1605)
groups designer rows by question. The public launch script explicitly groups
executor trajectories across all designs for that question. Its `question`
override matters: leaving `executor_group_mode` at its automatic default selects
within-design groups when executions-per-design exceeds two. Thus the engine
docstring's within-design description does not describe the public launch.

This is trajectory-level grouping: an execution producing more workflow response
rows can contribute more entries to the executor group's mean/std. The inspected
GRPO implementation normalizes per UID over those rows. Default loss aggregation
is token-mean, PPO clipping is0.2, and PPO epochs is1; this is not our unnormalized,
root-token-sum, fixed64-denominator RLOO objective.

Two further differences should survive any port: unexpected exceptions escaping
a design/execution task are skipped before the designer mean is formed; ordinary
execution timeouts caught inside the executor instead return zero reward. A mean
can therefore use fewer than the requested executions. Also the actual reward
adds up to0.8 for delivery/solution formatting to correctness. Neither behavior
matches our transport-failure halt and terminal-EM-only reward.

Actionable diagnostic, conditional on the current full-pass readout: freeze the
first batch's16 training parents and four saved SFT48 plans each; use the selected
helper-SFT36 and base final for four fresh execution seeds, shared across plans
within each parent. Compare choosing a plan from one seed versus the mean of
three, evaluated on the excluded fourth; rotate exclusions. This asks whether
averaging improves out-of-seed selection, not whether the maximum observed mean
is an oracle. Keep parse/dependency failures in planned slots; stop on transport
failure rather than treating unavailable execution as semantic failure. Record
parent-paired differences and native cost. Approximately768–1024 calls for two-
or three-step plans; at most2304 with eight-step plans, under a predeclared
one-A100-hour cap. An incomplete pilot is inconclusive, not a completed estimate.
Promote averaging only for useful held-seed gains relative to its added cost;
otherwise prefer more distinct training parents. No GPU job is accepted here.

## 2. An inactive shared-weight role is not an independently frozen policy

The [public training script](https://github.com/pettingllms-ai/PettingLLMs/blob/a054fb18b83f4dc7c1d49aba3b71669d1f93ca71/scripts/train/autoeval/example_cotrain_autoeval.sh)
defaults to Qwen3-1.7B, eight GPUs, full-parameter training (`lora_rank=0`), batch8,
four designs by four executions,400 steps,8K prompt/8K response limits, and
designer/executor learning rates1e-9/5e-6 swapped every10 steps. Both role
sub-batches still invoke `update_actor` on the same policy/optimizer; the worker
temporarily overrides optimizer LR, updates, then restores LR. This is soft role
weighting, not gradient masking or a frozen inactive model. Even zeroing one
role's LR would not prevent the other role's updates changing its behavior.

The paper's shared-weight stage description likewise updates shared parameters
from active-role trajectories; freezing inactive-role data is not freezing its
weights. Its reported schedule is30-step stages, not the public example's10.
A separate `math_L1_iterated_br.yaml` uses two policy instances,30-step stages,
executor first, and both LRs5e-6. In that path the trainer resolves the inactive
policy and skips its update entirely. That is the code path closest to genuine
stagewise policy freezing—not the public shared-policy example.

Actionable comparison requirement: keep our helper weights independently fixed
when estimating the incremental root effect. Any later shared-weight stage
experiment must measure inactive-role drift on fixed prompts, not label its
role-filtered updates a frozen-executor control. Our current separate helper
model and disabled-adapter base final provide a cleaner intervention, but do not
replicate MetaAgent-X co-evolution.

## What a faithful one-A100 comparison would require

The reported paper uses eight H200s and a workflow-code cold start; its full-scale
setup is not a one-A10040GB baseline. A reduced implementation comparison would
need explicit deviations: sequential four-by-four rollouts; smaller batch and
context limits; LoRA instead of the public full-parameter default; cold-start
compatibility with the executable workflow contract; and safe execution of
generated programs. Preserve the selected grouping, trajectory reward
assignment, role-stage schedule, PPO/GRPO normalization/clipping and compute
accounting if calling it a method reproduction. At least a complete executor
and designer stage is needed to test alternation; four updates cannot do so.
Full-parameter optimizer memory, vLLM/FSDP residency and dynamic workflow lengths
make a one-GPU fit unverified. A LoRA rewrite or our fixed-question-list adaptation
must be labeled as such, not advertised as an official configuration run.

The minimal frozen-plan diagnostic above needs none of this migration. It tests
one claimed mechanism without changing our accepted RL runner. Training planners
against outcome rewards, executor-first adaptation, stagewise learning and
hierarchical repeated rollouts are prior art. Our potential contribution must
instead be evidence about when extra delegation/repetition is worth its cost,
with role interventions, protocol/content separation and transfer—not a claim
that planner–executor learning itself is new. Static inspection does not establish
the released framework's runtime correctness or reproduce its reported gains.

## Selected SHA-256 receipts

Paths are relative to the external checkout above; its commit pins the rest.

| File | SHA-256 |
|---|---|
| `LICENSE` | `aaa8916c62d70472845645334afeee02f9e08e2ba8c05d47565c66adb1ea9bec` |
| `scripts/train/autoeval/example_cotrain_autoeval.sh` | `2fc5bd95ddecfd71f9da434da954e2acddc0b10f67496ce7c59561c0eeb4800a` |
| `pettingllms/config/autoevol/math_L1_prompt.yaml` | `a0b8262370644623d77374b5cae4f4c4513a863ca5c97a9a253a5228690c8a08` |
| `pettingllms/config/autoevol/math_L1_iterated_br.yaml` | `c99a8285cebf84069980f7916f8581306883c06b5df0df578a6f5973818c9c51` |
| `pettingllms/trainer/multi_agents_ppo_trainer.py` | `b9404938041348fb1754fb44407cef6a406c52a6adc598d186e83dc666cfad28` |
| `pettingllms/trainer/multi_agents_execution_engine_autoevol.py` | `89827ffa3439b83c595f85de7c131b0ec3db41bc424642582eba7de293fb3153` |
| `pettingllms/multi_agent_env/autoevol/gen_agent.py` | `aefaa7caaee6308543de814a2c60fe8ca5c6b144d2e7225fca844d05a695fa03` |
| `pettingllms/config/ppo_trainer/eval.yaml` | `24bc3f9dc8e516ca639fbc2f08c1f6c770b0eb2a61ceb43af56e400afc525d0f` |
| `verl/verl/workers/fsdp_workers.py` | `e881eb851895c07352934f55c26aed868b33b13016fd970963a1f9b41c0eb087` |
| `verl/verl/trainer/ppo/core_algos.py` | `3e53df184d8b4c1603558bf8b4dff99b3180b86491d4b7ba09d0850f98f02d52` |
