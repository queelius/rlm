# Restricted categorical controller: CPU feasibility draft

Status: design only, 2026-09-21. No trainer implemented or launched. Main waits for
the probe's action-value signal before implementation. All inspection here was CPU-only.

## Data and observation

Inputs are `inputs-001/cases.jsonl` under the external selective-delegation sidecar:
256 train, 64 validation, 64 four-hop transfer parents. Train has 198 two-hop and
58 three-hop parents; validation has 44 and 20. Atomic components do not cross
these splits. Collection may initially cover only 32 parents; report actual complete
coverage, never the prepared count as the training count.

Use an explicit allowlist to serialize question, paragraph id/title index, and the
actual shared checkpoint (`answer`, `evidence_question`, `subquestions`). Do not
include paragraph text, source IDs, opaque parent IDs, hop counts, reference
decomposition, gold answers, correctness, future reports, utilities, or action
outcomes. Paragraph IDs are legitimate public source references. The checkpoint's
provisional answer is model-generated observation, not the host gold.

Action mapping is fixed: A=finish, B=reconsider, C=targeted, D=decompose. Include
short action definitions in the prompt. CPU tokenizer verification gives bare
letter token IDs A=32, B=33, C=34, D=35. Leading-space variants differ
(362/425/356/422); never silently interchange them. Render the original chat
template with `add_generation_prompt=True, enable_thinking=False,
tokenize=True, return_dict=False`. Its assistant prefix ends with a newline.

Take the final-position logits for these four IDs and apply softmax over only
those four values, at temperature 1. This is an explicitly **restricted categorical
policy**, not ordinary unconstrained next-token generation. The model cannot choose
another string or emit reasoning. Report the same restriction for base, SFT and RL.
Use one unpadded example per microbatch initially; right padding would make the
last-position logits wrong unless true sequence ends were gathered.

## Targets and optimization

Freeze a host-side table indexed by parent, action and continuation seed. Require
all four actions and known usage for each retained seed. Invalid returned answer
JSON has EM=0; missing transport usage is unknown, not zero-cost failure. Report
excluded/incomplete parents separately. Never forward the table through a prompt.

For each table entry use
`u(s,a,k) = EM(s,a,k) - 0.02 * (incremental_prompt_tokens + incremental_completion_tokens) / 1000`.
Incremental means helper plus final for B/C/D and final for A; exclude the common
checkpoint. Average over the fixed training continuation seeds to obtain `q(s,a)`.
Charge common checkpoint and actual router prompt plus one output token when
reporting deployed policy costs. They are common to the four choices at this
checkpoint and are not action-specific decision penalties. Also report collection
and training compute separately, including already acquired counterfactual calls.

SFT: target `argmax_a q(s,a)` and optimize restricted four-way cross-entropy.
Break exact utility ties by lower mean cost, then fixed action order. Preserve
utility margins and disagreement over seeds; a three-seed winner is a noisy label.

RL: draw `a ~ Categorical(pi_theta(.|s))` freshly for every visit. Draw one stored
continuation seed uniformly for its reward. Use the action-independent state
baseline `b(s) = sum_a pi_theta(a|s).detach() * q(s,a)` and loss
`-log pi_theta(a|s) * (u(s,a,k)-b(s)).detach() - 0.01 * entropy(pi_theta(.|s))`.
The baseline and reward must be detached. Sampling the action from the current
policy makes this a genuine policy-gradient update on the frozen empirical table;
it is **offline contextual-bandit training**, not fresh environment rollouts or
end-to-end RLM RL. No importance ratio is needed for these freshly sampled actions.
Do not describe replaying old selected actions under changed logits as equivalent.

Compare untrained restricted router, always-finish, each fixed arm, the best fixed
arm chosen on training only, a frozen observation-only heuristic, SFT, and RL.
For the first clean comparison, both learned policies start from the same base and
fresh same-seed zero-effect LoRA; do not warm-start RL from SFT without naming that
additional condition. Propose LR=1e-4, AdamW weight_decay=0, gradient clip=1,
microbatch=1, 16 parents/update, and four fixed passes through complete training
parents, saving a curve after each pass. Repeat promising learning with a second
training seed rather than selecting a lucky run. With 256 parents this is 64 updates.

Use validation only for declared model-selection decisions, then collect fresh
continuation seeds on held-out parents. Four-hop transfer remains unopened until
training decisions are fixed. When inspecting several epochs, present all points;
do not treat selected validation performance as untouched test performance.

## Existing runtime and loading recipe

Use the existing successful training Python without mutating it:

`/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python`

CPU inspection: Python 3.12.12; torch 2.13.0+cu130; transformers 5.15.1;
PEFT 0.20.0; safetensors 0.8.0; accelerate 1.14.0; NumPy 2.5.2. Its sibling
`pyproject.toml` and `uv.lock` exist. Record their hashes and observed versions in
the eventual run; do not run `uv sync` against this shared existing environment.
The Prime inference environment has no PEFT and transformers 5.6.2, so it is not
the training interpreter.

Base model, immutable revision:
`/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554`.

Reuse the successful loading configuration from
`sidecars/openai-mrcr-procedural-sft-warmstart-v1/train.py`: local-files-only
`AutoModelForCausalLM.from_pretrained`, BF16 base, `attn_implementation='sdpa'`,
`device_map={'':'cuda:0'}`, `trust_remote_code=False`; set `use_cache=False`.
Apply `LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',task_type='CAUSAL_LM',
target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])`
with `autocast_adapter_dtype=True`. Enable nonreentrant gradient checkpointing and
input gradients; verify only FP32 LoRA tensors are trainable. This architecture
has 504 trainable LoRA tensors and 16,515,072 LoRA parameters.

The installed Qwen forward explicitly supports `logits_to_keep=1`. Call without
labels, take `out.logits[0,-1,[32,33,34,35]].float()`, and compute the categorical
loss yourself. This avoids producing sequence-length by vocabulary logits.

CPU qualification commands can use `CUDA_VISIBLE_DEVICES=''` followed by the
training interpreter. Future GPU launch must be through the main-owned coordinator
after inference releases the exclusive lock, with its assigned visible device;
there is no runnable trainer command yet. Preserve the same owner deadline and
allocation-end-minus-ten-minutes limit. Do not launch HF beside the live vLLM owner.

## Checkpoints and practical limits

Reuse the checkpoint pattern, not the old experiment-specific verifier:
`openai-mrcr-procedural-sft-warmstart-v1/train.py:save_checkpoint` and
`openai-mrcr-procedural-sft-continue32-v1/resume.py:restore_state`.
Each committed checkpoint needs adapter safetensors/config, optimizer state,
Python/NumPy/CPU/CUDA RNG states, exact parent order/cursor/epoch/update,
objective/hyperparameters, prompt/tokenizer/action mapping, source/table hashes,
and a final hash receipt written after all files. Restore optimizer moments,
parameter order, RNG and cursor before continuing; do not restart Adam at resume.
If using a separate torch sampling generator, save its state too. Save only after
complete optimizer steps, and honor interruption at these boundaries.

Prior successful Qwen4B rank8 continuation reported 13,662,985,728 bytes peak
allocated and 967.5 seconds for updates 5 through 32. This is evidence that this
model/recipe fits one A100, not a prediction of this run's measured footprint.
BF16 weights are roughly 8 GB; FP32 adapter weights/gradients/Adam moments add
roughly 0.27 GB before activations and allocator overhead. The new short title-index
observations should be cheaper than old long-input SFT, but measure peak allocated,
peak reserved, input length and throughput on the first real update.

Main failure modes: overfitting 32 parents; three-seed utility noise and winner's
curse; useful distinctions absent from the compressed controller observation;
base-model letter priors; action-cost correlations mistaken for reasoning gains;
fixed checkpoint quality limiting all branches; entropy overpowering tiny utility
differences; unknown usage silently treated as zero; and inconsistent tokenization
between inference and training environments. A single promising pilot should enable
the planned training-data expansion, not establish a learned selective policy.
