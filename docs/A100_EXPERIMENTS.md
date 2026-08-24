# A100 Experiment Quickstart

This guide keeps model training and serving on the GPU host while preserving RLM's small,
Responses-only boundary. The recommended first setup is **not** an in-process Transformers
adapter: serve the Hugging Face checkpoint with vLLM, then point RLM at the loopback endpoint.
This makes checkpoint swaps, LoRA experiments, request capture, and direct/RLM comparisons
independent of the training environment.

## 1. Clone and install

```bash
git clone https://github.com/queelius/rlm.git
cd rlm
uv sync --extra dev
uv run pytest
```

Use separate environments for RLM, vLLM, and training when their CUDA or PyTorch constraints
differ. RLM itself does not depend on PyTorch or Transformers.

## 2. Serve a checkpoint

Install a current vLLM release in its own environment, then serve the base model or merged
checkpoint. The model name supplied to requests must match `--served-model-name`.

```bash
vllm serve /absolute/path/to/checkpoint \
  --served-model-name rlm-model \
  --host 127.0.0.1 \
  --port 8001 \
  --api-key local
```

Verify the exact API RLM requires before starting an experiment:

```bash
curl --fail-with-body -sS http://127.0.0.1:8001/v1/models \
  -H 'Authorization: Bearer local'

curl --fail-with-body -sS http://127.0.0.1:8001/v1/responses \
  -H 'Authorization: Bearer local' \
  -H 'Content-Type: application/json' \
  -d '{"model":"rlm-model","input":"Reply with exactly READY.","max_output_tokens":32}'
```

A server exposing only `/v1/chat/completions` or a provider-specific `/generate` route is not
compatible. Current vLLM documents its Responses routes at
<https://docs.vllm.ai/en/latest/serving/online_serving/>.

If RLM runs on another machine, keep vLLM loopback-only and forward it securely:

```bash
ssh -N -L 8001:127.0.0.1:8001 your-a100-host
```

## 3. Run RLM and smoke-test Python execution

```bash
uv run rlm serve \
  --host 127.0.0.1 \
  --port 8000 \
  --upstream-base-url http://127.0.0.1:8001/v1 \
  --upstream-api-key local \
  --controller-model rlm-model \
  --controller-option temperature=0 \
  --controller-option max_output_tokens=2048 \
  --max-parallel-model-calls 1 \
  --max-depth 0 \
  --max-total-tokens 20000 \
  --deadline-seconds 600 \
  --trace-dir runs/a100-smoke \
  --trace-markdown
```

In another shell:

```bash
curl --fail-with-body -sS -D /tmp/rlm-a100.headers \
  http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rlm-model",
    "temperature": 0,
    "input": "Compute the SHA-256 hex digest of the UTF-8 string formed by RLM-A100-smoke: followed by exactly 10000 lowercase x characters. Use Python and return only the digest."
  }'
```

Require HTTP 200, a completed Responses object with non-empty output text, the expected
`X-RLM-*` headers, and a canonical trace showing request inspection, Python execution, and an
explicit final submission. Do not treat protocol completion alone as task correctness.

## 4. Rapid SFT loop

For a single A100, the simplest loop is sequential:

1. Serve a checkpoint and collect fixed-harness RLM/direct trajectories.
2. Join each episode to an external verifier; retain verified successful trajectories only.
3. Export controller-role `model.request`/`model.response` pairs by call ID.
4. Stop vLLM to release GPU memory.
5. Train a LoRA adapter with TRL `SFTTrainer` and PEFT.
6. Restart vLLM with the new adapter or merged checkpoint.
7. Re-run the same held-out, paired, fixed-budget evaluation.

Use the checkpoint tokenizer's exact chat template during both export and inference. For training,
Hugging Face recommends applying the chat template without a generation prompt; avoid duplicating
special tokens. See <https://huggingface.co/docs/transformers/chat_templating> and
<https://huggingface.co/docs/trl/sft_trainer>.

Never use `run.completed` as a reward label: it proves protocol success, not answer correctness.
Record at least the base checkpoint digest, adapter digest, tokenizer revision, chat-template
digest, dataset/split identity, harness fingerprint, controller options, sampling configuration,
budget, verifier identity, and code revision for every run.

## Current boundaries

- RLM accepts a custom Python `ModelBackend`, but no native Transformers adapter ships today.
- The CLI uses one upstream endpoint for controller and public-model calls; model IDs may differ if
  that endpoint serves both. The Python API can provide a separate controller backend.
- Traces are sufficient for verified SFT export after retokenization. They do not yet contain token
  IDs, action masks, sampling metadata, or old-policy log probabilities required for on-policy RL.
- Generated Python is process-isolated but not sandboxed. Traces contain raw prompts and model
  output; treat both as sensitive.

