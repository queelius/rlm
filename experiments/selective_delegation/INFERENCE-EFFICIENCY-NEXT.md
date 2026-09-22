# Batching across crafting episodes: worthwhile bounded qualification

CPU-only inspection, 2026-09-22. No backend changes, installation, GPU launch, or
interruption of 057/061/062. Recommendation: prepare a small **future** batching
qualification, not a framework migration or reuse of historical HF baselines.

## Observed bottleneck

Completed `R/textcraft-trained-readout-001` has 1,884 native calls, 6,798,564
input and 35,475 output tokens. Summed native call intervals are 3,178.98 s
against 3,228.47 s owner wall time (98.47%); median call 1.581 s, p90 2.305 s.
Thus repeated summaries/outer orchestration cannot explain most of this run's
time. Each call averages 3,609 input versus 18.8 output tokens. Repeated full
history prefill and serial short generations are plausible batching targets;
these receipts do **not** isolate CUDA prefill/decode or prove GPU saturation.

Current `eval_textcraft.NativeClient.call` tokenizes each complete prompt and
calls HF `generate` with batch size one. It resets global Torch/CUDA seeds for
each request: concurrent threads around this client would not preserve the
existing per-request RNG contract. The telemetry snapshot reported MIG
7g.40gb, 35,094 MiB allocated and 113.74 W, but GPU/memory utilization was N/A.
Loaded memory is not evidence of useful utilization or permission to launch.

## Already available implementation evidence

The existing environment `/project/alex_phd/envs/prime-rl-5990b1b/bin/python`
currently reports vLLM 0.28.0, Torch 2.13.0+cu130, Transformers 5.6.2. PEFT
package metadata is absent there; do not mutate it or assume the HF trainer
environment is interchangeable.

Under `/project/alex_phd/runs/rlm-research-r4/sidecars/`:

- `root-qs6-fixed-helper-top20-batch-invariant-v1/outputs/attempt-003` has a
  complete/released owner, no errors, and a serving manifest for our exact
  cached Qwen3-4B-Instruct-2507 revision `cdbee75…`, rank-8 LoRA, context 8192,
  max 16 sequences. Its `service_batch_invariant_v4.py` is an existing service
  wrapper, not a crafting-ready client.
- `root-qs6-leaf-rloo-fresh-batch-invariant-qualification-v2/outputs/attempt-001`
  contains actual native responses and service logs showing four concurrent
  requests and about 128 generated tokens/s. This was a different, highly
  prefix-cached workload: **not a crafting speedup estimate**. Its later HF
  qualifier failed for missing `xgrammar`; no successful cross-backend
  equivalence qualification is claimed.
- `leaf-mixed-size-sft-v1/probe.py` has working semaphore-4 request collection
  and serving alias/base/adapter checks. Its six-request vLLM probe had zero
  transport failures but zero valid arrays. Transport readiness is not task
  competence. That probe explicitly notes FP32-HF versus BF16-serving LoRA
  differences. Its optional prompt-ID comparison can be unavailable, which is
  insufficient for a new strict native qualification.

## Smallest useful next test and stop rule

Use one fixed adapter and 32 outcome-independent saved crafting requests,
including short/long histories, for concurrency 1 versus 4 on the existing
vLLM path, plus a small HF reference. Cap qualification at 15 GPU minutes only
when independently accepted; abandon/defer if existing serving cannot expose
native input/output IDs without infrastructure work. Require exact prompt
tokens/chat template, model/tokenizer/adapter hashes and actual dtype, seed,
T=.5/top-p=1/top-k=0, EOS, context/no-truncation, output caps and request-response
identity. Report transport/protocol errors, token-ID agreement/disagreement,
native action parsing/scoring, wall throughput and latency—not only occupancy.
A prospective ≥1.5× throughput gain without receipt/contract failures would
justify a fresh small matched policy comparison; this is a decision threshold,
not a predicted speedup.

Then batch **across independent episodes**, retaining sequential feedback inside
each episode and global 96-call/8192-output-token accounting across tree nodes.
Keep native inventory transitions/root checker, invalid-action charging, and
unknown/capped slots unchanged. Do not mix mutable adapter activation between
concurrent calls. Prefix cache configuration and wall-cap censoring must be
declared. HF batching would instead require explicit per-sequence RNG,
left-padding/masks, EOS and budget handling; it is not achieved by adding threads.

Serving precision, attention kernels, sampling implementation and batch shape
can alter responses even with equal seeds. Batch-invariant kernel flags do not
establish HF equivalence. Recollect **both** policy arms under the selected
backend/schedule; do not splice a faster trained arm into an old HF base result.

## Inspection provenance

`R` denotes the selective-delegation-20260921 sidecar store. Source SHA256s:

- `E/eval_textcraft.py`: `1f712775e8f9ced8f500cda6b5c7eabfe0fd02ea9a31f85910e781f5363e580a`
- `leaf-mixed-size-sft-v1/probe.py`: `d7c425f64ce1084f59088f8d158b2f051e9c3bc130a8b33f47cf3ba8a662c5a7`
- `root-qs6-fixed-helper-top20-batch-invariant-v1/service_batch_invariant_v4.py`:
  `d3daa1bf91e22885205dfb1ca3f268f24d19f6a6da963f0004074a3afd5b865f`

This is a read-only readiness scout, not a qualified new inference backend.
