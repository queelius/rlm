# SPO native environment and public model receipt

Prepared on 2026-10-08 UTC for GPU owner `/root`; this worker launched no GPU process.

## Native executable paths

- Python: `/project/alex_phd/envs/spo-6c5f947/bin/python`
- DeepSpeed: `/project/alex_phd/envs/spo-6c5f947/bin/deepspeed`
- Source: `/project/alex_phd/research-cache/repos/SPO`
- Source revision: `6c5f94723c20e4de68578545b8bb98186e956797`
- Source URL: `https://github.com/AIFrameResearch/SPO`
- Source license: MIT (repository LICENSE; copyright 2024 McGill NLP).
- Environment inputs: `environment-requirements.txt`; resolved, SHA256-pinned packages: `environment-lock.txt`.
- Lock SHA256: `75e5e9cc7dd294b380624b0a9ceabf8c3484ad0dc2e76a1708242454560b97a4`.
- Authors requirements SHA256: `c8f4d47025037c26496eb0187a24fd1860483da9cebe4089a39dabe318f777d0`.
- Authors Dockerfile SHA256: `1a7d6ed1dbef74d2c39999a09a92e45fbd896d662efc4cbea5fbe4027c47cde1`.

Python 3.10.19; uv 0.12.9; Torch 2.1.2+cu121; packaged CUDA 12.1;
host nvcc 12.5.82; FlashAttention 2.5.5; DeepSpeed 0.14.1; transformers 4.38.1;
datasets 2.17.1; authors' custom VinePPO vLLM 0.4.0.post1; xformers 0.0.23.post1.
Exact remaining package versions and distribution hashes are in the lock.
The environment is new and outside the clone. The existing R1 environment was
used only for its Python interpreter and public download libraries, without mutation.

Authors' direct pins were preserved. Additional explicit constraints are pyarrow
15.0.2 (datasets 2.17 uses PyExtensionType), pydantic 2.0.3 (intersection of
vLLM >=2.0 and redis_om <2.1), and setuptools 69.2.0. Their unpinned `lighteval`
was omitted because it is for separate long-CoT evaluation; native GSM8K does not
use it. Attempts with pydantic 1.10.13 and 2.6.4 were rejected by dependency
resolution before installation; no runtime result is implied by those attempts.

## Checkpoints

Base:
`/project/alex_phd/research-cache/spo/models/models--realtreetune--rho-1b-sft-GSM8K/snapshots/b28fda3216f8178c1f37c3c764196e3ca2439a30`

Author SPO-chain-int5:
`/project/alex_phd/research-cache/spo/models/models--gyr66--spo-chain-int5-rho1.1B-gsm8k/snapshots/a806ef8ef83beca009725da026fb7bde467a9dde`

Their source URLs, immutable revisions, acquisition dates, file sizes and SHA256
checksums are in `realtreetune--rho-1b-sft-GSM8K.json` and
`gyr66--spo-chain-int5-rho1.1B-gsm8k.json`. Both snapshots completed.
Neither model card declares a license; receipts deliberately record null.
The base card names microsoft/rho-math-1b-v0.1 as ancestry. Neither model download
contains executable remote-model Python; configs identify native LlamaForCausalLM.
Training split provenance and GSM8K saved dataset acquisition are owned by the
other preparation worker, not inferred here.

Use `HF_HUB_CACHE=/project/alex_phd/research-cache/spo/models` or the full snapshot
paths above. Native Jsonnet initial_model_name_or_path should point to the base
snapshot to seal the revision; `--last_policy_path` should point to the author
snapshot for checkpoint evaluation. Set PATH to the environment bin directory so
the native vLLM shell subprocess also uses this environment.

## Verification and source inspection

Completed CPU-only verification with CUDA_VISIBLE_DEVICES empty:

- `uv pip check`: all 160 installed packages compatible.
- Native `src/treetune/main.py --help`: exit 0.
- Imported torch, flash_attn, deepspeed, transformers, datasets, vllm, and
  `treetune.runtime.Runtime`: exit 0, versions as above.
- Custom vLLM OpenAI server `--help`: exit 0.
- Both checkpoint AutoConfig/AutoTokenizer loaded locally without remote code:
  exit 0; native llama model type and identical eleven-token simple input.

`environment.sh` contains the cache/PATH/runtime settings to source before the
native launcher. It creates only task-scoped cache and W&B directories, and
leaves experiment directory, experiment name, GPU visibility, and launcher port
to the owner. Set APP_OPENAI_VLLM_API_BASE=none even before Jsonnet resolution.
The authors' guidance config supplies api_key='EMPTY'; no real OpenAI key is
needed. Native runtime/guidance has no Redis call. W&B is offline; evaluation
does not require cloud login. The named vLLM shell forces BF16. Keep
disable_sliding_window and disable_frontend_multiprocessing false: the authors'
vLLM wheel does not accept those newer flags according to its CLI help.

Inspected authors Dockerfile, requirements, README, entrypoint, runtime import
graph, vLLM process startup/cleanup, dataset shell script, license, and model
configs/cards before native code execution. The shared source clone was not edited.
GPU was idle (1 MiB) during initial read, communicated promptly to parent.

The authors' dataset shell script contains broad copy/delete staging commands and
was not executed by this worker. VLLMServer cleanup has pkill -f -9 and a
port-process kill helper (the calls to that helper are commented out). The active
pkill pattern targets vllm.entrypoints.openai.api_server.*port followed by this
server's port; use a unique owner port. Host CUDA differs from the
authors' container, so optional JIT-built optimizer kernels remain a runtime
compatibility question; imports alone do not establish successful training.
