# rlm

`rlm` is a small, request-preserving Recursive Language Model runtime for the
non-streaming OpenAI **Responses** API. A private controller writes exactly one
Python cell per turn in a persistent IPython process; that cell can inspect the
unchanged public request, use the versioned helper ABI, and submit a final
Responses result.

The public surface is deliberately narrow:

- `GET /v1/models` is passed through to the configured upstream.
- `POST /v1/responses` runs the RLM loop.
- Chat Completions, streaming, and background Responses are unsupported.
- `RLM.run_direct(request)` is the explicit, attested direct-model baseline. The RLM loop
  never falls back to it, retries a request, or substitutes an alternate answer.

## Install

```bash
uv sync --extra dev
uv run pytest
```

For a GPU-hosted Hugging Face workflow, including vLLM serving, an A100 smoke test, and a
trace-to-LoRA iteration loop, see [A100 Experiment Quickstart](docs/A100_EXPERIMENTS.md).

## Library usage

Configuration is nested. `HarnessSpec`, limits, execution, and trace structure
are frozen; `ControllerConfig.options` is intentionally mutable between runs.
Each run takes one owned deep snapshot before its first model call, so later
caller mutations cannot alter that run. `HarnessSpec` defines model-visible,
fingerprinted behavior—including the typed first-turn bootstrap contract—while
controller selection remains a separate experiment control.

```python
from rlm import (
    ControllerConfig,
    ExecutionConfig,
    HarnessSpec,
    RunAttestation,
    OpenAIEndpoint,
    RLM,
    RLMConfig,
    RunLimits,
    TraceConfig,
    load_trace,
    response_output_text,
)
from rlm.prompts import default_harness_spec

harness: HarnessSpec = default_harness_spec()
config = RLMConfig(
    harness=harness,
    limits=RunLimits(max_turns=12, deadline_seconds=600, max_depth=0),
    controller=ControllerConfig(model="controller-model", options={"temperature": 0}),
    execution=ExecutionConfig(working_directory=None),
    tracing=TraceConfig(enabled=True, directory="runs", markdown=True),
)
rlm = RLM(OpenAIEndpoint(base_url="https://api.openai.com/v1"), config=config)

request = {"model": "public-model", "input": "Explain recursion in one paragraph."}
run = rlm.run(request)
assert run.response["object"] == "response"
assert response_output_text(run.response)
attestation = RunAttestation.from_result(run)
trace = load_trace(run.trace_directory, expected_run_id=attestation.run_id)

# An attested evaluation baseline, never an automatic recovery path:
direct_run = rlm.run_direct(request)
```

`RLM.complete(request)` returns only the completed Responses object.
`RLM.run(request)` additionally returns the run ID, aggregate reported usage,
turn count, controller model/options identity, harness fingerprint, duration,
and optional trace directory.

Python orchestration is preferred for the first A100 experiment because
`RunResult.trace_directory` names its artifact unambiguously. `RunAttestation`
provides the same typed execution identity for in-process and HTTP consumers,
while `load_trace` verifies the manifest, JSONL bytes, schema, causal graph,
model-call outcomes, and terminal provenance before returning owned values.

## Kernel contract

The controller's first user item is a typed bootstrap schema-v2 envelope. It
declares the ABI-derived `request` binding, compact request metadata, and the
required final kinds: `text` plus `response` for ordinary requests, or only
`response` when a synthetic text response would lose request semantics. Caller
content is absent from the envelope; it is runtime metadata, never the task.
The controller determines the task from bound `request` by inspecting the
fields it needs or deliberately delegating the exact request with
`model_complete(request)`; neither path requires a separate host-mandated
inspection turn. Each controller response must contain one assistant
`output_text` item whose whole content is exactly one:

````text
```python
# one non-empty Python cell
FINAL_TEXT("answer")
```
````

No prose, extra fences, extra messages, or empty cells are accepted. Each
nonterminal turn receives a strict-JSON `rlm.controller_observation` schema-v2
item with explicit request binding, absent/required submission state, and a
bounded nested execution record. `execution.status: "ok"` means only that the
cell ran; `submission.status: "absent"` means no valid final was accepted. An
accepted final terminates without another observation. Printed or displayed
values are private and do not submit an answer. Configured observation budgets
must be at least 512 characters so every built-in repair identity can be shown.
The latest controller response retains its reasoning and message output items,
followed by this runtime item; IPython itself is durable working memory.

The versioned environment ABI provides `model_complete`, `ask`, their batch
variants, `FINAL_TEXT`, `FINAL_RESPONSE`, and `SHOW_VARS`; recursive helpers
are present only when the depth limit permits them. `FINAL_TEXT` synthesizes a
minimal completed text response only when that can faithfully satisfy the
public request. Its argument must already be a string; invalid values become a
typed repair observation and are never coerced. `FINAL_RESPONSE` accepts an
existing, fully validated Responses object.

`ask_batch` returns an `AskBatchResult`, with successful texts aligned to the
input indexes and failures retained separately. A controller can inspect
`failed_indexes` and call `require_texts()` when complete text is required;
there is no global pending-error flag and no automatic recomputation.

Malformed controller output, Python exceptions, empty strict-text helper
results, and invalid final submissions are typed controller faults. The
configured recovery policy returns a bounded observation and consumes a normal
turn (or explicitly aborts). Invalid public requests, malformed/failed
upstream responses, backend deadline failures, exhausted limits, host or
executor failures, and enabled-trace failures are fatal. Captured `stderr` is
ordinary output, not a failure signal.

Every backend call receives the remaining timeout and a typed call context
(`controller`, `public`, or `subcall`). Backends must honor that timeout; a
custom backend that blocks beyond it violates the contract. The shared ledger
also rejects calls that return after the run deadline.

## Server usage

```bash
uv run rlm serve \
  --upstream-base-url https://api.openai.com/v1 \
  --controller-model controller-model \
  --controller-option temperature=0 \
  --trace-dir runs
```

Controller option values are strict JSON. They cannot replace runtime-owned
model, instructions, or input fields. Successful proxy responses carry
`X-RLM-*` run metadata, including controller identity, harness fingerprint,
and aggregate usage; their body remains a normal Responses object. HTTP clients
can decode those case-insensitive headers with
`RunAttestation.from_headers(response.headers)`. No header exposes an absolute
path on the server filesystem; artifact transfer remains an explicit
orchestration responsibility.

## Traces and research use

With tracing enabled, JSONL is the canonical append-only trace. Markdown and
the manifest are derived views. The trace schema is strict and records the
exact public request, effective configuration, complete harness and its
fingerprint, exact rendered-prompt digests, ABI identity, raw model requests
and responses, typed call roles, causal IDs, observations, recovery decisions,
usage completeness, limits, timings, and a final response or typed failure.
Unknown trace values are rejected rather than serialized with `repr`.
Post-hoc consumers should call `load_trace(path, expected_run_id=...)` rather
than parsing individual lines. Corruption, truncation, run-ID disagreement,
causal errors, orphaned model calls, and manifest/final-event disagreement are
fatal `TraceError` failures; the reader never repairs or skips an event.

Task conditioning, datasets, verifiers, rewards, and benchmark recipes belong
outside `src/rlm/`. The optional `benchmarks/oolong.py` runner performs paired
RLM/direct evaluation: arm-specific prompt profiles are derived from one
immutable comparison, with identical task, context, model, seed, sampling, and
budget. It keeps transport/decode/provenance failures separate from verifier
scores, groups split data by context digest, records repetitions and complete
provenance, and audits RLM aggregate usage rather than a leaf response.

The intended sequence is fixed-harness evaluation, successful-trajectory
self-SFT, later fixed-harness RL with verifiable rewards, and only then
symbolic harness search under held-out, paired, fixed-budget evaluation. A
harness optimizer never receives verifier answers or authority to alter held-out
splits, global budgets, trace semantics, or fatal-error handling.

## Security

The IPython executor is process-isolated but **not** a security sandbox.
Generated code has the file, network, and process permissions of the current
OS user. Run only trusted generated code and treat traces as sensitive.
