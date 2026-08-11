# rlm

`rlm` is a small Recursive Language Model runtime. It decorates an
OpenAI-compatible model without flattening the caller's request: a complete
Chat Completions or Responses body is placed in a persistent IPython child,
and a private controller can inspect it, run Python, make ordinary model calls,
batch work, and return a complete response.

The project is inference machinery, not an evaluation or RL framework. See
[DESIGN.md](DESIGN.md) for the contract and [RESEARCH.md](RESEARCH.md) for the
papers and implementations that shaped it.

## Supported surface

- `POST /v1/chat/completions`
- `POST /v1/responses`
- Full ordered message/input objects and unknown provider fields
- Native `ipython` function-tool actions or fenced-Python actions
- Optional same-engine recursive children
- Canonical JSONL debug traces and a readable Markdown rendering

Streaming and asynchronous Responses background mode are rejected explicitly.

## Install

```bash
uv sync --extra dev
uv run pytest
```

## Library usage

```python
from rlm import DebugConfig, OpenAIEndpoint, RLM, RLMConfig

upstream = OpenAIEndpoint(
    base_url="http://192.168.0.204:11434/v1",
    api_key="ollama",
)
model = RLM(
    upstream,
    config=RLMConfig(
        controller_model="qwen3.5:latest",
        controller_options={"temperature": 0, "think": False},
        # Ollama 0.30.7 + qwen3.5 currently behaves better with fenced code.
        action_mode="code",
        debug=DebugConfig(enabled=True, directory="runs"),
    ),
)

result = model.run(
    "chat.completions",
    {
        "model": "qwen3.5:latest",
        "messages": [
            {"role": "system", "content": "Be exact."},
            {"role": "user", "content": "Hi, how are you?"},
        ],
    },
)
print(result.response["choices"][0]["message"]["content"])
print(result.trace_directory)
```

`RLM.chat_completions(request)` and `RLM.responses(request)` return only the
wire response. `RLM.run(api, request)` additionally returns aggregate hidden
usage, turns, duration, stop reason, run ID, and the trace location.

For a matched baseline, `RLM.direct(api, request)` sends the request to the
wrapped public model unchanged.

## Optional Codex agent controller

`CodexAgentBackend` can evaluate an existing, locally authenticated Codex CLI
as the **private controller only**. It is intentionally not a REST wrapper or
a wire-equivalent GPT backend: Codex has its own system behavior, agent loop,
and observable event trajectory. Keep an ordinary OpenAI-compatible endpoint
as the public and worker backend, and use fenced-code actions:

```python
from rlm import CodexAgentBackend, OpenAIEndpoint, RLM, RLMConfig

upstream = OpenAIEndpoint(base_url="http://192.168.0.204:11434/v1", api_key="ollama")
controller = CodexAgentBackend(model="gpt-5.6-sol", reasoning_effort="max")
model = RLM(
    upstream,
    controller_backend=controller,
    config=RLMConfig(
        controller_model="gpt-5.6-sol",
        action_mode="code",
    ),
)
```

Each controller call runs `codex exec` with JSONL output, an ephemeral session,
ignored user configuration and rules, a read-only sandbox, and a fresh empty
temporary Git workspace. The adapter sends the request over stdin using argv
execution without a shell. The subprocess inherits the ordinary environment so
the installed CLI can use its supported login state; the adapter never
inspects, extracts, copies, or reconstructs authentication token values.
The temporary workspace isolates working context; it is not a claim that the
subprocess cannot read outside that directory. The installed Codex sandbox's
read policy remains the security boundary.

The raw observable Codex JSONL events (including any provider-exposed reasoning
summaries and tool events) are retained under the `x_rlm_codex_agent` response
extension and therefore appear in RLM debug traces. Hidden reasoning is not
available. Native OpenAI tool calls cannot be reconstructed from an agent
message, so this backend rejects them and requires `action_mode="code"`.

To serve the normal RLM REST proxy with Ollama as the public and worker model
and Codex as only the private controller:

```bash
uv run rlm serve \
  --upstream-base-url http://192.168.0.204:11434/v1 \
  --upstream-api-key ollama \
  --controller-backend codex \
  --controller-model gpt-5.6-sol \
  --codex-reasoning-effort max \
  --action-mode code \
  --worker-model qwen3.5:latest \
  --debug-dir runs/codex-controller
```

Requests sent to this proxy still name the public Ollama model, for example
`qwen3.5:latest`. `model_complete()` identity calls and explicit worker calls
remain on Ollama; only private controller turns invoke the authenticated local
Codex CLI.

## Proxy usage

```bash
uv run rlm serve \
  --upstream-base-url http://192.168.0.204:11434/v1 \
  --upstream-api-key ollama \
  --controller-model qwen3.5:latest \
  --controller-option think=false \
  --action-mode code \
  --debug-dir runs
```

Then use an ordinary OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen3.5:latest",
    messages=[{"role": "user", "content": "Hi, how are you?"}],
)
```

RLM metadata is returned in `X-RLM-*` headers. The JSON body remains a normal
response for the selected OpenAI API family.

## The IPython contract

Every branch receives an exact private `request` object and these helpers:

```python
model_complete(request=None, api=None)  # no args: exact identity call
model_complete_batch(requests, api=None)
ask(prompt, system=None, model=None, api=None)
ask_batch(prompts, system=None, model=None, api=None)
rlm_complete(request=None, api=None)  # when max_depth permits it
rlm_complete_batch(requests, api=None)
FINAL_RESPONSE(response)
FINAL_TEXT(text)
SHOW_VARS()
```

Termination is explicit: a controller must execute `FINAL_TEXT(text)` or
`FINAL_RESPONSE(response)` inside IPython. Ordinary controller prose, printed
output, and a cell's last expression do not become the public answer. Prose
without an executable action receives private repair feedback; repeated
protocol errors eventually use the exact-request direct fallback.

Controller, public, and worker backends may be different objects. By default
they are the same backend. Recursive children use the same engine and protocol
with a fresh kernel and a shared run-wide budget.

## Debugging

With `DebugConfig(enabled=True)`, each run directory contains:

- `trace.jsonl`: canonical append-only events with causal IDs
- `trace.md`: exact readable rendering of those events
- `manifest.json`: termination, timing, usage, and limits

The trace includes the public request, controller protocol and prompts,
provider-exposed reasoning fields, every raw model request/response, Python,
captured execution output, bounded controller observations, subcalls, and final
response. Captured stdout, stderr, and display text are each limited by
`max_execution_output_chars` (one million by default) before crossing IPC. A
provider's hidden chain-of-thought cannot be recovered.

## Security and training boundaries

The local executor is process-isolated but not sandboxed. Only run trusted
workloads until a sandbox executor is added. Host environment variables are
removed from the child, model credentials are never injected, IPC uses strict
JSON rather than pickle, output is bounded, and POSIX descendant processes are
cleaned up. Generated code can still access files, networks, and processes
allowed to the current OS user. A non-cooperative custom backend may continue
its own in-flight callback thread after the cell deadline, so backends must also
enforce finite I/O timeouts.

OpenAI-compatible JSON is enough for inference and evaluation; it is not
token-exact RL telemetry. Policy-gradient training additionally needs sampled
token IDs, log probabilities, renderer parity, and generated-token masks. Those
adapters, along with benchmarks, rewards, prompt evolution, and model updates,
belong in downstream research projects.

Codex subscriptions are also not OpenAI API entitlements. The optional local
adapter above is a Codex agent trajectory rather than a raw GPT endpoint or
hidden reasoning trace. Review the applicable OpenAI terms before collecting
outputs for training a different model.
