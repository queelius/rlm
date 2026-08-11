# RLM design

## Contract

An RLM is a model decorator:

```text
RLM : OpenAI-compatible model -> OpenAI-compatible model
```

It accepts one complete, non-streaming request for either
`POST /v1/chat/completions` or `POST /v1/responses` and returns a response for
the same API family. The exact request body is retained as an opaque object.
Unknown provider fields are not normalized away.

The public request and the controller conversation are different objects. The
RLM protocol prompt is never inserted into the caller's message history.

Streaming is intentionally outside the contract. `stream: true` is rejected
with an OpenAI-shaped invalid-request error.

## Runtime

Each branch has three parts:

1. A host-side endpoint transport owns credentials and all model calls.
2. A controller model receives a small, private protocol conversation.
3. A fresh persistent IPython subprocess holds the exact public request and
   intermediate state.

The kernel exposes typed host bridges:

```python
model_complete(request=None, api=None) -> dict
model_complete_batch(requests, api=None) -> list[dict]
ask(prompt, system=None, model=None, api=None) -> str
ask_batch(prompts, system=None, model=None, api=None) -> list[str]
rlm_complete(request=None, api=None) -> dict       # when recursion is enabled
rlm_complete_batch(requests, api=None) -> list[dict]
FINAL_RESPONSE(response) -> None
FINAL_TEXT(text) -> None
SHOW_VARS() -> dict[str, str]
```

Calling `model_complete()` without arguments sends the exact original request
to the upstream model. This is the identity/pass-through path and makes the
base model a contained special case of the RLM.

`FINAL_RESPONSE` is the lossless path: it submits a complete response object.
`FINAL_TEXT` wraps caller-facing text for an ordinary text request. A branch
terminates only when one of these functions is invoked by executable IPython;
controller prose, printed output, and a cell's last expression are never
implicit finals. Requests involving caller tools or structured output must
finish with `FINAL_RESPONSE`.

Host bridges never place API keys in the kernel.
All child/host pipe frames use strict UTF-8 JSON rather than pickle and have a
64 MiB frame ceiling. Public library requests and submitted response objects
must therefore use the ordinary JSON data model: null, booleans, finite
numbers, Unicode strings, lists, and string-keyed objects.

## Controller action protocols

The engine has one semantic action—execute an IPython cell—but two encodings:

- `tool`: advertise one OpenAI function tool named `ipython`; this is the
  default for tool-capable models.
- `code`: parse one fenced `python`, `py`, or `repl` cell from controller text;
  this is the portability fallback for endpoints with weak tool support.

In either mode, a controller turn with no executable action is a recoverable
protocol error. The raw controller response is retained in its private history,
followed by repair feedback directing it to execute `FINAL_TEXT` or
`FINAL_RESPONSE`. Execution feedback is bounded before it re-enters the
controller context; a larger, independently bounded capture remains in the
debug trace.

## Recursion and limits

An ordinary subcall is a model call, not another agent. `rlm_complete` invokes
the same engine with a new controller conversation and kernel, an incremented
depth, and the same shared ledger and trace tree.

Recursive children are disabled by default. The host, not the model, enforces:

- root turns per branch;
- total model and submodel calls;
- concurrent fan-out;
- recursive depth;
- wall-clock deadline;
- cell timeout, captured output size, and controller-visible observation size;
- consecutive protocol or execution errors.

No automatic compaction is performed in version 0.1. Compaction changes the
sampled transcript and will be introduced only behind an explicit policy.

## Debug trace

Debug mode writes an append-only `trace.jsonl`. Every event has stable causal
coordinates:

```text
run_id, event_id, parent_event_id, branch_id, depth, timestamp, type, payload
```

The trace records exact public traffic, every private controller request and
raw response, provider-exposed reasoning fields, code, bounded captured and
visible execution output, subcalls, timings, usage, prompt hashes, termination,
and the final response. A Markdown rendering is derived from JSONL for reading.

Hidden chain-of-thought that an endpoint does not return cannot be recorded.
Debug mode is data-sensitive and intended for trusted local research.

## Compatibility scope

Version 0.1 targets the create operations of Chat Completions and Responses.
It does not implement streaming, WebSockets, background mode, or OpenAI's
retrieve/delete/list resource endpoints. Stateful identifiers such as
`previous_response_id` are preserved and forwarded to the upstream endpoint;
the RLM does not invent a second provider-side conversation store.

The transport preserves arbitrary JSON. Provider capability failures are
surfaced explicitly rather than silently dropping request fields.

### Distinct Codex controller semantics

`CodexAgentBackend` is a controller-evaluation adapter, not another public
OpenAI-compatible endpoint. It serializes a private controller request into a
single `codex exec` instruction and synthesizes the selected response family
from the last observable `agent_message` event. The Codex agent's own system
behavior and internal tool loop mean this transformation cannot provide raw
model-completion equivalence.

Accordingly, it is supported only as a private controller in `code` action
mode. Native tool-call requests are rejected. Every call uses an ephemeral
read-only CLI session in a new empty temporary Git workspace; the CLI remains
solely responsible for its existing authentication. The subprocess inherits
the normal environment, but the adapter does not inspect, extract, copy, or
reconstruct authentication values. Observable JSONL events are retained as a
provider extension for the normal debug trace, while hidden reasoning remains
unavailable. The temporary directory isolates working context, not all
filesystem reads; the installed Codex sandbox remains the security boundary.

## Research boundary

This repository contains inference machinery, conformance tests, debug events,
and tiny examples. Benchmarks, graders, reward functions, prompt evolution,
Continual Harness refinement, RL algorithms, token-exact renderer integration,
and layer-selection experiments belong in downstream research projects.

For future policy-gradient training, OpenAI wire compatibility is not enough:
the integration must additionally preserve exact sampled token IDs,
log-probabilities, masks, and renderer behavior.
