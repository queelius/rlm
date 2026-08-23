# RLM Kernel and Experiment Contract Design

**Status:** Proposed for written review

**Date:** 2026-08-23

## Goal

Build a small, strict Recursive Language Model kernel that is easy to read,
test, reproduce, and use as an experimental environment. The public interface
remains one non-streaming OpenAI Responses request in and one completed
Responses object out. Internally, a controller produces one Python cell per
turn, observes its execution, and either repairs its work or submits a final
answer.

The same kernel must support fixed-harness evaluation, successful-trajectory
self-SFT, later RL with verifiable rewards, and eventually controlled symbolic
harness search. Experimental support must not turn the kernel into a general
agent framework.

## Non-goals

- Chat Completions, streaming, background responses, or automatic API-family
  conversion.
- Automatic retries, alternate-model substitution, or direct-model fallback.
- A generic plugin, scheduler, verifier, training, or distributed-compute
  framework.
- Task-family recipes in the universal controller prompt.
- Allowing an optimizer to change verifiers, held-out data, global budgets,
  trace semantics, or fatal-error handling.

## Architectural rule

Use immutable data for decisions that experiments may vary, protocols only at
real effect boundaries, and concrete kernel invariants for safety and API
correctness. Every model-visible or experimentally meaningful choice must be
serializable, versioned, fingerprinted, and recorded. There are no behavioral
defaults that exist only in process state.

## Configuration model

The runtime configuration is conceptually divided as follows:

```text
RLMConfig
├── harness: HarnessSpec
├── limits: RunLimits
├── controller: ControllerConfig
├── execution: ExecutionConfig
└── tracing: TraceConfig
```

`HarnessSpec` contains model-facing behavior:

```python
@dataclass(frozen=True, slots=True)
class HarnessSpec:
    schema_version: str
    prompt: PromptSpec
    abi_version: str
    recovery: RecoverySpec
    observation: ObservationSpec
    context: ContextSpec
```

It has a canonical JSON representation and SHA-256 fingerprint. Initially
there is one implementation of each policy. The types establish stable seams;
they are not justification for multiple flags or speculative variants.

`RunLimits` is separate from `HarnessSpec`. Turns, model calls, subcalls,
parallelism, recursion depth, wall time, execution time, and output bounds are
externally enforced experimental controls. A harness optimizer must not gain
reward by increasing them.

`ObservationSpec` may choose representation and truncation behavior only
within the hard output caps imposed by `RunLimits`; it cannot raise those caps.

Controller model selection and sampling options belong to `ControllerConfig`,
not the harness prompt. Trace storage and rendering belong to `TraceConfig`.

## Small component boundaries

The kernel has five narrow boundaries:

```python
class ModelBackend(Protocol):
    def complete(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float,
        context: ModelCallContext,
    ) -> dict[str, Any]: ...

class ExecutionEnvironment(Protocol):
    def execute(...) -> ExecutionResult: ...

class TraceSink(Protocol):
    def event(...) -> EventRef: ...

class ControllerProtocol(Protocol):
    def parse(...) -> ControllerAction: ...

class RecoveryPolicy(Protocol):
    def decide(...) -> RecoveryDecision: ...
```

`ModelCallContext` supplies a typed role, run/branch/call identifiers, and
depth. It lets a local training backend associate token-level rollout data
with the canonical trace without changing the Responses wire object. The HTTP
backend ignores training metadata but must honor the supplied timeout.

Effects are replaceable for testing. Protocol and recovery implementations
are pure, described by `HarnessSpec`, and have no hidden mutable state.

## Versioned environment ABI

The Python namespace is described by one immutable `EnvironmentABI`. It is a
typed registry, not a dynamic plugin manager. The executor namespace, host IPC
dispatcher, prompt documentation, and trace manifest derive from the same
function definitions so they cannot drift independently.

The initial namespace retains:

- the exact public `request`;
- `model_complete` and `model_complete_batch`;
- strict text conveniences `ask` and `ask_batch`;
- optional `rlm_complete` and `rlm_complete_batch` when depth permits;
- `FINAL_TEXT`, `FINAL_RESPONSE`, and `SHOW_VARS`.

Host operations use typed discriminants at the Python/IPC boundary and strict
JSON payloads. Unknown operations and invalid payloads fail explicitly.

`ask` returns non-empty text or raises a recoverable `ModelOutputFault` inside
the controller's cell. `ask_batch` returns a structured batch object with
successes and failures aligned by input index. It never uses a singleton
"pending error" flag, never clears one failure because another call succeeds,
and never automatically recomputes a leaf. The controller explicitly decides
which failed indexes to retry.

## Controller protocol and context

The universal prompt describes only the stable RLM protocol and concise
general policy. Determinant algorithms, Oolong instructions, `DATA=` parsing,
and other benchmark recipes move to task conditioning outside the kernel.

The controller returns exactly one complete `python` fenced cell and no prose.
Parsing is an exact line-oriented protocol rather than a permissive regular
expression. Empty cells, extra prose, extra fences, or unsupported controller
output items become recoverable protocol faults.

The first turn is not host-mandated inspection. The prompt may recommend
inspection when it is useful, but the model may immediately perform a valid
action or final submission.

The rolling controller context preserves every output item from the latest
controller response, including reasoning items, followed by the structured
execution observation. IPython remains durable working memory. Large request
contents are not copied into the private conversation.

## Typed fault boundary

Controller-produced defects are repairable because the controller can change
its next action:

- malformed controller output;
- Python syntax or runtime exceptions;
- empty text from a strict text helper;
- invalid final submission.

They are represented as typed faults and converted by `RecoveryPolicy` into a
structured, bounded observation. Each repair consumes a normal turn. The
default policy repairs all such faults until the global turn limit is reached.
There is no automatic retry.

Infrastructure and invariant failures are fatal and bypass `RecoveryPolicy`:

- invalid public requests;
- upstream HTTP, timeout, or invalid-JSON failures;
- malformed backend Responses objects;
- host bridge, IPC, executor startup, or worker-process failures;
- deadline, model-call, subcall, recursion, or other budget exhaustion;
- strict trace-write failures when tracing is enabled.

`stderr` is ordinary captured output, not an execution-status bit.
`ExecutionResult` carries a separate structured exception field. A warning on
stderr cannot discard a valid final. A final submitted before a later Python
exception or host fault is discarded.

The host does not reject repeated source text heuristically. Persistent state
can make an identical cell legitimate; maximum turns provide the loop bound.

## Structured observations

An execution observation has one stable JSON-compatible shape containing
stdout, stderr, display output, exception metadata, namespace summaries,
duration, and truncation metadata. Repair guidance is derived from the typed
fault, not inferred from the presence of text on stderr. The model receives a
compact observation; the trace retains the complete bounded execution record.

## Responses contract

Backend responses and `FINAL_RESPONSE` values are validated at their boundary.
A completed response must be strict JSON with a valid Responses envelope,
including `object == "response"`, a completed status, identifying fields, and
a non-empty structurally valid output list. Message output and its content
parts receive discriminant-specific structural validation. Unknown provider
fields and well-formed future output-item types are preserved rather than
normalized away.

`FINAL_TEXT` constructs one minimal completed text response. It is rejected
when the request requires tool, structured-output, continuation, or other
semantics that a synthetic text response cannot faithfully provide.

A successful non-JSON HTTP body is always an upstream protocol error. Error
status bodies may be preserved as explicitly labeled raw text. Invalid or
malformed usage fields never silently become zero. Missing usage is recorded
as unreported so aggregate telemetry cannot imply false completeness.

## Deadlines and concurrency

The run ledger owns one monotonic deadline shared by every branch. Remaining
time is passed through the executor callback boundary into every backend call.
The built-in HTTP backend uses the smaller of its configured transport timeout,
the cell's remaining time, and the run's remaining time. The ledger checks the
deadline again after the call, preventing a nonconforming custom backend from
returning a successful result late.

The backend protocol requires timeout compliance. Python cannot safely cancel
an arbitrary blocking in-process implementation, so custom backends that
ignore the contract are defective rather than silently accommodated by leaked
daemon threads or unbounded fallback behavior.

## Canonical trace and experiment boundary

When disabled, the null trace sink performs no copying, serialization, event
retention, or filesystem work. When enabled, JSONL is the canonical append-only
record. Markdown and manifests are derived views. Serialization is strict;
unknown objects are not replaced with `repr`.

The trace schema is versioned and records:

- exact public request and effective configuration;
- complete harness specification and fingerprint;
- exact model-visible request and raw model response for every call;
- typed model-call role and causal identifiers;
- parsed action, execution result, observation, and recovery decision;
- recursive branch relationships;
- final response or explicit run failure;
- usage completeness, limits, and timings.

An experiment runner owns model/checkpoint digest, source revision, dependency
lock hash, dataset revision, example ID, split, condition, repetition, seed,
verifier result, and reward. It stores them in a sidecar keyed by `run_id`.
Verifier outputs and rewards never enter controller context.

Training exporters derive examples from the canonical events:

```text
exact model-visible request        -> causal-LM state
model-authored response content    -> prediction target
environment output                 -> later context, never a target
transport IDs, status, and usage   -> metadata, never a target
```

The raw Responses object remains in the trace for reproducibility. Exporters
select model-authored reasoning and assistant output items structurally; they
do not train on response IDs, timestamps, status fields, usage, or the parsed
Python cell reconstructed by the host.

Every model call is assigned a typed role. The first self-SFT condition uses
controller calls at every recursive depth as targets and treats public/leaf
subcall responses as context-only. The stored roles permit later ablations or
full-policy RL without changing the kernel trace.

## Training and co-adaptation sequence

The intended research progression is:

1. Evaluate plain and fixed-harness baseline models.
2. Generate verified fixed-harness trajectories and run success-filtered
   self-SFT, including the matched plain-model control.
3. Run fixed-harness RL with verifiable rewards from both the base and self-SFT
   checkpoints.
4. Freeze a model and optimize symbolic `HarnessSpec` candidates on development
   tasks using paired, fixed-budget evaluation.
5. Alternate model-weight updates and harness search as coordinate ascent.

Harness search initially changes only serialized prompt, context, observation,
and safe ABI-selection data. Later source-code candidates must carry a source
hash and pass the complete contract and failure suite before evaluation. The
optimizer cannot mutate the verifier, held-out split, budgets, trace, public
contract, executor safety boundary, or fatal-fault taxonomy.

Different model families begin with the same canonical harness. Per-model
optimized harnesses are cross-evaluated to distinguish universal improvements
from genuine model/harness co-adaptation.

## Benchmark durability

Benchmark runners record a structured result for every example, including
transport and decoding failures. They atomically checkpoint after each
example, reject empty selections before computing aggregates, and never turn a
failed request into an empty answer with an ordinary score. Benchmark and task
conditioning remain outside `src/rlm/`.

## Test strategy

- Pure unit tests for configuration serialization, fingerprints, parsing,
  response validation, recovery decisions, observation encoding, and usage.
- Contract tests for backend timeout behavior, executor results, strict IPC,
  and trace sinks.
- A fault matrix proving each failure is either explicitly repairable or
  explicitly fatal.
- Deterministic state-machine tests using scripted backends and executors.
- Regression tests for partial batch failures, harmless stderr, late model
  calls, malformed success bodies, invalid finals, exact code framing,
  reasoning-item replay, and strict CLI option parsing.
- End-to-end HTTP and real-IPython tests.
- Golden tests for canonical harness fingerprints and trace schema.
- Trajectory-export tests in `rlm-bootstrap` proving observations are masked
  from targets and model-call roles remain intact.
- Full pytest, Ruff, formatting, build, and coverage reports before completion.

## Implementation scope

The implementation will proceed in small TDD changes while preserving the
current Responses-only simplification and removed Codex-specific backend. The
core loop will remain directly readable; abstractions that do not remove a
current ambiguity, enable deterministic testing, or support a named experiment
are out of scope.
