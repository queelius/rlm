# RLM design

## Public contract

RLM accepts one non-streaming OpenAI Responses request and returns one completed
Responses object. Unknown, strict-JSON provider fields are preserved. Its only
HTTP operations are `GET /v1/models` pass-through and `POST /v1/responses`.
Chat Completions, streaming, background Responses, and implicit API conversion
are unsupported.

`RLM.direct(request)` is a deliberate direct-model control for evaluation. It
is never used to recover an RLM failure. The runtime has no retry policy,
alternate-model path, answer fallback, or error-to-empty conversion.

## Structural configuration and harness identity

`RLMConfig` has five nested sections:

```text
RLMConfig(harness: HarnessSpec,
          limits: RunLimits,
          controller: ControllerConfig,
          execution: ExecutionConfig,
          tracing: TraceConfig)
```

`HarnessSpec` is canonical JSON with a SHA-256 fingerprint. It contains the
prompt fragments, digests of both exact rendered controller-prompt variants,
the typed bootstrap specification, environment ABI version and digest,
recovery specification, observation specification, and context specification.
Initialization verifies the ABI and both rendered prompt digests before model
work; the selected variant is checked again when rendered. Controller
model/options are separate from the harness, and limits are externally enforced
experiment controls.

The structural specification dataclasses are frozen. `ControllerConfig.options`
is deliberately an owned mutable mapping between runs so an experiment can
change sampling without rebuilding a configuration graph. `RLM.run()` and
`RLM.run_direct()` take one deep strict-JSON snapshot before work starts; that
snapshot, not a claim that the whole graph is immutable, provides per-run
isolation and trace reproducibility.

## Exact controller/executor loop

Each branch has a private controller conversation and persistent IPython child:

```python
for turn in range(1, limits.max_turns + 1):
    response = controller.complete(private_request, timeout=remaining, context=context)
    action = parse_exactly_one_python_cell(response)
    result = executor.execute(action.code, timeout=remaining)
    if result.submission:
        return validate_final(result.submission)
    append_latest_controller_items_and_observation()
raise LimitExceededError
```

The private context starts with a typed `rlm.controller_bootstrap` schema-v2
envelope. Its `content_in_message: false` invariant is part of `HarnessSpec`;
its request binding comes from the environment ABI. Its required `submission`
contract derives allowed final kinds from the exact public request: ordinary
requests allow `text` and `response`, while requests that cannot be represented
by a minimal text response allow only `response`. The envelope is runtime
metadata, not caller content. A controller determines the task by inspecting
needed fields of the bound request or deliberately delegating with
`model_complete(request)`. This does not impose a separate inspection turn.

Controller output is strict: exactly one assistant message, exactly one
`output_text` item, and exactly one non-empty fenced `python` cell with no extra
prose or fences. Every nonterminal result becomes one strict-JSON
`rlm.controller_observation` schema-v2 user item. It binds the observation to
`request`, marks the required submission absent, and nests the bounded execution
record; execution status `ok` means only that the cell ran, and submission status
`absent` means that no valid final was accepted. An accepted final terminates the
branch without another observation. Configured observation budgets have a
512-character floor that supports every built-in repair identity. There are no
XML wrappers or imperative prompt suffixes. The private context keeps the bootstrap,
every eligible item from the latest controller response—including reasoning
items when enabled—and that observation. IPython holds prior state.

The versioned ABI is one typed registry shared by prompt documentation,
executor globals, IPC operations, and trace manifest. It exposes
`model_complete`, `model_complete_batch`, `ask`, `ask_batch`, `FINAL_TEXT`,
`FINAL_RESPONSE`, and `SHOW_VARS`; recursive helpers are enabled only within
the depth limit. Host IPC accepts strict JSON typed payloads.

`ask_batch` returns `AskBatchResult`, retaining aligned successes and
`TextFailure` records. `failed_indexes` identifies only failed entries;
`require_texts()` raises a typed `ModelOutputFault` if any remain. There is no
singleton pending-error state, implicit retry, or recomputation of successes.

`ExecutionResult.submission` is the closed union
`TextSubmission | ResponseSubmission | None`, so mismatched FINAL kind/value
pairs cannot exist. `FINAL_TEXT` is permitted only for public requests that a
minimal synthetic text response can faithfully satisfy, and it accepts only an
already-formed string. A non-string is a recoverable `final_submission` fault,
never an implicit `str(...)` conversion. A cell that submits a final and then has
a Python exception or host fault has no final submission.
Captured stderr is data, not an error status. Printed, displayed, or returned
intermediate values never become a public answer; every successful branch ends
only through an explicit valid `FINAL_TEXT` or `FINAL_RESPONSE` call.

## Fault and deadline boundaries

Recoverable controller faults are malformed controller output, execution
exceptions, strict-text helper failures, and invalid final submissions. The
sealed `RecoveryPolicy` maps them to typed `Repair` or `Abort` decisions.
Repairs are observations and consume ordinary turns; they do not retry an
operation.

Invalid public requests, upstream HTTP/transport/timeout/non-JSON failures,
malformed completed Responses, host bridge and executor failures, deadline or
other limit exhaustion, and strict trace failures are fatal. They bypass the
recovery policy and preserve their typed error record.

The shared ledger owns a monotonic run deadline, model/subcall/parallel/depth
limits, and aggregate reported usage. Every backend call receives a remaining
timeout and `ModelCallContext` with role, run/branch/call IDs, and depth. The
built-in endpoint uses the minimum transport and supplied timeout. The ledger
checks again after every call; custom backends must comply rather than relying
on leaked helper threads or a fallback.

## Trace and experiment boundary

Disabled tracing uses a null sink with no payload conversion, event retention,
manifest construction, serialization, or filesystem work. Enabled tracing is
strict JSONL with frozen event payloads and causal-parent validation. The
manifest is a validated final schema object; trace write/finalization failures
are fatal. Trace serialization rejects unknown values rather than using
`repr`.

Canonical events retain exact public and model-visible requests, raw Responses
objects, reasoning output, roles, IDs, actions, execution/observation/recovery
records, branch relationships, usage completeness, limits, timings, complete
harness identity, and final response or run failure.

The runtime owns this execution contract. Benchmarks own task conditioning,
data splits, verifier/reward code, and reports. The Oolong runner creates
typed RLM/direct prompt profiles from one immutable comparison, keeps paired
conditions identical in task/context/model/seed/sampling/budget, records full
provenance and repeated rollouts, audits RLM aggregate usage, and retains
failures outside verifier scoring. Context-sharing rows are split as groups.

The executor is process-isolated, not sandboxed: it must never run untrusted
code in a security-sensitive environment. On POSIX the worker must establish a
dedicated process group during bootstrap; failure is fatal because descendant
containment would otherwise be unverifiable.
