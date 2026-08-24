# RLM Kernel Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous recovery, execution, transport, response, trace, and benchmark behavior with one strict, typed, experiment-ready RLM state machine that never hides a defect or substitutes an answer.

**Architecture:** Keep the controller → Python → observation loop directly readable. Put model-facing experimental choices in a canonical `HarnessSpec`; use typed boundaries for model calls, recovery, host operations, execution results, and tracing; keep public API validation, deadlines, budgets, and fatal faults as non-configurable kernel invariants.

**Tech Stack:** Python 3.10, dataclasses and enums, strict JSON, IPython subprocess execution, httpx, FastAPI, pytest, pytest-cov, Ruff, Hatchling/uv.

**Spec:** `docs/superpowers/specs/2026-08-23-rlm-kernel-and-experiment-contract-design.md`

## Global Constraints

- Preserve only the non-streaming OpenAI Responses API; do not add Chat Completions compatibility.
- Preserve unknown request and provider fields after strict JSON validation.
- Never add automatic retries, direct-answer fallback, alternate-model fallback, or error-to-empty-value conversion.
- Controller-authored defects may consume another turn; transport, backend schema, IPC, executor, trace, deadline, and budget failures are fatal.
- Keep task conditioning, datasets, verifiers, rewards, and training outside `src/rlm/`.
- Use strict structured data at boundaries; do not parse structured data with regular expressions or maintain unvalidated parallel lookup tables.
- Keep Python 3.10 compatibility and the Ruff 100-character line limit.
- Add a regression test before each behavior change and observe the expected failure before implementation.
- A collection/import failure proves only that a declaration is missing. Add the minimum declaration,
  then rerun each targeted test until its behavioral assertion fails before implementing that behavior;
  never treat one masked collection error as evidence for the rest of a task's red tests.
- The worktree already contains overlapping, reviewed changes. Do not stage or commit during these
  tasks: path-level `git add` would silently include pre-existing hunks. At each checkpoint inspect
  the named path diff and run `git diff --check`; after the complete implementation is reviewed,
  ask the user whether to create one commit or a deliberately partitioned series.
- Preserve the existing Codex-backend deletion and do not reintroduce Codex-specific behavior.

## File Responsibility Map

- `src/rlm/specs.py`: immutable, canonical model-facing harness specification.
- `src/rlm/config.py`: operational controller, limit, execution, and trace configuration.
- `src/rlm/recovery.py`: typed recoverable controller faults and recovery decisions.
- `src/rlm/abi.py`: versioned environment function definitions and typed host operations.
- `src/rlm/types.py`: runtime value types such as model-call context and execution results.
- `src/rlm/json.py`: shared strict JSON decoding, validation, and canonical encoding.
- `src/rlm/response.py`: Responses-envelope, terminal-response, and text construction contract.
- `src/rlm/backend.py`: deadline-aware provider-neutral Responses transport.
- `src/rlm/executor.py`: IPython worker, ABI binding, strict host IPC, and structured execution outcomes.
- `src/rlm/protocol.py`: exact controller action parsing and rolling Responses context.
- `src/rlm/prompts.py`: lean universal protocol rendering and request metadata only.
- `src/rlm/trace.py`: zero-cost disabled sink and strict canonical JSONL sink.
- `src/rlm/engine.py`: the single controller/action/observation state machine.
- `benchmarks/oolong.py`: durable external benchmark runner and checkpoints.

---

### Task 1: Canonical Harness and Runtime Configuration

**Files:**
- Create: `src/rlm/specs.py`
- Create: `src/rlm/recovery.py`
- Create: `tests/test_specs.py`
- Create: `tests/test_recovery.py`
- Modify: `src/rlm/config.py`
- Modify: `src/rlm/prompts.py`
- Modify: `src/rlm/types.py`
- Modify: `src/rlm/__init__.py`
- Modify: `src/rlm/engine.py`
- Modify: `src/rlm/ledger.py`
- Modify: `src/rlm/cli.py`
- Modify: `tests/fakes.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_engine.py`
- Delete: `src/rlm/codex_backend.py`
- Delete: `tests/test_codex_backend.py`

**Interfaces:**
- Produces: `HarnessSpec.to_dict() -> dict[str, Any]` and `HarnessSpec.fingerprint() -> str`.
- Produces: `RLMConfig(harness, limits, controller, execution, tracing)` with nested typed
  configuration and strict `to_dict()` serialization.
- Produces: `ControllerFault`, `Repair`, `Abort`, and sealed `RecoveryPolicy.decide(...)`.
- Produces: `ModelRole` and `ModelCallContext` for later backend and trace work.

- [ ] **Step 1: Add failing canonical-spec and recovery-policy tests**

Keep one shared request factory in `tests/fakes.py`; tests call the factory so no mutable request
object leaks state between cases:

```python
def request() -> dict[str, Any]:
    return {
        "model": "public-model",
        "input": [{"role": "user", "content": "Preserve this request."}],
        "temperature": 0,
        "provider_extension": {"keep": [1, {"nested": True}]},
    }
```

```python
# tests/test_specs.py
from dataclasses import replace
from pathlib import Path

from rlm import ExecutionConfig, RLMConfig
from rlm.prompts import default_harness_spec
from rlm.response import json_compatibility_error


def test_harness_fingerprint_is_stable_and_content_addressed() -> None:
    original = default_harness_spec()
    same = replace(original)
    changed = replace(
        original,
        prompt=replace(original.prompt, policy=original.prompt.policy + "\nVerify once."),
    )

    assert same.fingerprint() == original.fingerprint()
    assert changed.fingerprint() != original.fingerprint()
    assert len(original.fingerprint()) == 64


def test_harness_spec_is_strict_json() -> None:
    spec = default_harness_spec()
    assert spec.to_dict()["schema_version"] == "1"
    assert spec.to_dict()["context"]["preserve_reasoning"] is True


def test_runtime_config_serializes_without_credentials_or_python_objects() -> None:
    config = RLMConfig(execution=ExecutionConfig(working_directory=Path("work")))
    value = config.to_dict()

    assert value["execution"]["working_directory"] == "work"
    assert json_compatibility_error(value) is None
```

```python
# tests/test_recovery.py
from rlm.recovery import ControllerFault, FaultKind, RecoveryPolicy, Repair
from rlm.specs import RecoverySpec


def test_default_policy_repairs_controller_fault() -> None:
    fault = ControllerFault(FaultKind.EXECUTION, "NameError: missing")
    decision = RecoveryPolicy(RecoverySpec()).decide(fault, turn=2)

    assert isinstance(decision, Repair)
    assert decision.fault is fault
```

```python
# tests/test_engine.py
from collections.abc import Mapping
from dataclasses import replace


def test_explicit_falsey_controller_backend_is_not_replaced() -> None:
    class FalseyBackend(ScriptedBackend):
        def __bool__(self) -> bool:
            return False

    public = ScriptedBackend([])
    controller = FalseyBackend([controller_code('FINAL_TEXT("done")')])
    result = RLM(public, controller_backend=controller).run(request())

    assert extract_text(result.response) == "done"
    assert len(controller.calls) == 1
    assert public.calls == []


def test_run_configuration_snapshot_detaches_mutable_options() -> None:
    options = {"temperature": 0, "nested": {"seed": 7}}
    config = RLMConfig(controller=ControllerConfig(options=options))
    snapshot = config.snapshot()

    options["temperature"] = 1
    config.controller.options["nested"]["seed"] = 9

    assert snapshot.controller.options == {"temperature": 0, "nested": {"seed": 7}}


def test_run_uses_one_snapshot_when_caller_options_mutate_between_turns() -> None:
    options = {"temperature": 0}

    class MutatingBackend(ScriptedBackend):
        def complete(self, model_request: Mapping[str, Any]) -> dict[str, Any]:
            response = super().complete(model_request)
            options["temperature"] = 1
            return response

    controller = MutatingBackend(
        [controller_code("print('continue')"), controller_code('FINAL_TEXT("done")')]
    )
    config = RLMConfig(controller=ControllerConfig(options=options))

    RLM(ScriptedBackend(), controller_backend=controller, config=config).run(request())

    assert [call.request["temperature"] for call in controller.calls] == [0, 0]


def config_with(**limit_overrides: Any) -> RLMConfig:
    return RLMConfig(limits=replace(RunLimits(), **limit_overrides))
```

- [ ] **Step 2: Run the new tests and observe import failures**

Run: `uv run pytest tests/test_specs.py tests/test_recovery.py -q`

Expected: FAIL because `rlm.specs`, `rlm.recovery`, and the exported types do not exist.

- [ ] **Step 3: Implement canonical spec and typed recovery values**

```python
# src/rlm/specs.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class HistoryMode(str, Enum):
    LATEST_TURN = "latest_turn"


@dataclass(frozen=True, slots=True)
class PromptSpec:
    name: str
    version: str
    protocol: str
    policy: str


class RecoveryMode(str, Enum):
    REPAIR = "repair"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class RecoverySpec:
    mode: RecoveryMode = RecoveryMode.REPAIR
    version: str = "1"


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    schema_version: str = "1"


@dataclass(frozen=True, slots=True)
class ContextSpec:
    history: HistoryMode = HistoryMode.LATEST_TURN
    preserve_reasoning: bool = True


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    prompt: PromptSpec
    schema_version: str = "1"
    abi_version: str = "1"
    recovery: RecoverySpec = field(default_factory=RecoverySpec)
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    context: ContextSpec = field(default_factory=ContextSpec)

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json())
        if not isinstance(value, dict):
            raise TypeError("canonical HarnessSpec must encode an object")
        return value

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
```

Every spec dataclass validates non-empty names and versions in `__post_init__`. The JSON
round-trip in `to_dict` deliberately removes dataclass and enum instances, so callers receive
only strict JSON values rather than a Python-shaped approximation of the wire representation.

```python
# src/rlm/recovery.py
from dataclasses import dataclass
from enum import Enum
from rlm.specs import RecoveryMode, RecoverySpec


class FaultKind(str, Enum):
    CONTROLLER_PROTOCOL = "controller_protocol"
    EXECUTION = "execution"
    MODEL_OUTPUT = "model_output"
    FINAL_SUBMISSION = "final_submission"


@dataclass(frozen=True, slots=True)
class ControllerFault:
    kind: FaultKind
    message: str
    details: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class Repair:
    fault: ControllerFault


@dataclass(frozen=True, slots=True)
class Abort:
    fault: ControllerFault


RecoveryDecision = Repair | Abort


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    spec: RecoverySpec

    def decide(self, fault: ControllerFault, *, turn: int) -> RecoveryDecision:
        if self.spec.mode is RecoveryMode.REPAIR:
            return Repair(fault)
        if self.spec.mode is RecoveryMode.ABORT:
            return Abort(fault)
        raise AssertionError(f"unhandled recovery mode: {self.spec.mode!r}")
```

`ControllerFault`, `Repair`, and `Abort` each expose strict `to_dict()` values for tracing.
`RecoveryPolicy` is a sealed interpreter constructed from the effective run snapshot; `RLM` does
not accept an arbitrary policy object. Adding a mode requires extending the enum, exhaustive
interpreter, spec version, golden harness, and tests together, so mutable behavior cannot hide
behind a claimed spec or unchanged fingerprint.

- [ ] **Step 4: Introduce nested operational configuration and migrate call sites mechanically**

Use these exact public types in `src/rlm/config.py`:

```python
@dataclass(frozen=True, slots=True)
class RunLimits:
    max_turns: int = 12
    max_model_calls: int = 64
    max_subcalls: int = 48
    max_parallel_model_calls: int = 8
    max_depth: int = 0
    max_total_tokens: int | None = None
    deadline_seconds: float = 600.0
    execution_timeout_seconds: float = 120.0
    max_execution_output_chars: int = 1_000_000
    max_observation_chars: int = 6_000


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    model: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    working_directory: str | Path | None = None


@dataclass(frozen=True, slots=True)
class TraceConfig:
    enabled: bool = False
    directory: str | Path = "runs"
    markdown: bool = False


@dataclass(frozen=True, slots=True)
class RLMConfig:
    harness: HarnessSpec = field(default_factory=default_harness_spec)
    limits: RunLimits = field(default_factory=RunLimits)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    tracing: TraceConfig = field(default_factory=TraceConfig)
```

Implement the nested dataclass declarations and mechanically migrate constructors/accesses first,
but leave the old backend-default expression and per-turn config reads in place. Then run:

Run: `uv run pytest tests/test_engine.py -k "falsey_controller_backend or configuration_snapshot or one_snapshot" -q`

Expected: the snapshot tests fail because `snapshot()`/run-local use is absent, and the falsey
backend test fails because the old truthiness default selects the public backend. Only after seeing
those behavioral failures, implement `snapshot()`, `is None` defaulting, and run-local config use.

Move each existing validation to the owning nested dataclass. Replace flat accesses exactly:

```text
config.max_turns                  -> config.limits.max_turns
config.max_model_calls            -> config.limits.max_model_calls
config.max_subcalls               -> config.limits.max_subcalls
config.max_parallel_subcalls      -> config.limits.max_parallel_model_calls
config.max_depth                  -> config.limits.max_depth
config.deadline_seconds           -> config.limits.deadline_seconds
config.execution_timeout          -> config.limits.execution_timeout_seconds
config.max_execution_output_chars -> config.limits.max_execution_output_chars
config.max_observation_chars      -> config.limits.max_observation_chars
config.controller_model           -> config.controller.model
config.controller_options         -> config.controller.options
config.working_directory          -> config.execution.working_directory
config.debug                      -> config.tracing
```

`max_total_tokens` is a new optional positive aggregate budget. When it is configured, every
backend response must report valid usage so the ledger can enforce it; missing usage is then a
fatal backend-accounting error rather than an unreported observation.

All integer limits reject booleans and enforce their documented positive/non-negative range;
`max_observation_chars` is at least 256 so the versioned error skeleton always fits. All
timeouts reject booleans, non-numbers, `NaN`, and infinities before checking positivity.

Update tests to construct the owning nested value, for example:

```python
config = RLMConfig(
    limits=RunLimits(max_turns=2, max_model_calls=4, execution_timeout_seconds=10),
    controller=ControllerConfig(model="controller-model", options={"temperature": 0}),
)
```

Do not add deprecated flat aliases.

Implement `RLMConfig.to_dict()` from the nested structure with explicit `Path -> str` conversion
and strict JSON validation. It never includes backend credentials or process objects. Use this one
method for `run.started`, manifests, and public debug metadata instead of separate config shaping.

Implement `RLMConfig.snapshot()` as an explicit strict-JSON round-trip through the nested
constructors. `RLM.run()` and `run_direct()` take exactly one snapshot before allocating the
ledger, and every branch, request, limit check, trace event, and manifest in that run reads only
that local snapshot. This detaches mutable option mappings from caller-owned values and prevents
an external mutation between turns from changing behavior after `run.started` has recorded the
effective configuration. Add an engine regression test whose backend mutates the caller's original
options after the first controller call and verify that the second request still uses the
snapshotted value.

Rename the CLI flags to `--max-parallel-model-calls`, `--trace-dir`, and
opt-in `--trace-markdown` in the same mechanical migration; do not retain hidden aliases for the old
names. Add `--max-total-tokens` as an optional positive integer mapped directly to the new ledger
budget.

At every optional dependency boundary, default only on `is None`. In particular, preserve an
explicit falsey `controller_backend`, configuration, or execution factory; never
replace one through `value or default`.

- [ ] **Step 5: Add model-call role and context types**

```python
# src/rlm/types.py
class ModelRole(str, Enum):
    CONTROLLER = "controller"
    PUBLIC = "public"
    SUBCALL = "subcall"


@dataclass(frozen=True, slots=True)
class ModelCallContext:
    run_id: str
    branch_id: str
    call_id: str
    role: ModelRole
    depth: int
```

Export all new public configuration/specification types from `src/rlm/__init__.py`. Replace the old `Prompt` configuration type with `PromptSpec`; `default_harness_spec()` in `prompts.py` constructs the current prompt content until Task 5 simplifies it.

- [ ] **Step 6: Run focused and full tests**

Run: `uv run pytest tests/test_specs.py tests/test_recovery.py tests/test_cli.py tests/test_engine.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: the complete existing suite passes after the mechanical configuration migration.

- [ ] **Step 7: Review the configuration checkpoint without staging**

```bash
git diff --check
git diff --stat -- src/rlm tests
```

---

### Task 2: Strict Zero-Overhead Trace Sink

**Files:**
- Create: `src/rlm/json.py`
- Create: `tests/test_json.py`
- Create: `tests/test_trace.py`
- Create: `tests/trace_helpers.py`
- Create: `tests/fixtures/trace-schema-v1.json`
- Modify: `src/rlm/trace.py`
- Modify: `src/rlm/errors.py`
- Modify: `src/rlm/backend.py`
- Modify: `src/rlm/config.py`
- Modify: `src/rlm/prompts.py`
- Modify: `src/rlm/specs.py`
- Modify: `src/rlm/response.py`
- Modify: `src/rlm/executor.py`
- Modify: `src/rlm/engine.py`
- Modify: `tests/test_specs.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- Produces: one `strict_json_loads`, `strict_json_dumps`, `strict_json_sha256`, and
  `json_compatibility_error` contract shared by all boundaries.
- Consumes: `TraceConfig`, `HarnessSpec.fingerprint()`.
- Produces: the internal `TraceSink` effect boundary, a true `NullTraceSink`, and
  `EventRef = str | None`.
- Produces: closed `TraceEventKind` values and one reviewed golden trace-schema contract.
- Produces: immutable `ErrorRecord` and exact `TraceManifest` terminal metadata.
- Produces: `make_trace_sink(...)`, enabled `TraceRecorder.event(...) -> str | None`, and
  schema-versioned strict JSONL.
- Produces: fatal `TraceError` for enabled trace failures.

- [ ] **Step 1: Add failing trace regression tests**

```python
# tests/test_json.py
import pytest

from rlm.json import StrictJSONError, strict_json_loads


@pytest.mark.parametrize("text", ['{"key": 1, "key": 2}', "NaN", "Infinity"])
def test_strict_json_rejects_non_json_decoder_extensions(text: str) -> None:
    with pytest.raises(StrictJSONError):
        strict_json_loads(text)
```

```python
# tests/test_trace.py
import json
from pathlib import Path

import pytest

from rlm import TraceConfig
from rlm.errors import TraceError, UpstreamError
from rlm.trace import (
    TRACE_ENVELOPE_FIELDS,
    TRACE_MANIFEST_FIELDS,
    TRACE_SCHEMA_VERSION,
    NullTraceSink,
    RunStartedPayload,
    TraceEventKind,
    TraceManifest,
    TraceRecorder,
    trace_payload_schema,
)
from tests.trace_helpers import completed_manifest, failed_manifest, run_started_payload


class NoCopy:
    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise AssertionError("disabled tracing copied a payload")


def test_disabled_trace_does_no_payload_work() -> None:
    def unexpected_manifest() -> TraceManifest:
        raise AssertionError("disabled tracing built a manifest")

    payload = run_started_payload(request={"opaque": NoCopy()})
    trace = NullTraceSink()
    assert trace.event(payload, branch_id="root", depth=0) is None
    trace.close(manifest_factory=unexpected_manifest)


def test_jsonl_only_trace_does_not_retain_events(tmp_path: Path) -> None:
    trace = TraceRecorder(
        TraceConfig(enabled=True, directory=tmp_path, markdown=False), run_id="run"
    )
    trace.event(run_started_payload(), branch_id="root", depth=0)
    trace.close(manifest_factory=lambda: completed_manifest("run"))

    assert trace.events == []
    event = json.loads((trace.directory / "trace.jsonl").read_text().splitlines()[0])
    assert event["schema_version"] == TRACE_SCHEMA_VERSION


def test_enabled_trace_rejects_unknown_python_objects(tmp_path: Path) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")
    with pytest.raises(TraceError, match="strict JSON"):
        trace.event(
            run_started_payload(request={"value": object()}),
            branch_id="root",
            depth=0,
        )


def test_enabled_trace_rejects_events_after_close(tmp_path: Path) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")
    trace.close(manifest_factory=lambda: completed_manifest("run"))

    with pytest.raises(TraceError, match="closed"):
        trace.event(run_started_payload(), branch_id="root", depth=0)


def test_trace_finalization_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")

    def fail_write(name: str, value: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(trace, "write_json", fail_write)
    with pytest.raises(TraceError, match="finalize"):
        trace.close(manifest_factory=lambda: completed_manifest("run"))


def test_trace_failure_preserves_an_active_run_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")
    original = UpstreamError(503, {"error": {"message": "down"}}, endpoint="/responses")

    def fail_write(name: str, value: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(trace, "write_json", fail_write)

    with pytest.raises(TraceError) as caught:
        trace.close(
            manifest_factory=lambda: failed_manifest("run", original.to_record()),
            cause=original,
        )

    assert caught.value.__cause__ is original


def test_trace_schema_matches_reviewed_golden_contract() -> None:
    fixture = Path(__file__).parent / "fixtures/trace-schema-v1.json"
    contract = json.loads(fixture.read_text())

    assert contract["schema_version"] == TRACE_SCHEMA_VERSION
    assert contract["envelope_fields"] == list(TRACE_ENVELOPE_FIELDS)
    assert contract["manifest_fields"] == list(TRACE_MANIFEST_FIELDS)
    assert contract["event_types"] == [kind.value for kind in TraceEventKind]
    assert contract["required_payload_fields"] == trace_payload_schema()


def test_enabled_trace_rejects_a_dangling_causal_parent(tmp_path: Path) -> None:
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")
    trace.event(run_started_payload(), branch_id="root", depth=0)

    with pytest.raises(TraceError, match="parent_event_id"):
        trace.event(
            run_started_payload(),
            branch_id="root",
            depth=0,
            parent_event_id="missing",
        )


def test_enabled_trace_rejects_payload_shape_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = run_started_payload()
    monkeypatch.setattr(RunStartedPayload, "to_dict", lambda self: {"request": self.request})
    trace = TraceRecorder(TraceConfig(enabled=True, directory=tmp_path), run_id="run")

    with pytest.raises(TraceError, match="payload fields"):
        trace.event(payload, branch_id="root", depth=0)
```

Add an engine-level guard that fails if disabled tracing invokes payload conversion or the lazy
manifest builder:

```python
# tests/test_engine.py
def test_disabled_tracing_does_not_convert_payloads_or_build_a_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("disabled tracing performed trace-only work")

    monkeypatch.setattr(RunStartedPayload, "to_dict", unexpected)
    monkeypatch.setattr("rlm.engine._trace_manifest", unexpected)

    result = RLM(ScriptedBackend([controller_code('FINAL_TEXT("done")')])).run(request())
    assert extract_text(result.response) == "done"
```

Create exact shared readers in `tests/trace_helpers.py`; every reader uses `strict_json_loads`,
asserts the expected top-level shape, and returns typed dictionaries. It defines:

```python
def traced_config(directory: Path, *, max_turns: int = 12) -> RLMConfig:
    return RLMConfig(
        limits=replace(RunLimits(), max_turns=max_turns),
        tracing=TraceConfig(enabled=True, directory=directory),
    )


def read_events(directory: Path | None) -> list[dict[str, Any]]: ...


def read_manifest(directory: Path | None) -> dict[str, Any]: ...


def read_trace_contract() -> dict[str, Any]: ...


def run_started_payload(*, request: dict[str, Any] | None = None) -> RunStartedPayload: ...


def completed_manifest(run_id: str) -> TraceManifest: ...


def failed_manifest(run_id: str, error: ErrorRecord) -> TraceManifest: ...
```

- [ ] **Step 2: Establish unmasked JSON and trace failures**

Run: `uv run pytest tests/test_json.py -q`

Expected: FAIL because the shared strict JSON module does not exist. Add and finish only the
strict-JSON functions described below, migrate the minimum imports needed for collection, then
rerun `tests/test_json.py` to PASS.

Next add the trace enums/payload/manifest declarations with method stubs and run each new trace test
by name. Expected behavioral failures: disabled tracing invokes payload or manifest work,
JSONL-only tracing retains events, unknown objects become `repr`, dangling parents and malformed
payload shapes are accepted, events after close are accepted, finalization errors are hidden or
lose their active cause, and typed/golden schema checks fail. Do not implement the trace behavior
until each applicable assertion has failed independently.

- [ ] **Step 3: Implement strict trace behavior**

The `rlm.json` module established in Step 2 is the sole JSON semantics module.
`strict_json_loads` uses `json.loads` with a
duplicate-key-rejecting `object_pairs_hook` and a `parse_constant` callback that rejects `NaN` and
infinities, then calls the recursive `json_compatibility_error`. `strict_json_dumps` validates
before encoding with `allow_nan=False`; `strict_json_sha256` hashes that one canonical encoding
with sorted keys and compact separators. Move `json_compatibility_error` out of `response.py` and
migrate all imports, including `HarnessSpec` canonicalization; do not retain a second
implementation or permissive compatibility alias.

Define the effect boundary used by the engine:

```python
EventRef = str | None


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    exception_type: str
    message: str
    code: str
    status_code: int
    public_error_type: str
    parameter: str | None
    body: dict[str, Any]
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]: ...


class TraceEventKind(str, Enum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    BRANCH_STARTED = "branch.started"
    BRANCH_COMPLETED = "branch.completed"
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MODEL_FAILED = "model.failed"
    CONTROLLER_ACTION = "controller.action"
    CONTROLLER_EXECUTION = "controller.execution"
    CONTROLLER_OBSERVATION = "controller.observation"
    CONTROLLER_RECOVERY_DECISION = "controller.recovery_decision"
    CONTROLLER_REPAIR = "controller.repair"


@dataclass(frozen=True, slots=True)
class ModelRequestPayload:
    kind: ClassVar[TraceEventKind] = TraceEventKind.MODEL_REQUEST
    context: ModelCallContext
    request: dict[str, Any]

    def to_dict(self) -> dict[str, Any]: ...


# Define one frozen payload dataclass for every TraceEventKind. Each class owns its
# kind and exact fields; representative additional classes are RunStartedPayload,
# RunCompletedPayload, RunFailedPayload, ModelResponsePayload, ModelFailedPayload,
# ControllerActionPayload, ControllerExecutionPayload, RecoveryDecisionPayload,
# and RepairPayload.
TracePayload = (
    RunStartedPayload
    | RunCompletedPayload
    | RunFailedPayload
    | BranchStartedPayload
    | BranchCompletedPayload
    | ModelRequestPayload
    | ModelResponsePayload
    | ModelFailedPayload
    | ControllerActionPayload
    | ControllerExecutionPayload
    | ControllerObservationPayload
    | RecoveryDecisionPayload
    | RepairPayload
)


class RunTraceStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TraceManifest:
    run_id: str
    status: RunTraceStatus
    stop_reason: str | None
    turns: int
    duration_seconds: float
    usage: dict[str, Any]
    error: ErrorRecord | None
    harness: dict[str, Any]
    harness_fingerprint: str
    effective_config: dict[str, Any]
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]: ...


ManifestFactory = Callable[[], TraceManifest]
TRACE_MANIFEST_FIELDS = tuple(field.name for field in fields(TraceManifest))


@dataclass(frozen=True, slots=True)
class TraceEvent:
    schema_version: str
    run_id: str
    event_id: str
    parent_event_id: EventRef
    branch_id: str
    depth: int
    timestamp: float
    type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]: ...


TRACE_ENVELOPE_FIELDS = tuple(field.name for field in fields(TraceEvent))


class TraceSink(Protocol):
    directory: Path | None

    def event(
        self,
        payload: TracePayload,
        *,
        branch_id: str,
        depth: int,
        parent_event_id: EventRef = None,
    ) -> EventRef: ...

    def close(
        self,
        *,
        manifest_factory: ManifestFactory,
        cause: BaseException | None = None,
    ) -> None: ...
```

Type engine helpers against `TraceSink`. `make_trace_sink` returns a stateless `NullTraceSink`
when tracing is disabled and an enabled `TraceRecorder` otherwise; `TraceRecorder` itself never
has a disabled mode. This interface exists for deterministic effect tests, not as a
user-selectable trace policy. `NullTraceSink.event` returns `None` without inspecting its payload,
and `close` returns without invoking `manifest_factory`. The engine constructs lightweight frozen
payload values whose constructors neither copy nor validate nested data; it performs no payload
conversion or copy at call sites. The enabled sink owns those checks. The engine passes a closure
around the pure `_trace_manifest(...)` builder so all manifest shaping remains lazy.
`trace_payload_schema()` derives event kinds and required keys from the closed union's dataclass
fields and rejects missing, duplicate, or extra enum coverage at import time. Event callers cannot
pass an independent kind string or untyped mapping. The enabled sink accepts only an exact member
type of that closed union, calls `to_dict()` once, and verifies that the resulting key set exactly
matches the derived fields before any write. It maintains the set of written event IDs:
`run.started` must have no parent, while every later event must name an already-written parent.
A type-correct object with a defective serializer or a dangling causal reference is a fatal
`TraceError`, not canonical-looking incomplete JSONL.

Add `TRACE_SCHEMA_VERSION = "1"`. The factory returns the null sink before allocating a lock,
directory, or buffer. Append to `self.events` only when Markdown rendering is enabled. Serialize
with `allow_nan=False`; the only explicit non-JSON conversion permitted inside the sink is
`Path -> str`. An event attempted after an enabled sink is closed raises `TraceError`; repeated
`close()` remains idempotent. Wrap validation, copy, serialization, open, write, flush, and close
failures in:

```python
class TraceError(RLMError):
    def __init__(self, message: str):
        super().__init__(message, code="trace_error")
```

Construct the event envelope only through:

```python
event = TraceEvent(
    schema_version=TRACE_SCHEMA_VERSION,
    run_id=self.run_id,
    event_id=event_id,
    parent_event_id=parent_event_id,
    branch_id=branch_id,
    depth=depth,
    timestamp=time.time(),
    type=payload.kind.value,
    payload=copy.deepcopy(payload.to_dict()),
).to_dict()
```

Remove the `to_dict()`/`repr()` fallback. Update engine event-reference annotations to accept `None`. Include `harness`, `harness_fingerprint`, and `trace_schema_version` in `run.started`.

Derive `TRACE_ENVELOPE_FIELDS` from `TraceEvent`; do not maintain a second envelope key list.
The reviewed `trace-schema-v1.json` fixture pins those envelope fields, every `TraceEventKind`,
manifest fields, and required payload keys for model context, recovery decisions, observations,
terminal state, and errors. It also declares the exact deterministic direct, repair, recursive, and
backend-failure event sequences exercised in Tasks 3 and 6. The fixture is a test oracle only—runtime dispatch uses the
enum and typed values, never a parallel string lookup. Enabled `close()` requires a factory that
returns exactly `TraceManifest`, so a schema-versioned run cannot finalize with missing or extra
manifest fields. Task 6 adds end-to-end causal sequence, role, exact payload, failure-body, and
manifest assertions after the state machine exists.

Enabled `close()` invokes its manifest factory exactly once and is subject to the same strict
contract. If construction or finalization fails it raises `TraceError`.
`RLMError.to_record()` creates the exact strict record above, including its public body. When
another run exception is already active, the engine passes it as `cause`; a finalization failure
raises `TraceError` explicitly chained from that original exception so neither defect disappears.
On successful close, flush and `fsync` JSONL, write any Markdown derived view, then atomically write
and `fsync` the manifest and parent directory last. A missing manifest therefore marks an
interrupted or failed trace finalization rather than a falsely completed episode.

- [ ] **Step 4: Run trace and engine tests**

Run: `uv run pytest tests/test_json.py tests/test_trace.py tests/test_engine.py -q`

Expected: PASS.

- [ ] **Step 5: Review the strict-tracing checkpoint without staging**

```bash
git diff --check
git diff --stat -- src/rlm tests
```

---

### Task 3: Responses Contract, Usage Integrity, and Deadline-Aware Backend

**Files:**
- Create: `tests/test_response.py`
- Modify: `src/rlm/response.py`
- Modify: `src/rlm/backend.py`
- Modify: `src/rlm/errors.py`
- Modify: `src/rlm/ledger.py`
- Modify: `src/rlm/prompts.py`
- Modify: `src/rlm/types.py`
- Modify: `src/rlm/engine.py`
- Modify: `src/rlm/server.py`
- Modify: `tests/fakes.py`
- Modify: `tests/test_backend.py`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_server.py`

**Interfaces:**
- Consumes: `ModelCallContext`, `ModelRole`, remaining timeout from `Ledger`.
- Produces: `validate_response_envelope(value) -> str | None`.
- Produces: `validate_terminal_response(value) -> str | None`.
- Produces: `ModelBackend.complete(request, *, timeout, context)`.
- Produces: optional runtime-checkable `ModelCatalogBackend.list_models()` capability.
- Produces: explicit `TokenUsage.unreported_calls`.

- [ ] **Step 1: Add failing Responses validation tests**

```python
# tests/test_response.py
from tests.fakes import responses_text

from rlm.response import validate_response_envelope, validate_terminal_response


def test_terminal_response_requires_complete_responses_envelope() -> None:
    assert validate_terminal_response({"output": [{}]}) is not None
    assert validate_terminal_response({"output": [{"type": "bogus"}]}) is not None


def test_valid_unknown_output_item_is_preserved() -> None:
    response = responses_text("done")
    response["output"] = [{"id": "future_1", "type": "future_terminal", "payload": {"value": 1}}]
    assert validate_terminal_response(response) is None


def test_reasoning_only_response_is_not_terminal() -> None:
    response = responses_text("done")
    response["output"] = [{"id": "rs_1", "type": "reasoning", "summary": []}]
    assert validate_response_envelope(response) is None
    assert validate_terminal_response(response) == "Responses response contains no terminal output"


def test_empty_text_is_structurally_valid_but_strict_helpers_may_reject_it() -> None:
    response = responses_text("")
    assert validate_response_envelope(response) is None
    assert validate_terminal_response(response) is None
```

- [ ] **Step 2: Add failing HTTP and usage tests**

Use `request` and backend doubles from `tests.fakes`; import `traced_config`, `read_events`, and
`read_trace_contract` from `tests.trace_helpers`. Define `CALL_CONTEXT` exactly as shown rather than
relying on hidden module state.

```python
# tests/test_backend.py
CALL_CONTEXT = ModelCallContext(
    run_id="run-test",
    branch_id="branch-root",
    call_id="call-1",
    role=ModelRole.PUBLIC,
    depth=0,
)


def test_successful_non_json_response_is_an_upstream_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not json"))
    endpoint = OpenAIEndpoint(
        base_url="https://provider.test/v1", client=httpx.Client(transport=transport)
    )

    with pytest.raises(UpstreamError, match="non-JSON success body"):
        endpoint.complete(request(), timeout=5, context=CALL_CONTEXT)


def test_successful_non_json_models_body_is_an_upstream_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not json"))
    endpoint = OpenAIEndpoint(
        base_url="https://provider.test/v1",
        client=httpx.Client(transport=transport),
    )

    with pytest.raises(UpstreamError, match="non-JSON success body"):
        endpoint.list_models()


def test_endpoint_uses_smaller_call_timeout() -> None:
    seen: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(float(request.extensions["timeout"]["read"]))
        return httpx.Response(200, json=responses_text("done"))

    endpoint = OpenAIEndpoint(
        base_url="https://provider.test/v1",
        timeout=300,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    endpoint.complete(request(), timeout=0.25, context=CALL_CONTEXT)
    assert seen == [pytest.approx(0.25)]
```

```python
# tests/test_engine.py
def test_missing_usage_is_reported_not_silently_counted_as_complete() -> None:
    response = controller_code('FINAL_TEXT("done")')
    response.pop("usage")
    result = RLM(ScriptedBackend([response])).run(request())
    assert result.usage.unreported_calls == 1


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"input_tokens": True, "output_tokens": 0, "total_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1, "total_tokens": 0},
        {"input_tokens": 1.5, "output_tokens": 1, "total_tokens": 2.5},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 3},
    ],
)
def test_present_usage_must_be_complete_nonnegative_integers_with_exact_total(
    usage: dict[str, Any],
) -> None:
    response = controller_code('FINAL_TEXT("done")')
    response["usage"] = usage

    with pytest.raises(BackendProtocolError, match="usage"):
        RLM(ScriptedBackend([response])).run(request())


def test_configured_token_budget_requires_reported_usage() -> None:
    response = controller_code('FINAL_TEXT("done")')
    response.pop("usage")
    config = RLMConfig(limits=RunLimits(max_total_tokens=100))

    with pytest.raises(BackendProtocolError, match="token budget requires usage"):
        RLM(ScriptedBackend([response]), config=config).run(request())


def test_aggregate_token_budget_is_fatal_when_exceeded() -> None:
    response = controller_code('FINAL_TEXT("done")', input_tokens=3, output_tokens=2)
    config = RLMConfig(limits=RunLimits(max_total_tokens=4))

    with pytest.raises(LimitExceededError, match="token limit"):
        RLM(ScriptedBackend([response]), config=config).run(request())


def test_failed_backend_response_is_fatal() -> None:
    response = responses_text("")
    response.update(status="failed", error={"code": "provider_failed", "message": "boom"})

    with pytest.raises(BackendResponseError, match="boom"):
        RLM(ScriptedBackend([response])).run(request())


def test_nonterminal_backend_response_is_a_protocol_error() -> None:
    response = responses_text("")
    response["status"] = "in_progress"

    with pytest.raises(BackendProtocolError, match="nonterminal status"):
        RLM(ScriptedBackend([response])).run(request())


def test_final_text_fails_closed_for_unknown_request_semantics() -> None:
    public_request = {**request(), "future_provider_mode": {"enabled": True}}
    controller = ScriptedBackend(
        [
            controller_code('FINAL_TEXT("unsafe")'),
            controller_code(f"FINAL_RESPONSE({responses_text('safe')!r})"),
        ]
    )

    result = RLM(controller, config=config_with(max_turns=2)).run(public_request)
    assert extract_text(result.response) == "safe"


def test_synthetic_response_does_not_claim_complete_usage_when_unreported() -> None:
    response = controller_code('FINAL_TEXT("done")')
    response.pop("usage")
    result = RLM(ScriptedBackend([response])).run(request())

    assert result.response["usage"] is None


def test_direct_baseline_has_canonical_public_trace(tmp_path: Path) -> None:
    public = ScriptedBackend([responses_text("direct", model="public-model")])
    result = RLM(public, config=traced_config(tmp_path)).run_direct(request())

    assert extract_text(result.response) == "direct"
    assert result.stop_reason == "direct"
    assert result.turns == 0
    assert public.contexts[0].role is ModelRole.PUBLIC
    events = read_events(result.trace_directory)
    contract = read_trace_contract()
    assert [event["type"] for event in events] == contract["canonical_direct_sequence"]
    model_response = next(event for event in events if event["type"] == "model.response")
    assert model_response["payload"]["context"]["role"] == "public"
    assert extract_text(model_response["payload"]["response"]) == "direct"


def test_close_attempts_every_distinct_backend_and_reports_all_failures() -> None:
    first = FailingCloseBackend("first")
    second = FailingCloseBackend("second")
    rlm = RLM(first, controller_backend=second)

    with pytest.raises(BackendCloseError) as caught:
        rlm.close()

    assert first.close_calls == second.close_calls == 1
    assert [record.backend_name for record in caught.value.failures] == ["first", "second"]


def test_same_backend_is_closed_exactly_once() -> None:
    backend = RecordingCloseBackend()
    RLM(backend, controller_backend=backend).close()
    assert backend.close_calls == 1
```

```python
# tests/test_server.py
@pytest.mark.parametrize(
    "body",
    [b'{"model":"m","input":"a","input":"b"}', b'{"model":"m","temperature":NaN}'],
)
def test_public_request_rejects_non_strict_json(body: bytes) -> None:
    app = create_app(RLM(ScriptedBackend()), close_on_shutdown=False)
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"
```

- [ ] **Step 3: Run focused tests and verify failures**

Run: `uv run pytest tests/test_response.py tests/test_backend.py tests/test_engine.py tests/test_server.py -q`

Expected: FAIL because envelope/terminal validators and deadline-aware backend arguments do not exist, and a non-JSON 200 currently becomes a successful message object.

- [ ] **Step 4: Implement envelope and terminal validators**

`validate_response_envelope` must check, in order:

1. strict JSON object;
2. non-empty string `id`;
3. `object == "response"`;
4. non-empty string `model`;
5. recognized Responses status string;
6. a `failed` response has an error object with a non-empty message and an optional string code;
7. list-valued `output`;
8. every output item is an object with a non-empty string `type` and `id`;
9. `message` items have assistant role, recognized status, list content, and typed content objects;
10. `output_text` and `refusal` parts contain their required string field; an empty string is
   envelope-valid but not usable terminal output.

`validate_terminal_response` first calls the envelope validator, then requires `status == "completed"`, no response error, a non-empty output list, and at least one structurally valid non-`reasoning` item. Empty text remains structurally valid at this wire boundary; `ask` and task-specific benchmark extractors impose their own non-empty-answer contract. Unknown item types remain valid only when they satisfy the generic `id`, `type`, and non-empty payload requirements.

Use these explicit protocol sets rather than accepting arbitrary status strings:

```python
RESPONSE_STATUSES = frozenset(
    {"cancelled", "completed", "failed", "in_progress", "incomplete", "queued"}
)
OUTPUT_ITEM_STATUSES = frozenset({"completed", "in_progress", "incomplete"})
```

For an unknown output item, "non-empty payload" means at least one field other than `id`,
`type`, and optional `status`; the validator must not guess at future provider-specific fields.

Use `rlm.json.json_compatibility_error` as the one recursive strict-JSON validator. Rename current
`validate_response` call sites to the appropriate envelope or terminal function. Parse HTTP bodies
with `strict_json_loads(response.content)` rather than `response.json()` so duplicate keys and
non-finite constants cannot disappear before validation.

Likewise, the FastAPI adapter reads `await request.body()` and passes those bytes through
`strict_json_loads`; do not use `Request.json()`. Require a top-level object after strict decoding
and preserve every unknown field in that object.

Move the synthetic-text compatibility rule out of `prompts.py` and into `response.py`. Implement it
as a fail-closed allowlist of request fields whose semantics the minimal response constructor can
faithfully preserve; any unknown field or known tool/structured-output/continuation behavior makes
`FINAL_TEXT` a recoverable final-submission fault requiring `FINAL_RESPONSE`. The allowlist is the
single versioned wire contract, not a list of known-unsafe features. Validate every constructed
text response with `validate_terminal_response` as an internal invariant.

- [ ] **Step 5: Make transport responses and usage explicit**

Change the backend protocol and implementation:

```python
class ModelBackend(Protocol):
    def complete(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float,
        context: ModelCallContext,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ModelCatalogBackend(Protocol):
    def list_models(self) -> dict[str, Any]: ...


@runtime_checkable
class CloseableBackend(Protocol):
    def close(self) -> None: ...
```

The `/v1/models` adapter checks `isinstance(rlm.backend, ModelCatalogBackend)` and either calls the
typed capability or returns the explicit existing `models_not_supported` error. Do not use dynamic
`getattr` dispatch.

Use `CloseableBackend` for optional backend cleanup as well; close each distinct backend exactly
once without name-based reflection. If one close raises an ordinary exception, record it, continue
closing the remaining distinct backends, then raise a typed `BackendCloseError` chained from the
first cause and containing every backend/exception type and message. Do not suppress cleanup
defects.

`OpenAIEndpoint.__init__` and `complete` reject boolean, non-finite, and non-positive timeouts;
`complete` passes `min(self.timeout, timeout)` to httpx. On JSON decode failure:

```python
if response.is_success:
    raise UpstreamError(
        502,
        {"message": "upstream returned a non-JSON success body", "body": response.text},
        endpoint=endpoint,
    )
payload = {"message": response.text, "content_type": response.headers.get("content-type")}
```

Add `BackendProtocolError` for malformed successful Responses objects and
`BackendResponseError` for a structurally valid `failed` or `cancelled` response. Update
`_call_model` to construct `ModelCallContext` and pass `ledger.timeout(call_cap)`. Immediately after
backend completion, check the ledger, verify the return is a strict JSON object, trace that exact
raw response, validate its envelope and usage, record reported consumption, then interpret its
status. Accept only `completed` or `incomplete`
from the synchronous non-background
backend contract; `queued` or `in_progress` is a fatal `BackendProtocolError`. A failed/cancelled
response is a fatal `BackendResponseError` whose message and trace retain its structured error.

Use the immutable `ErrorRecord` introduced in Task 2. A helper converts an unexpected ordinary
exception to an infrastructure record without `repr`. `RemoteRLMError.from_record()` restores status, code,
parameter, details, and `to_body()` from the record without guessing a concrete subclass. Use
`ErrorRecord.to_dict()` for every `model.failed`, `run.failed`, executor host failure, and manifest
error payload. `UpstreamError` details include endpoint while its record retains the exact parsed
body or explicitly labeled raw body.

Extend `TokenUsage` with `unreported_calls: int = 0`. If `usage` is absent or null, increment that field. If a present token count is boolean, negative, or non-integer, raise `BackendProtocolError`; never convert it to zero. Validate `total_tokens` and require that it equals `input_tokens + output_tokens`.

Expose the aggregate as `X-RLM-Usage-Unreported-Calls` beside the existing usage headers so an HTTP
experiment client cannot mistake partial token telemetry for complete accounting.

When `unreported_calls > 0`, a synthetic `FINAL_TEXT` response sets standard `usage` to `null`
rather than publishing partial totals as though they were complete. The exact partial counters and
completeness field remain in the run trace, headers, and `RunResult`.

After every recorded response, the ledger checks `RunLimits.max_total_tokens`. Exceeding it raises
`LimitExceededError(code="token_limit")`. If that limit is configured and usage is missing, raise
`BackendProtocolError` before accepting the response because the budget cannot be audited.

Replace the old unbounded `direct` backend call with `run_direct(request) -> RunResult`. It uses the
same request validation, deadline-aware `_call_model`, usage ledger, strict trace lifecycle, and
terminal response validation, but makes exactly one `ModelRole.PUBLIC` call and reports zero turns
with stop reason `direct`. Keep `direct(request) -> dict` as the intentional response-only sibling
of `complete`, implemented only as `return self.run_direct(request).response`; it is a baseline API,
never an error fallback.

- [ ] **Step 6: Update deterministic fakes to honor the backend contract**

Every fake backend accepts keyword-only `timeout` and `context`, records both, and otherwise preserves existing scripted behavior:

```python
def complete(
    self,
    request: Mapping[str, Any],
    *,
    timeout: float,
    context: ModelCallContext,
) -> dict[str, Any]:
    self.requests.append(copy.deepcopy(dict(request)))
    self.timeouts.append(timeout)
    self.contexts.append(context)
    return self._next_response(request)
```

Also add explicit close doubles used above:

```python
class RecordingCloseBackend(ScriptedBackend):
    def __init__(self, name: str = "recording") -> None:
        super().__init__(name=name)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FailingCloseBackend(RecordingCloseBackend):
    def close(self) -> None:
        super().close()
        raise RuntimeError(f"{self.name} close failed")
```

- [ ] **Step 7: Run focused and full tests**

Run: `uv run pytest tests/test_response.py tests/test_backend.py tests/test_engine.py tests/test_server.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 8: Review the transport checkpoint without staging**

```bash
git diff --check
git diff --stat -- src/rlm tests
```

---

### Task 4: Versioned ABI and Structured IPython Outcomes

**Files:**
- Create: `src/rlm/abi.py`
- Create: `tests/test_abi.py`
- Modify: `src/rlm/errors.py`
- Modify: `src/rlm/executor.py`
- Modify: `src/rlm/specs.py`
- Modify: `src/rlm/prompts.py`
- Modify: `src/rlm/trace.py`
- Modify: `src/rlm/types.py`
- Modify: `src/rlm/engine.py`
- Modify: `tests/test_executor.py`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_specs.py`
- Modify: `tests/test_trace.py`
- Modify: `tests/fixtures/trace-schema-v1.json`

**Interfaces:**
- Produces: `HostOperation`, `FunctionSpec`, and `ENVIRONMENT_ABI` version `1`.
- Produces: canonical `EnvironmentABI.to_dict()` and digest bound into `HarnessSpec`.
- Produces: `HostRequestPayload | HostBatchPayload` after strict IPC decoding.
- Produces: `ExecutionEnvironment` and `ExecutionEnvironmentFactory` effect boundaries.
- Produces: `ExecutionException` independent of stdout/stderr.
- Produces: closed `TextSubmission | ResponseSubmission | None` execution outcomes.
- Produces: `AskBatchResult` with index-aligned successes and failures.
- Changes: action handler receives `(HostOperation, payload, timeout_seconds)`.

- [ ] **Step 1: Add failing structured-execution tests**

```python
# tests/test_executor.py
def test_stderr_does_not_make_a_successful_final_fail() -> None:
    with IPythonExecutor(request=request(), allow_recursion=False) as executor:
        result = executor.execute(
            'import sys\nprint("warning", file=sys.stderr)\nFINAL_TEXT("done")',
            action_handler=lambda operation, payload, timeout: None,
            timeout=10,
        )

    assert result.exception is None
    assert result.stderr == "warning\n"
    assert result.submission == TextSubmission("done")


def test_exception_is_structured_and_discards_earlier_final() -> None:
    with IPythonExecutor(request=request(), allow_recursion=False) as executor:
        result = executor.execute(
            'FINAL_TEXT("wrong")\nraise ValueError("cell failed")',
            action_handler=lambda operation, payload, timeout: None,
            timeout=10,
        )

    assert result.exception.type == "ValueError"
    assert result.exception.message == "cell failed"
    assert result.submission is None


def test_multiple_finals_are_a_structured_submission_fault() -> None:
    with IPythonExecutor(request=request(), allow_recursion=False) as executor:
        result = executor.execute(
            'FINAL_TEXT("first")\nFINAL_TEXT("second")',
            action_handler=lambda operation, payload, timeout: None,
            timeout=10,
        )

    assert result.exception.type == "FinalSubmissionFault"
    assert result.submission is None
```

```python
def test_ask_batch_preserves_each_failed_index_without_global_state() -> None:
    responses = [responses_text("kept"), responses_text(""), responses_text("later")]

    def handle(operation: HostOperation, payload: HostPayload, timeout: float) -> object:
        assert timeout > 0
        if operation is HostOperation.MODEL_COMPLETE_BATCH:
            assert isinstance(payload, HostBatchPayload)
            return [responses.pop(0) for _ in payload.requests]
        assert operation is HostOperation.MODEL_COMPLETE
        assert isinstance(payload, HostRequestPayload)
        return responses.pop(0)

    with IPythonExecutor(request=request(), allow_recursion=False) as executor:
        result = executor.execute(
            "batch = ask_batch(['a', 'b'])\nother = ask('c')\nprint(batch.failed_indexes)",
            action_handler=handle,
            timeout=10,
        )

    assert result.exception is None
    assert result.stdout.strip() == "(1,)"
    assert result.namespace["batch"] == "AskBatchResult(len=2, failures=1)"


def test_fatal_host_failure_is_separate_from_controller_exception() -> None:
    def fail(operation: HostOperation, payload: HostPayload, timeout: float) -> object:
        raise UpstreamError(503, {"message": "down"}, endpoint="/responses")

    with IPythonExecutor(request=request(), allow_recursion=False) as executor:
        result = executor.execute('ask("leaf")', action_handler=fail, timeout=10)

    assert result.host_failure is not None
    assert result.host_failure.kind is HostFailureKind.RLM_ERROR
    assert result.host_failure.error.code == "upstream_error"
    assert result.submission is None


def test_late_synchronous_handler_result_is_rejected_after_return() -> None:
    seen_timeouts: list[float] = []

    def slow_handler(
        operation: HostOperation,
        payload: HostPayload,
        timeout: float,
    ) -> dict[str, str]:
        seen_timeouts.append(timeout)
        time.sleep(0.06)
        return {"too": "late"}

    with IPythonExecutor(request=request(), allow_recursion=False) as executor:
        started = time.monotonic()
        with pytest.raises(ExecutionTimeoutError):
            executor.execute(
                "model_complete({})",
                action_handler=slow_handler,
                timeout=0.02,
            )

    assert seen_timeouts and 0 < seen_timeouts[0] <= 0.02
    assert time.monotonic() - started >= 0.06
```

- [ ] **Step 2: Add failing ABI consistency test**

```python
# tests/test_abi.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

import pytest
from rlm.abi import (
    ENVIRONMENT_ABI,
    EnvironmentABI,
    HostOperation,
    environment_function,
    validate_environment_bindings,
)
from rlm.json import json_compatibility_error

from rlm import RLM, RLMConfig
from rlm.errors import HarnessContractError, HostProtocolError
from rlm.executor import ActionHandler, IPythonExecutor
from rlm.prompts import default_harness_spec
from rlm.protocol import extract_text
from rlm.types import ExecutionResult, TextSubmission
from tests.fakes import ScriptedBackend, controller_code, request


def test_environment_abi_has_unique_names_and_operations() -> None:
    assert ENVIRONMENT_ABI.version == "1"
    assert len(ENVIRONMENT_ABI.functions) == len({item.name for item in ENVIRONMENT_ABI.functions})
    host_operations = [
        item.metadata.operation
        for item in ENVIRONMENT_ABI.functions
        if item.metadata.operation is not None
    ]
    assert len(host_operations) == len(set(host_operations))
    assert HostOperation.MODEL_COMPLETE in host_operations
    assert IPythonExecutor.abi_version == ENVIRONMENT_ABI.version
    assert all("self" not in item.render() for item in ENVIRONMENT_ABI.functions)
    assert json_compatibility_error(ENVIRONMENT_ABI.to_dict()) is None


def test_harness_fingerprint_includes_exact_environment_abi() -> None:
    spec = default_harness_spec()
    first = ENVIRONMENT_ABI.functions[0]
    changed_first = replace(
        first,
        metadata=replace(first.metadata, description=first.metadata.description + " changed"),
    )
    changed = replace(
        ENVIRONMENT_ABI,
        functions=(changed_first, *ENVIRONMENT_ABI.functions[1:]),
    )

    assert spec.abi_digest == ENVIRONMENT_ABI.digest()
    assert changed.digest() != ENVIRONMENT_ABI.digest()
    assert replace(spec, abi_digest=changed.digest()).fingerprint() != spec.fingerprint()


def test_run_rejects_stale_abi_digest_before_a_model_call() -> None:
    backend = ScriptedBackend([controller_code('FINAL_TEXT("unused")')])
    harness = replace(default_harness_spec(), abi_digest="0" * 64)

    with pytest.raises(HarnessContractError, match="ABI digest"):
        RLM(backend, config=RLMConfig(harness=harness)).run(request())

    assert backend.calls == []


def test_worker_binding_rejects_signature_drift_and_extra_callable() -> None:
    class TinyNamespace(Protocol):
        @staticmethod
        @environment_function(description="Return text.")
        def ask(prompt: str) -> str: ...

    def correct(prompt: str) -> str:
        return prompt

    def wrong() -> str:
        return "wrong"

    abi = EnvironmentABI.from_protocol(TinyNamespace, version="test", request_name="request")
    validate_environment_bindings(abi, {"ask": correct})

    with pytest.raises(HostProtocolError, match="signature"):
        validate_environment_bindings(abi, {"ask": wrong})
    with pytest.raises(HostProtocolError, match="undocumented"):
        validate_environment_bindings(abi, {"ask": correct, "extra": correct})


def test_host_operation_rejects_the_wrong_payload_shape() -> None:
    with pytest.raises(HostProtocolError, match="model_complete"):
        HostOperation.MODEL_COMPLETE.decode_payload({"requests": []})


def test_explicit_falsey_execution_factory_is_used() -> None:
    class FinalEnvironment:
        abi_version = ENVIRONMENT_ABI.version

        def __enter__(self) -> FinalEnvironment:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def execute(
            self,
            code: str,
            *,
            action_handler: ActionHandler,
            timeout: float,
        ) -> ExecutionResult:
            return ExecutionResult(submission=TextSubmission("done"))

    class FalseyFactory:
        def __init__(self) -> None:
            self.called = False

        def __bool__(self) -> bool:
            return False

        def __call__(self, **kwargs: Any) -> FinalEnvironment:
            self.called = True
            return FinalEnvironment()

    factory = FalseyFactory()
    controller = ScriptedBackend([controller_code("FINAL_TEXT('ignored by fake')")])
    result = RLM(controller, execution_factory=factory).run(request())

    assert factory.called
    assert extract_text(result.response) == "done"
```

- [ ] **Step 3: Run the focused tests and verify failures**

Run: `uv run pytest tests/test_abi.py tests/test_executor.py -q`

Expected first: collection fails because ABI declarations do not exist. Add only the enums,
dataclass/protocol declarations, and test imports. Then run the stderr, multiple-final, batch,
host-failure, late-handler, ABI-digest, and falsey-factory tests individually and observe their
behavioral assertions fail before implementing them. In particular, do not count the missing ABI
module as evidence that stderr handling or timeout propagation is wrong.

- [ ] **Step 4: Define ABI and execution value types**

```python
# src/rlm/abi.py
@dataclass(frozen=True, slots=True)
class HostRequestPayload:
    request: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class HostBatchPayload:
    requests: tuple[dict[str, Any], ...]


HostPayload = HostRequestPayload | HostBatchPayload


class HostOperation(str, Enum):
    MODEL_COMPLETE = "model_complete"
    MODEL_COMPLETE_BATCH = "model_complete_batch"
    RLM_COMPLETE = "rlm_complete"
    RLM_COMPLETE_BATCH = "rlm_complete_batch"

    def decode_payload(self, raw: object) -> HostPayload:
        """Validate one strict wire payload and return its typed representation."""
        json_error = json_compatibility_error(raw)
        if json_error is not None:
            raise HostProtocolError(f"{self.value} payload is not strict JSON: {json_error}")
        if not isinstance(raw, dict):
            raise HostProtocolError(f"{self.value} payload must be an object")
        if self in (HostOperation.MODEL_COMPLETE, HostOperation.RLM_COMPLETE):
            if set(raw) != {"request"}:
                raise HostProtocolError(f"{self.value} payload requires only 'request'")
            request = raw["request"]
            if request is not None and not isinstance(request, dict):
                raise HostProtocolError(f"{self.value} request must be an object or null")
            return HostRequestPayload(copy.deepcopy(request))
        if set(raw) != {"requests"}:
            raise HostProtocolError(f"{self.value} payload requires only 'requests'")
        requests = raw["requests"]
        if not isinstance(requests, list) or not all(isinstance(item, dict) for item in requests):
            raise HostProtocolError(f"{self.value} requests must be a list of objects")
        return HostBatchPayload(tuple(copy.deepcopy(requests)))


@dataclass(frozen=True, slots=True)
class FunctionMetadata:
    description: str
    operation: HostOperation | None = None
    requires_recursion: bool = False


class EnvironmentNamespace(Protocol):
    @staticmethod
    @environment_function(
        description="Call the public model and return its exact Responses object.",
        operation=HostOperation.MODEL_COMPLETE,
    )
    def model_complete(request_obj: dict[str, Any] | None = None) -> dict[str, Any]: ...

    # Declare model_complete_batch, ask, ask_batch, rlm_complete,
    # rlm_complete_batch, FINAL_TEXT, FINAL_RESPONSE, and SHOW_VARS here once.
    # SHOW_VARS has the real return annotation: dict[str, str].


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    name: str
    signature: inspect.Signature
    metadata: FunctionMetadata

    def render(self) -> str:
        return f"{self.name}{self.signature}"


@dataclass(frozen=True, slots=True)
class EnvironmentABI:
    version: str
    request_name: str
    functions: tuple[FunctionSpec, ...]

    @classmethod
    def from_protocol(
        cls,
        protocol: type[EnvironmentNamespace],
        *,
        version: str,
        request_name: str,
    ) -> EnvironmentABI: ...

    def canonical_json(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...

    def digest(self) -> str: ...


ENVIRONMENT_ABI = EnvironmentABI.from_protocol(
    EnvironmentNamespace,
    version="1",
    request_name="request",
)
```

Declare every namespace method as `@staticmethod`, so rendered and worker-bound signatures never
contain an implicit `self`. The decorator stores typed metadata on each protocol method;
`EnvironmentABI.from_protocol` derives names and `inspect.Signature` values rather than accepting
hand-written signature strings. It unwraps each `staticmethod`, resolves postponed annotations in
the defining module, and orders methods by their declaration order before canonicalization.
The omitted methods in the abbreviated example are fully declared in production, including
`SHOW_VARS() -> dict[str, str]`. The worker implements that protocol, and `_worker_main` calls the
pure `validate_environment_bindings(abi, bindings)` before installing anything. The validator
requires the exact bound signature for every enabled spec and raises `HostProtocolError` for a
missing, duplicate, signature-mismatched, or undocumented RLM-owned callable. Python/IPython
builtins are outside that registry. The prompt and trace manifest render the same descriptors and
`request_name`, so namespace installation and model documentation cannot drift.

Add `abi_digest: str` to `HarnessSpec`. `default_harness_spec()` sets it to
`ENVIRONMENT_ABI.digest()`, where the digest covers version, request name, every rendered signature,
description, operation, and recursion flag in canonical order. Validate the digest before creating
a trace, ledger, executor, or model call; a mismatch raises fatal `HarnessContractError`. Record the
complete ABI plus digest in `RunStartedPayload` and `TraceManifest`, and include the digest in the
harness canonical JSON/fingerprint. Thus even an accidental description-only change changes the
reviewed golden harness and cannot alter the model prompt behind a stale fingerprint.

Add the recoverable controller-cell exception to `errors.py`:

```python
class ModelOutputFault(Exception):
    """A strict text helper received a valid response without non-empty text."""

    def __init__(self, message: str, *, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class FinalSubmissionFault(Exception):
    """A controller final cannot cross the strict final-submission boundary."""

    def __init__(self, message: str, *, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class HostProtocolError(RLMError):
    def __init__(self, message: str):
        super().__init__(message, code="host_protocol_error")
```

Add these types to `types.py`:

```python
class HostFailureKind(str, Enum):
    RLM_ERROR = "rlm_error"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True, slots=True)
class HostFailure:
    kind: HostFailureKind
    error: ErrorRecord


@dataclass(frozen=True, slots=True)
class ExecutionException:
    type: str
    message: str
    code: str | None = None
    traceback: str | None = None
    details: dict[str, Any] | None = None


class OutputChannel(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    DISPLAY = "display"
    TRACEBACK = "traceback"


@dataclass(frozen=True, slots=True)
class OutputTruncation:
    channel: OutputChannel
    original_chars: int
    retained_chars: int
    omitted_chars: int


@dataclass(frozen=True, slots=True)
class TextSubmission:
    text: str


@dataclass(frozen=True, slots=True)
class ResponseSubmission:
    response: dict[str, Any]


ExecutionSubmission = TextSubmission | ResponseSubmission


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    display: str = ""
    exception: ExecutionException | None = None
    host_failure: HostFailure | None = None
    submission: ExecutionSubmission | None = None
    namespace: dict[str, str] = field(default_factory=dict)
    truncations: tuple[OutputTruncation, ...] = ()
    duration_seconds: float = 0.0


class TextFailureKind(str, Enum):
    EMPTY_TEXT = "empty_text"
    INCOMPLETE_RESPONSE = "incomplete_response"


@dataclass(frozen=True, slots=True)
class TextFailure:
    index: int
    kind: TextFailureKind
    response: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "status": self.response.get("status"),
            "incomplete_details": self.response.get("incomplete_details"),
            "usage": self.response.get("usage"),
        }


@dataclass(frozen=True, slots=True)
class AskBatchResult:
    texts: tuple[str | None, ...]
    responses: tuple[dict[str, Any], ...]
    failures: tuple[TextFailure, ...]

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.responses):
            raise ValueError("batch texts and responses must have equal lengths")
        failed = tuple(item.index for item in self.failures)
        if len(failed) != len(set(failed)) or any(
            index < 0 or index >= len(self.texts) for index in failed
        ):
            raise ValueError("batch failures must have unique in-range indexes")
        missing = tuple(index for index, text in enumerate(self.texts) if text is None)
        if failed != missing:
            raise ValueError("batch failures must exactly identify missing texts")

    @property
    def failed_indexes(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.failures)

    def require_texts(self) -> list[str]:
        if self.failures:
            raise ModelOutputFault(
                f"ask_batch has unusable text at indexes {self.failed_indexes}",
                details={"failures": [item.summary() for item in self.failures]},
            )
        if any(text is None for text in self.texts):
            raise ValueError("batch result violates its alignment invariant")
        return [cast(str, text) for text in self.texts]
```

- [ ] **Step 5: Replace singleton helper-error state with structured results**

Remove `pending_text_error` completely. Scalar `ask` raises `ModelOutputFault` when the response is
not completed or text extraction is empty. `ask_batch` always returns `AskBatchResult`; successful
text and the exact response remain aligned with their input index, while every incomplete or empty
result becomes one `TextFailure`. Neither a later scalar call nor another batch mutates an earlier
batch result. Raw `model_complete` returns only an accepted synchronous envelope (`completed` or
`incomplete`) for explicit controller handling; `failed`, `cancelled`, `queued`, and `in_progress`
remain fatal at the backend boundary exactly as defined in Task 3. `FINAL_TEXT` and
`FINAL_RESPONSE` do not inspect global helper state.

Delete `_empty_text_reason` and all helper-authored retry advice. `TextFailureKind` distinguishes an
incomplete response from a completed response with empty text, while the exact response and its
status/usage/incomplete details remain structured data. When the cell raises `ModelOutputFault`,
copy its strict details into `ExecutionException.details` so the recovery observation contains
facts, not a hardcoded retry policy.

`FINAL_RESPONSE` raises `FinalSubmissionFault` for a non-mapping or non-strict-JSON value before IPC,
with the validation reason in structured details. A successful final becomes exactly one
`TextSubmission` or `ResponseSubmission`; there is no independent string kind/value pair.
Constructors validate and defensively copy their owned value so `ExecutionResult` cannot encode a
text kind with a response payload or vice versa. The engine classifies that exception and any
host-side terminal validator rejection as `FINAL_SUBMISSION`, rather than disguising it as an
ordinary Python exception.
Submitting more than one final in a cell raises the same fault instead of silently letting the last
call overwrite the first.

Give `_variable_summary` explicit support for `AskBatchResult` so `SHOW_VARS` reports its length and failure count without exposing contents.

- [ ] **Step 6: Separate exceptions from stderr and pass callback deadlines**

The worker sends an `exception` object independently from captured stderr. The parent constructs
`ExecutionException`. Replace `host_errors: list[dict[...]]` with one typed
`host_failure: HostFailure | None`; a cell stops at its first fatal host failure. IPC reply
discriminants are enums and each reply is decoded once into its dataclass before control logic sees
it. An `InvalidRequestError` caused by a controller-authored nested request becomes a recoverable
cell exception with its code; all other `RLMError` and unexpected host failures populate
`host_failure` and bypass recovery. Set `submission` to `None` when `exception is not None` or
`host_failure is not None`. Preserve stderr even on success.
Capture the IPython-formatted traceback in the exception field under the same hard execution-output
bound; do not scrape exception identity from stderr.

Replace independent per-stream caps with one shared retained-character budget across stdout,
stderr, display output, and traceback. Record an `OutputTruncation` entry for every truncated
channel. Truncation is explicit in the execution result and observation; no channel silently loses
text, and total retained execution output cannot exceed
`RunLimits.max_execution_output_chars`.

Change the callback type to:

```python
ActionHandler = Callable[[HostOperation, HostPayload, float], Any]
```

Define the execution effect boundary around that callback:

```python
class ExecutionEnvironment(Protocol):
    abi_version: str

    def __enter__(self) -> ExecutionEnvironment: ...

    def __exit__(self, *exc_info: object) -> None: ...

    def execute(
        self,
        code: str,
        *,
        action_handler: ActionHandler,
        timeout: float,
    ) -> ExecutionResult: ...


class ExecutionEnvironmentFactory(Protocol):
    def __call__(
        self,
        *,
        request: dict[str, Any],
        allow_recursion: bool,
        working_directory: str | Path | None,
        startup_timeout: float,
        max_output_chars: int,
    ) -> ExecutionEnvironment: ...
```

`RLM.__init__` accepts an optional typed factory and otherwise uses `IPythonExecutor`; never choose
by truthiness. On entry, the engine requires `executor.abi_version == config.harness.abi_version`
before sending controller code. This makes execution replaceable in deterministic tests without
allowing an undocumented namespace.

Before invoking it, validate the wire operation with `HostOperation(raw_operation)`, decode the
payload through `operation.decode_payload`, and pass the typed result plus remaining cell time.
Unknown operations or malformed payloads raise fatal `HostProtocolError`; controller-authored
invalid model request contents still return through the cell as a structured recoverable Python
exception, because the controller can repair those contents.

Define a private `IPCMessageKind(str, Enum)` for every worker/parent frame kind and decode it
immediately after strict JSON loading. All control flow compares enum members, never raw `"type"`
strings; unknown kinds and kind-specific missing/extra fields are fatal IPC protocol errors.

Before constructing the spawned process, encode the exact public request to strict UTF-8 JSON and
pass only those bytes plus primitive startup values as process arguments. The worker decodes those
bytes with `strict_json_loads` before creating IPython. Do not pass a Python request dictionary
through multiprocessing's implicit object serialization path.

Delete `_ActionHandlerTimeout` and `_call_action_handler`; do not create a daemon thread around the
callback. Invoke the handler synchronously with the remaining timeout. The built-in backend is
required to honor it, and the ledger checks the deadline immediately afterward. A custom backend
that ignores the typed timeout contract is a defective backend, not work the kernel hides in a
leaked thread.

Replace the legacy `test_host_callback_cannot_overrun_cell_timeout`, whose early-return assertion
requires the daemon-thread behavior being removed, with the late-synchronous-handler regression
above. It proves the deadline is supplied, a late value is never accepted, and the kernel does not
pretend it can cancel arbitrary in-process code.

- [ ] **Step 7: Run executor and engine tests**

Run: `uv run pytest tests/test_abi.py tests/test_executor.py tests/test_engine.py tests/test_specs.py tests/test_trace.py -q`

Expected: PASS.

- [ ] **Step 8: Review the ABI/executor checkpoint without staging**

```bash
git diff --check
git diff --stat -- src/rlm tests
```

---

### Task 5: Exact Controller Protocol and Lean Universal Prompt

**Files:**
- Create: `tests/test_protocol.py`
- Create: `tests/fixtures/harness-v1.json`
- Create: `tests/fixtures/controller-prompt-v20-nonrecursive.txt`
- Create: `tests/fixtures/controller-prompt-v20-recursive.txt`
- Modify: `src/rlm/engine.py`
- Modify: `src/rlm/protocol.py`
- Modify: `src/rlm/prompts.py`
- Modify: `src/rlm/specs.py`
- Modify: `src/rlm/abi.py`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_specs.py`

**Interfaces:**
- Consumes: `PromptSpec`, `ENVIRONMENT_ABI`, `ContextSpec`.
- Produces: one kernel-owned `PythonCellControllerProtocol.parse(...)` behavior.
- Produces: rolling context that retains reasoning and message output items.
- Produces: typed digests and reviewed goldens for both exact rendered prompt variants.

- [ ] **Step 1: Add failing exact-protocol tests**

```python
# tests/test_protocol.py
import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from rlm import RLM, ContextSpec, RLMConfig
from rlm.errors import HarnessContractError, ProtocolError
from rlm.prompts import default_harness_spec, render_prompt
from rlm.protocol import ControllerConversation
from tests.fakes import ScriptedBackend, controller_code, request, responses_text


def parse(text: str) -> str:
    response = responses_text(text)
    conversation = ControllerConversation(system="system", first_user="start")
    return conversation.parse(response).code


@pytest.mark.parametrize(
    "text",
    [
        "prose\n```python\nprint(1)\n```",
        "```python\nprint(1)\n```\nprose",
        "```py\nprint(1)\n```",
        "```python\nprint(1)\n```\n```python\nprint(2)\n```",
    ],
)
def test_parser_rejects_anything_except_one_exact_python_cell(text: str) -> None:
    with pytest.raises(ProtocolError):
        parse(text)


def test_parser_accepts_one_exact_python_cell() -> None:
    assert parse("```python\nvalue = 1\nprint(value)\n```") == "value = 1\nprint(value)"


def test_parser_rejects_multiple_controller_messages() -> None:
    response = responses_text("```python\nprint(1)\n```")
    response["output"].append(copy.deepcopy(response["output"][0]))
    conversation = ControllerConversation(system="system", first_user="start")

    with pytest.raises(ProtocolError, match="one assistant message"):
        conversation.parse(response)
```

```python
def test_latest_controller_reasoning_is_replayed() -> None:
    response = responses_text("```python\nprint(1)\n```")
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": []}
    response["output"].insert(0, reasoning)
    conversation = ControllerConversation(system="system", first_user="start")

    conversation.append_response(response)
    conversation.append_observation('{"status":"ok"}')

    assert conversation.items[1] == reasoning


def test_context_spec_can_mask_reasoning_without_changing_message_history() -> None:
    response = responses_text("```python\nprint(1)\n```")
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": []}
    response["output"].insert(0, reasoning)
    conversation = ControllerConversation(
        system="system",
        first_user="start",
        context=ContextSpec(preserve_reasoning=False),
    )

    conversation.append_response(response)

    assert reasoning not in conversation.items
    assert any(item.get("type") == "message" for item in conversation.items)
```

- [ ] **Step 2: Add failing prompt separation test**

```python
def test_default_prompt_contains_no_benchmark_recipe() -> None:
    rendered = render_prompt(default_harness_spec(), allow_recursion=True)
    forbidden = ("determinant", "DATA=", "Oolong", "mandatory inspection")
    assert not any(value in rendered for value in forbidden)


def test_default_harness_matches_reviewed_golden_file() -> None:
    expected = (Path(__file__).parent / "fixtures/harness-v1.json").read_text().strip()
    spec = default_harness_spec()
    assert spec.canonical_json() == expected
    assert spec.fingerprint() == hashlib.sha256(expected.encode()).hexdigest()


@pytest.mark.parametrize(
    ("allow_recursion", "fixture_name", "digest_field"),
    [
        (False, "controller-prompt-v20-nonrecursive.txt", "nonrecursive"),
        (True, "controller-prompt-v20-recursive.txt", "recursive"),
    ],
)
def test_exact_rendered_prompt_matches_golden_and_harness_identity(
    allow_recursion: bool,
    fixture_name: str,
    digest_field: str,
) -> None:
    spec = default_harness_spec()
    rendered = render_prompt(spec, allow_recursion=allow_recursion)
    expected = (Path(__file__).parent / "fixtures" / fixture_name).read_text().rstrip("\n")

    assert rendered == expected
    assert (
        getattr(spec.rendered_prompts, digest_field)
        == hashlib.sha256(rendered.encode()).hexdigest()
    )


@pytest.mark.parametrize("digest_field", ["nonrecursive", "recursive"])
def test_stale_rendered_prompt_digest_fails_before_model_call(digest_field: str) -> None:
    backend = ScriptedBackend([controller_code('FINAL_TEXT("unused")')])
    spec = default_harness_spec()
    stale = replace(
        spec,
        rendered_prompts=replace(spec.rendered_prompts, **{digest_field: "0" * 64}),
    )

    with pytest.raises(HarnessContractError, match="rendered prompt"):
        RLM(backend, config=RLMConfig(harness=stale)).run(request())

    assert backend.calls == []
```

- [ ] **Step 3: Run protocol tests and observe failures**

Run: `uv run pytest tests/test_protocol.py tests/test_engine.py -q`

Expected: FAIL because the regex parser accepts surrounding prose and fence aliases, reasoning is discarded, and the prompt contains task-specific recipes and mandatory-inspection language.

- [ ] **Step 4: Replace regex parsing with exact line parsing**

Define the pure boundary and its only production implementation:

```python
class ControllerProtocol(Protocol):
    def parse(self, response: Mapping[str, Any]) -> ControllerAction: ...


class PythonCellControllerProtocol:
    def parse(self, response: Mapping[str, Any]) -> ControllerAction:
        return ControllerAction(code=_python_cell(controller_output_text(response)))


def controller_output_text(response: Mapping[str, Any]) -> str:
    output = response.get("output")
    if not isinstance(output, list):
        raise ProtocolError("controller response output must be a list")
    if any(
        not isinstance(item, Mapping) or item.get("type") not in {"reasoning", "message"}
        for item in output
    ):
        raise ProtocolError("controller output may contain only reasoning and one message")
    messages = [item for item in output if item.get("type") == "message"]
    if len(messages) != 1:
        raise ProtocolError("controller must return exactly one assistant message")
    message = messages[0]
    if message.get("role") != "assistant":
        raise ProtocolError("controller message must have assistant role")
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        raise ProtocolError("controller message must contain exactly one output_text part")
    part = content[0]
    text = part.get("text") if isinstance(part, Mapping) else None
    if (
        not isinstance(part, Mapping)
        or part.get("type") != "output_text"
        or not isinstance(text, str)
    ):
        raise ProtocolError("controller message must contain exactly one output_text part")
    return text
```

`ControllerProtocol` is an internal type boundary for focused tests, not a user-selectable policy.
`ControllerConversation` always uses the module's single immutable
`PythonCellControllerProtocol`; neither `RLM`, the CLI, nor configuration accepts a protocol
implementation. Any production parser change must update `PromptSpec.protocol`, bump the prompt
version, and update the reviewed harness golden file, so behavior cannot change behind an unchanged
harness fingerprint.

`controller_output_text` is the single structured extractor shown above; there is no alias to the
permissive public `extract_text` helper and no `output_text` shortcut outside the message item.
Implement one deterministic cell parser:

```python
def _python_cell(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) < 3 or lines[0] != "```python" or lines[-1] != "```":
        raise ProtocolError("controller must return exactly one fenced python cell")
    if any(line.strip().startswith("```") for line in lines[1:-1]):
        raise ProtocolError("controller returned more than one fenced cell")
    code = "\n".join(lines[1:-1]).strip()
    if not code:
        raise ProtocolError("controller returned an empty Python cell")
    return code
```

Delete `_CODE_FENCE` and `_BARE_FINAL_CALL`. Require exactly one assistant `message` output item
with exactly one `output_text` content part for executable controller text. Reject refusal parts,
additional messages/content parts, and controller output item types other than `reasoning` and
`message`. Preserve all latest reasoning items plus that message in `append_response` before
appending the observation. `append_observation` accepts the canonical observation mapping and
performs the one strict JSON serialization used in model context.

- [ ] **Step 5: Replace prompt version 19 with a lean versioned protocol**

Set prompt version `20`. Add the frozen `PromptDigests(nonrecursive, recursive)` value to
`HarnessSpec`; both fields require lowercase 64-character SHA-256 hex. A private pure renderer
materializes the effective prompt from `PromptSpec` plus the
active `ENVIRONMENT_ABI`, omitting recursion-only descriptors when the depth limit disables them.
`default_harness_spec()` hashes both exact renderings into `PromptDigests`; because those digests are
in canonical harness JSON, any renderer-only change changes the harness fingerprint automatically.
Define `validate_harness_contract(spec)` in `prompts.py`; at the first line of `RLM.run()` and
`run_direct()` after request validation, call it to recompute and verify both variants plus the ABI
digest before creating trace or model state. The public `render_prompt` also verifies the selected
variant. A persisted or optimizer-supplied spec therefore cannot claim an old identity while
producing new instructions, even when one variant is inactive for that run. The prompt must state
only:

- exact request is available as `request`;
- functions rendered from the active `ENVIRONMENT_ABI`;
- one exact `python` cell per turn;
- variables persist within a branch;
- use Python for deterministic computation and model calls for semantic work;
- inspect before acting only when inspection is useful;
- errors return as structured observations;
- finals use `FINAL_TEXT` or `FINAL_RESPONSE`;
- no exposure of private protocol or intermediate work.

Delete `_task_preview`, determinant instructions, `DATA=` parsing, Oolong-style retry recipes, and
the mandatory first-turn language. `metadata_message` contains only strict request metadata and
branch depth. Remove `json.dumps(..., default=str)` because the request is already strict JSON.
Write the complete reviewed canonical `HarnessSpec` JSON to `tests/fixtures/harness-v1.json` and
both exact rendered variants to the prompt fixtures above. These goldens make changes to source
fragments, assembly, ABI documentation, ordering, or recursion filtering equally explicit.

- [ ] **Step 6: Run protocol, prompt, and engine tests**

Run: `uv run pytest tests/test_protocol.py tests/test_engine.py -q`

Expected: PASS.

- [ ] **Step 7: Review the controller-protocol checkpoint without staging**

```bash
git diff --check
git diff --stat -- src/rlm tests
```

---

### Task 6: One Typed Recovery State Machine

**Files:**
- Create: `src/rlm/observation.py`
- Create: `tests/test_observation.py`
- Modify: `src/rlm/engine.py`
- Modify: `src/rlm/errors.py`
- Modify: `src/rlm/recovery.py`
- Modify: `src/rlm/executor.py`
- Modify: `src/rlm/ledger.py`
- Modify: `tests/fakes.py`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_server.py`
- Modify: `tests/test_trace.py`

**Interfaces:**
- Consumes: typed protocol faults, `ExecutionException`, terminal validation, recovery policy, callback timeout, strict trace.
- Produces: pure, bounded `controller_observation(...) -> dict[str, Any]` encoding.
- Produces: one explicit repair path and one fatal propagation path.

- [ ] **Step 1: Add a fault-matrix regression test**

Import `request` from `tests.fakes`; reuse the `config_with` helper added to `test_engine.py` in
Task 1; and import `traced_config`, `read_events`, `read_manifest`, and `read_trace_contract` from
`tests.trace_helpers`. Do not duplicate trace decoding in individual tests.

```python
# tests/test_observation.py
from rlm.json import strict_json_dumps
from rlm.observation import controller_observation
from rlm.recovery import ControllerFault, FaultKind
from rlm.specs import ObservationSpec

from rlm.types import ExecutionException, ExecutionResult


def test_model_observation_is_strict_structured_json_within_its_hard_cap() -> None:
    execution = ExecutionResult(
        stdout="x" * 10_000,
        exception=ExecutionException(type="NameError", message="missing value"),
    )
    observation = controller_observation(
        spec=ObservationSpec(),
        fault=ControllerFault(FaultKind.EXECUTION, "missing value"),
        execution=execution,
        max_chars=256,
    )
    encoded = strict_json_dumps(observation, sort_keys=True, separators=(",", ":"))

    assert len(encoded) <= 256
    assert observation["schema_version"] == "1"
    assert observation["fault"]["kind"] == "execution"
    assert observation["truncation"]["omitted_chars"] > 0
```

```python
# tests/test_engine.py
@pytest.mark.parametrize(
    ("controller_steps", "expected_text", "fault_kind"),
    [
        (
            [responses_text("prose"), controller_code('FINAL_TEXT("fixed")')],
            "fixed",
            "controller_protocol",
        ),
        ([controller_code("1 / 0"), controller_code('FINAL_TEXT("fixed")')], "fixed", "execution"),
        (
            [
                controller_code('FINAL_RESPONSE({"output": []})'),
                controller_code('FINAL_TEXT("fixed")'),
            ],
            "fixed",
            "final_submission",
        ),
    ],
)
def test_controller_faults_are_observed_then_repaired(
    tmp_path: Path,
    controller_steps: list[dict[str, Any]],
    expected_text: str,
    fault_kind: str,
) -> None:
    rlm = RLM(
        ScriptedBackend(controller_steps),
        config=traced_config(tmp_path, max_turns=2),
    )
    result = rlm.run(request())

    assert extract_text(result.response) == expected_text
    events = read_events(result.trace_directory)
    assert any(
        event["type"] == "controller.repair" and event["payload"]["fault"]["kind"] == fault_kind
        for event in events
    )
```

```python
# tests/test_trace.py
def test_repair_trace_matches_golden_schema_contract(tmp_path: Path) -> None:
    controller = ScriptedBackend([responses_text("prose"), controller_code('FINAL_TEXT("fixed")')])
    result = RLM(
        controller,
        config=traced_config(tmp_path, max_turns=2),
    ).run(request())
    events = read_events(result.trace_directory)
    manifest = read_manifest(result.trace_directory)
    contract = read_trace_contract()

    assert [event["type"] for event in events] == contract["canonical_repair_sequence"]
    assert set(manifest) == set(contract["manifest_fields"])
    prior_ids: set[str] = set()
    for event in events:
        assert set(event) == set(contract["envelope_fields"])
        required = contract["required_payload_fields"].get(event["type"], [])
        assert set(required) == set(event["payload"])
        if event["type"] == TraceEventKind.RUN_STARTED.value:
            assert event["parent_event_id"] is None
        else:
            assert event["parent_event_id"] in prior_ids
        assert event["event_id"] not in prior_ids
        prior_ids.add(event["event_id"])

    model_events = [event for event in events if event["type"].startswith("model.")]
    assert {event["payload"]["context"]["role"] for event in model_events} == {"controller"}


def test_failed_trace_has_exact_terminal_sequence_and_error_record(tmp_path: Path) -> None:
    body = {"error": {"message": "down", "provider_code": "unavailable"}}
    backend = ScriptedBackend([UpstreamError(503, body, endpoint="/responses")])

    with pytest.raises(UpstreamError):
        RLM(backend, config=traced_config(tmp_path)).run(request())

    run_directory = next(tmp_path.iterdir())
    events = read_events(run_directory)
    manifest = read_manifest(run_directory)
    contract = read_trace_contract()
    assert [event["type"] for event in events] == contract["canonical_failure_sequence"]
    assert events[-1]["payload"]["error"]["body"] == body
    assert manifest["status"] == "failed"
    assert manifest["error"]["body"] == body
```

```python
def test_incomplete_controller_output_is_recoverable_model_output() -> None:
    incomplete = controller_code("partial =")
    incomplete["status"] = "incomplete"
    incomplete["incomplete_details"] = {"reason": "max_output_tokens"}
    controller = ScriptedBackend([incomplete, controller_code('FINAL_TEXT("fixed")')])

    result = RLM(controller, config=config_with(max_turns=2)).run(request())

    assert extract_text(result.response) == "fixed"
    assert "model_output" in controller.requests[1]["input"][-1]["content"]
```

- [ ] **Step 2: Add fatal-fault and no-repeat-heuristic tests**

```python
def test_upstream_failure_is_fatal_not_controller_feedback() -> None:
    backend = ScriptedBackend([UpstreamError(503, {"message": "down"}, endpoint="/responses")])
    with pytest.raises(UpstreamError):
        RLM(backend).run(request())


def test_nested_upstream_failure_is_fatal_not_controller_feedback() -> None:
    public = ScriptedBackend([UpstreamError(503, {"message": "down"}, endpoint="/responses")])
    controller = ScriptedBackend([controller_code('answer = ask("leaf")\nFINAL_TEXT(answer)')])

    with pytest.raises(RLMError) as caught:
        RLM(public, controller_backend=controller).run(request())

    assert caught.value.code == "upstream_error"
    assert len(controller.requests) == 1


def test_nested_upstream_status_and_body_survive_host_ipc() -> None:
    upstream_body = {"error": {"message": "rate limited", "provider_code": "quota"}}
    public = ScriptedBackend([UpstreamError(429, upstream_body, endpoint="/responses")])
    controller = ScriptedBackend([controller_code('answer = ask("leaf")\nFINAL_TEXT(answer)')])
    app = create_app(RLM(public, controller_backend=controller), close_on_shutdown=False)

    with TestClient(app) as client:
        response = client.post("/v1/responses", json=request())

    assert response.status_code == 429
    assert response.json() == upstream_body


def test_identical_cells_can_run_again_after_state_changes() -> None:
    cell = "counter = globals().get('counter', 0) + 1\nprint(counter)"
    controller = ScriptedBackend(
        [controller_code(cell), controller_code(cell), controller_code("FINAL_TEXT(str(counter))")]
    )
    result = RLM(controller, config=config_with(max_turns=3)).run(request())
    assert extract_text(result.response) == "2"


def test_explicit_abort_decision_has_its_own_fatal_error() -> None:
    harness = replace(
        default_harness_spec(),
        recovery=RecoverySpec(mode=RecoveryMode.ABORT),
    )
    controller = ScriptedBackend([responses_text("prose")])

    with pytest.raises(RecoveryAbortedError) as caught:
        RLM(controller, config=RLMConfig(harness=harness)).run(request())

    assert caught.value.code == "recovery_aborted"
```

- [ ] **Step 3: Add a deadline propagation test at the action boundary**

```python
def test_subcall_timeout_is_bounded_by_cell_and_run_deadlines() -> None:
    public = ScriptedBackend([responses_text("leaf")])
    controller = ScriptedBackend([controller_code('text = ask("leaf")\nFINAL_TEXT(text)')])
    config = RLMConfig(
        limits=RunLimits(deadline_seconds=1, execution_timeout_seconds=0.2),
    )
    RLM(public, controller_backend=controller, config=config).run(request())
    assert 0 < public.timeouts[0] <= 0.2


def test_environment_call_roles_are_assigned_from_typed_operations() -> None:
    public = ScriptedBackend([responses_text("original"), responses_text("leaf")])
    controller = ScriptedBackend(
        [
            controller_code(
                "original = model_complete()\n"
                "leaf = model_complete({'model': 'm', 'input': 'leaf'})\n"
                "FINAL_RESPONSE(original)"
            )
        ]
    )

    RLM(public, controller_backend=controller).run(request())

    assert [context.role for context in public.contexts] == [
        ModelRole.PUBLIC,
        ModelRole.SUBCALL,
    ]


def test_recursive_branch_trace_is_canonical_and_causal(tmp_path: Path) -> None:
    controller = ScriptedBackend(
        [
            controller_code("nested = rlm_complete()\nFINAL_RESPONSE(nested)"),
            controller_code('FINAL_TEXT("nested")'),
        ]
    )
    config = traced_config(tmp_path)
    config = replace(config, limits=replace(config.limits, max_depth=1))

    result = RLM(controller, config=config).run(request())

    events = read_events(result.trace_directory)
    contract = read_trace_contract()
    assert [event["type"] for event in events] == contract["canonical_recursive_sequence"]
    assert len({event["branch_id"] for event in events if event["type"] == "branch.started"}) == 2
    assert {
        event["payload"]["context"]["role"]
        for event in events
        if event["type"] in {"model.request", "model.response"}
    } == {"controller"}
```

- [ ] **Step 4: Run the new engine tests and verify failures**

Run: `uv run pytest tests/test_observation.py tests/test_engine.py tests/test_server.py tests/test_trace.py -q`

Expected: FAIL because recovery feedback is prose assembled in several branches, stderr still influences observation logic, exact duplicate cells are host-rejected, and action remaining time is not propagated into model calls.

- [ ] **Step 5: Implement the recovery state machine**

Construct the sealed `RecoveryPolicy` from `effective_config.harness.recovery` at run start. It is
not an injectable `RLM.__init__` dependency. Fatal exceptions never enter the policy.

Replace the protocol, execution, and final-submission repair branches with one helper:

```python
def _repair(
    self,
    *,
    config: RLMConfig,
    recovery_policy: RecoveryPolicy,
    fault: ControllerFault,
    execution: ExecutionResult | None,
    turn: int,
    conversation: ControllerConversation,
    trace: TraceSink,
    causal: EventRef,
    branch_id: str,
    depth: int,
) -> EventRef:
    decision = recovery_policy.decide(fault, turn=turn)
    decision_event = trace.event(
        RecoveryDecisionPayload(turn=turn, decision=decision),
        branch_id=branch_id,
        depth=depth,
        parent_event_id=causal,
    )
    if isinstance(decision, Abort):
        raise RecoveryAbortedError(fault)
    observation = controller_observation(
        spec=config.harness.observation,
        fault=fault,
        execution=execution,
        max_chars=config.limits.max_observation_chars,
    )
    conversation.append_observation(observation)
    return trace.event(
        RepairPayload(turn=turn, fault=fault, observation=observation),
        branch_id=branch_id,
        depth=depth,
        parent_event_id=decision_event,
    )
```

The helper receives the run's effective configuration snapshot explicitly; it never reads mutable
instance configuration after the run begins. Every other trace call likewise constructs the exact
typed payload; callers never provide a raw event-type string or unvalidated mapping.

Give each recovery value a strict JSON representation. Implement `controller_observation` as a pure
function in `observation.py`; it returns a strict JSON mapping with schema version, status, optional
fault, stdout, stderr, display, exception, namespace, duration, and truncation fields. A successful
execution without a final emits the same shape with `status == "ok"` and no fault. Preserve the
schema/status/fault kind/exception type first, then allocate remaining characters deterministically
across text fields and record every omission. Never slice serialized JSON. Raise an invariant error
if the encoded result exceeds the supplied cap. `ControllerConversation.append_observation` owns
the single canonical serialization into a model-visible input item; the trace retains the complete
bounded `ExecutionResult` separately.

`RecoveryAbortedError` is a typed fatal `RLMError(code="recovery_aborted")` that retains the
`ControllerFault`; do not relabel an explicit policy abort as a parser error.

The loop order becomes:

1. call and trace controller;
2. repair an `incomplete` controller response as `MODEL_OUTPUT`, otherwise parse the action or
   repair a protocol fault;
3. execute and trace action;
4. raise host/infrastructure fault immediately;
5. repair a structured cell exception, classifying `ModelOutputFault` as `MODEL_OUTPUT`,
   `FinalSubmissionFault` as `FINAL_SUBMISSION`, and other controller-cell exceptions as
   `EXECUTION`;
6. validate submitted final or repair final fault;
7. append ordinary structured observation;
8. exhaust turns with explicit `max_turns_exceeded`.

`_handle_action` receives `HostOperation` and `HostPayload`, then uses one exhaustive structural
match over the four operation/payload combinations. It never branches on raw operation strings or
reads unvalidated dictionary fields.

Delete `seen_actions`, `controller.repeated_action`, the `if not execution.stderr` final gate, and prose-specific repair assembly.

Convert `HostFailure.error` with `RemoteRLMError.from_record()`, preserving the exact public body,
message, code, status, public error type, parameter, and structured details. Preserve
`exception_type` in the trace. This is typed cross-process reconstruction, not an alternate answer
or error fallback.

- [ ] **Step 6: Propagate callback timeout to every nested call**

The executor callback supplies remaining cell time. `_handle_action` passes that cap into `_call_model` and recursive branches. `_call_model` selects the minimum positive value from the callback cap and `ledger.timeout()`, calls the backend, then checks the ledger again before accepting the response. Trace request, response, and failure events all carry `ModelCallContext.role.value`.

Assign roles from the typed operation, never by comparing request dictionaries: `run_direct()` and
`model_complete()` with its omitted request are `PUBLIC`; controller-supplied `model_complete`
requests, every batch member, and `ask` helpers are `SUBCALL`; controller turns at every recursive
depth are `CONTROLLER`.

For batch operations, every member receives the same absolute remaining deadline converted to its own positive timeout at call start. Do not retry failed members and do not return partial host-operation success when a fatal member fails.

- [ ] **Step 7: Run engine and server tests**

Run: `uv run pytest tests/test_observation.py tests/test_engine.py tests/test_server.py tests/test_trace.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 8: Review the state-machine checkpoint without staging**

```bash
git diff --check
git diff --stat -- src/rlm tests
```

---

### Task 7: Strict CLI and Durable Oolong Benchmark

**Files:**
- Create: `tests/test_benchmark.py`
- Modify: `src/rlm/config.py`
- Modify: `src/rlm/cli.py`
- Modify: `src/rlm/engine.py`
- Modify: `src/rlm/protocol.py`
- Modify: `src/rlm/server.py`
- Modify: `src/rlm/types.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_server.py`
- Modify: `benchmarks/oolong.py`

**Interfaces:**
- Consumes: nested `RLMConfig` and strict JSON controller options.
- Produces: typed `ConditioningMode.MINIMAL | ORACLE` benchmark conditions.
- Produces: typed `BenchmarkCondition.RLM | DIRECT` paired targets.
- Produces: typed `PromptProfile` provenance for both arm-specific prompts.
- Produces: typed controller/harness run identity and aggregate-usage attestation headers.
- Produces: crash-durable, resumable benchmark checkpoints and structured per-example failures.

- [ ] **Step 1: Add failing strict CLI and run-attestation tests**

```python
# tests/test_cli.py
import argparse

import pytest

from rlm import ControllerConfig
from rlm.cli import _parse_options, _positive_float


def test_controller_option_requires_valid_json_value() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        _parse_options(["reasoning_effort=high"])

    assert _parse_options(['reasoning_effort="high"', "temperature=0", "think=false"]) == {
        "reasoning_effort": "high",
        "temperature": 0,
        "think": False,
    }

    with pytest.raises(ValueError, match="valid JSON"):
        _parse_options(["temperature=NaN"])


def test_background_controller_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="background"):
        ControllerConfig(options={"background": True})


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_positive_float_rejects_non_finite_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float(value)
```

```python
# tests/test_server.py
from fastapi.testclient import TestClient
from rlm.json import strict_json_sha256

from rlm import RLM, ControllerConfig, RLMConfig
from rlm.prompts import default_harness_spec
from rlm.server import create_app
from tests.fakes import ScriptedBackend, controller_code


def test_success_headers_attest_effective_controller_identity_and_usage() -> None:
    options = {"seed": 42, "temperature": 0}
    config = RLMConfig(controller=ControllerConfig(model="qwen", options=options))
    app = create_app(
        RLM(ScriptedBackend([controller_code('FINAL_TEXT("done")')]), config=config),
        close_on_shutdown=False,
    )

    with TestClient(app) as client:
        response = client.post("/v1/responses", json={"model": "qwen", "input": "task"})

    assert response.status_code == 200
    assert response.headers["x-rlm-controller-model"] == "qwen"
    assert response.headers["x-rlm-controller-options-sha256"] == strict_json_sha256(options)
    assert response.headers["x-rlm-harness-fingerprint"] == default_harness_spec().fingerprint()
    assert int(response.headers["x-rlm-input-tokens"]) >= 0
    assert int(response.headers["x-rlm-output-tokens"]) >= 0
    assert int(response.headers["x-rlm-usage-unreported-calls"]) == 0
```

- [ ] **Step 2: Make the pure benchmark module importable without the optional dataset**

Run: `uv run python -c 'import benchmarks.oolong'`

Expected: FAIL with `ModuleNotFoundError` for `harness_rl` in the RLM development environment.
Move only the existing `harness_rl.oolong` import into `main()` without changing call semantics,
then rerun the command.

Expected: PASS. Invoking `main()` still requires the dependency and fails explicitly; this
mechanical seam lets the pure benchmark types and checkpoint functions collect independently.

- [ ] **Step 3: Add failing benchmark durability tests**

```python
# tests/test_benchmark.py
import copy
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import benchmarks.oolong as oolong
from benchmarks.oolong import (
    BenchmarkBudget,
    BenchmarkCall,
    BenchmarkComparison,
    BenchmarkCondition,
    BenchmarkExample,
    BenchmarkFailure,
    BenchmarkFailureKind,
    BenchmarkTarget,
    ConditioningMode,
    PromptProfile,
    benchmark_targets,
    build_parser,
    instructions,
    paired_requests,
    run_comparisons,
    run_pair,
    summarize,
)
from tests.fakes import responses_text


def benchmark_comparison() -> BenchmarkComparison:
    return BenchmarkComparison.from_options(
        example=BenchmarkExample.from_context(
            model="qwen",
            context="same context",
            task="same task",
            expected_count=1,
            source_id="source-1",
        ),
        mode=ConditioningMode.ORACLE,
        rlm_harness_fingerprint="a" * 64,
        seed=42,
        repetition=3,
        sampling={"temperature": 0},
        budget=BenchmarkBudget(
            timeout_seconds=600,
            max_model_calls=64,
            max_total_tokens=20_000,
        ),
    )


def successful_benchmark_call(
    text: str,
    *,
    comparison: BenchmarkComparison | None = None,
    aggregate_input_tokens: int = 3,
    aggregate_output_tokens: int = 2,
) -> BenchmarkCall:
    if comparison is None:
        comparison = benchmark_comparison()
    return BenchmarkCall(
        status=200,
        response=responses_text(text, model="qwen"),
        headers={
            "x-rlm-run-id": "run-test",
            "x-rlm-model-calls": "1",
            "x-rlm-input-tokens": str(aggregate_input_tokens),
            "x-rlm-output-tokens": str(aggregate_output_tokens),
            "x-rlm-usage-unreported-calls": "0",
            "x-rlm-controller-model": comparison.example.model,
            "x-rlm-controller-options-sha256": comparison.model_options_sha256(),
            "x-rlm-harness-fingerprint": comparison.rlm_harness_fingerprint,
        },
        failure=None,
    )


def test_empty_benchmark_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="selected no examples"):
        summarize([])


def test_failure_is_not_averaged_as_a_zero_score() -> None:
    summary = summarize([{"score": None, "failure": {"kind": "timeout"}}])
    assert summary["examples"] == 1
    assert summary["scored"] == 0
    assert summary["failures"] == 1
    assert summary["mean_score"] is None


def test_checkpoint_resumes_only_missing_arm_and_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    comparison = benchmark_comparison()
    targets = benchmark_targets("http://rlm/responses", "http://direct/responses")
    calls: list[BenchmarkCondition] = []

    def interrupted(target: BenchmarkTarget, request: dict[str, Any]) -> BenchmarkCall:
        calls.append(target.condition)
        if target.condition is BenchmarkCondition.DIRECT:
            raise RuntimeError("simulated process interruption")
        return successful_benchmark_call("answer")

    with pytest.raises(RuntimeError, match="interruption"):
        run_comparisons([comparison], targets=targets, output=output, call=interrupted)

    assert calls == [BenchmarkCondition.RLM, BenchmarkCondition.DIRECT]
    assert len(json.loads(output.read_text())["results"]) == 1

    calls.clear()

    def resumed(target: BenchmarkTarget, request: dict[str, Any]) -> BenchmarkCall:
        calls.append(target.condition)
        return successful_benchmark_call("answer")

    run_comparisons([comparison], targets=targets, output=output, call=resumed)
    assert calls == [BenchmarkCondition.DIRECT]

    with pytest.raises(ValueError, match="provenance"):
        run_comparisons(
            [replace(comparison, seed=comparison.seed + 1)],
            targets=targets,
            output=output,
            call=resumed,
        )


def test_resume_rejects_prompt_text_or_digest_drift(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    comparison = benchmark_comparison()
    targets = benchmark_targets("http://rlm/responses", "http://direct/responses")
    run_comparisons(
        [comparison],
        targets=targets,
        output=output,
        call=lambda target, request: successful_benchmark_call("answer", comparison=comparison),
    )
    checkpoint = json.loads(output.read_text())
    checkpoint["results"][0]["prompt_text"] += " tampered"
    output.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match="prompt provenance"):
        run_comparisons(
            [comparison],
            targets=targets,
            output=output,
            call=lambda target, request: successful_benchmark_call("answer", comparison=comparison),
        )


def test_transport_failure_is_structured() -> None:
    failure = BenchmarkFailure(
        kind=BenchmarkFailureKind.TIMEOUT,
        exception_type="TimeoutError",
        message="timed out",
    )
    assert failure.to_dict() == {
        "kind": "timeout",
        "exception_type": "TimeoutError",
        "message": "timed out",
    }


def test_minimal_and_oracle_conditioning_are_explicitly_distinct() -> None:
    minimal = instructions("count labels", 20, mode=ConditioningMode.MINIMAL)
    oracle = instructions("count labels", 20, mode=ConditioningMode.ORACLE)

    assert "ask_batch" not in minimal
    assert "ask_batch" in oracle
    assert minimal != oracle


def test_paired_targets_share_task_context_and_not_rlm_only_scaffolding() -> None:
    seen: list[tuple[BenchmarkCondition, dict[str, Any]]] = []

    def call(target: BenchmarkTarget, request: dict[str, Any]) -> BenchmarkCall:
        seen.append((target.condition, copy.deepcopy(request)))
        if target.condition is BenchmarkCondition.RLM:
            request["input"] = "mutated by first target"
        return successful_benchmark_call("answer")

    comparison = benchmark_comparison()
    rlm_request, direct_request = paired_requests(comparison)
    assert rlm_request is not direct_request
    assert rlm_request["seed"] == direct_request["seed"] == 42
    assert rlm_request["temperature"] == direct_request["temperature"] == 0
    records = run_pair(
        benchmark_targets("http://rlm/responses", "http://direct/responses"),
        comparison=comparison,
        call=call,
    )

    assert [condition for condition, _ in seen] == [
        BenchmarkCondition.RLM,
        BenchmarkCondition.DIRECT,
    ]
    assert seen[0][1]["model"] == seen[1][1]["model"] == "qwen"
    assert seen[0][1]["input"] == seen[1][1]["input"] == "same context"
    assert "ask_batch" in seen[0][1]["instructions"]
    assert "ask_batch" not in seen[1][1]["instructions"]
    assert {record["pair_id"] for record in records} == {comparison.pair_id()}
    assert [record["prompt_profile"] for record in records] == [
        PromptProfile.RLM_ORACLE_V1.value,
        PromptProfile.DIRECT_V1.value,
    ]


def test_rlm_controller_attestation_rejects_sampling_drift() -> None:
    comparison = benchmark_comparison()

    def call(target: BenchmarkTarget, request: dict[str, Any]) -> BenchmarkCall:
        result = successful_benchmark_call("answer", comparison=comparison)
        if target.condition is BenchmarkCondition.RLM:
            headers = {**result.headers, "x-rlm-controller-options-sha256": "0" * 64}
            return replace(result, headers=headers)
        return result

    records = run_pair(
        benchmark_targets("http://rlm/responses", "http://direct/responses"),
        comparison=comparison,
        call=call,
    )
    rlm_record = next(record for record in records if record["condition"] == "rlm")

    assert rlm_record["answer"] is None
    assert rlm_record["score"] is None
    assert rlm_record["failure"]["kind"] == BenchmarkFailureKind.PROVENANCE.value


def test_rlm_token_budget_uses_aggregate_headers_not_leaf_usage() -> None:
    original = benchmark_comparison()
    comparison = replace(
        original,
        budget=replace(original.budget, max_total_tokens=4),
    )

    def call(target: BenchmarkTarget, request: dict[str, Any]) -> BenchmarkCall:
        result = successful_benchmark_call(
            "answer",
            comparison=comparison,
            aggregate_input_tokens=3,
            aggregate_output_tokens=2,
        )
        if target.condition is BenchmarkCondition.RLM:
            assert result.response is not None
            response = copy.deepcopy(result.response)
            response["usage"] = {
                "input_tokens": 1,
                "output_tokens": 0,
                "total_tokens": 1,
            }
            return replace(result, response=response)
        return result

    records = run_pair(
        benchmark_targets("http://rlm/responses", "http://direct/responses"),
        comparison=comparison,
        call=call,
    )
    rlm_record = next(record for record in records if record["condition"] == "rlm")

    assert rlm_record["failure"]["kind"] == BenchmarkFailureKind.BUDGET.value
    assert rlm_record["observed_total_tokens"] == 5


@pytest.mark.parametrize(
    "field_name",
    ["model", "input", "instructions", "stream", "background", "seed"],
)
def test_sampling_cannot_override_comparison_or_protocol_fields(field_name: str) -> None:
    comparison = benchmark_comparison()

    with pytest.raises(ValueError, match="sampling"):
        BenchmarkComparison.from_options(
            example=comparison.example,
            mode=comparison.mode,
            rlm_harness_fingerprint=comparison.rlm_harness_fingerprint,
            seed=comparison.seed,
            repetition=comparison.repetition,
            sampling={field_name: True},
            budget=comparison.budget,
        )


@pytest.mark.parametrize(
    "changed",
    [
        lambda value: replace(value, mode=ConditioningMode.MINIMAL),
        lambda value: replace(
            value,
            example=replace(value.example, model="different-model"),
        ),
        lambda value: replace(
            value,
            example=replace(value.example, task="different task"),
        ),
        lambda value: replace(
            value,
            example=replace(value.example, expected_count=value.example.expected_count + 1),
        ),
        lambda value: replace(
            value,
            example=replace(value.example, source_id="different-source"),
        ),
        lambda value: replace(
            value,
            example=BenchmarkExample.from_context(
                model=value.example.model,
                context="different context",
                task=value.example.task,
                expected_count=value.example.expected_count,
                source_id=value.example.source_id,
            ),
        ),
        lambda value: replace(value, seed=value.seed + 1),
        lambda value: replace(value, repetition=value.repetition + 1),
        lambda value: replace(value, rlm_harness_fingerprint="b" * 64),
        lambda value: replace(
            value,
            budget=replace(value.budget, timeout_seconds=value.budget.timeout_seconds + 1),
        ),
        lambda value: replace(
            value,
            budget=replace(value.budget, max_model_calls=value.budget.max_model_calls + 1),
        ),
        lambda value: replace(
            value,
            budget=replace(value.budget, max_total_tokens=20_001),
        ),
        lambda value: BenchmarkComparison.from_options(
            example=value.example,
            mode=value.mode,
            rlm_harness_fingerprint=value.rlm_harness_fingerprint,
            seed=value.seed,
            repetition=value.repetition,
            sampling={"temperature": 0.5},
            budget=value.budget,
        ),
    ],
)
def test_pair_id_covers_every_comparison_control(
    changed: Callable[[BenchmarkComparison], BenchmarkComparison],
) -> None:
    comparison = benchmark_comparison()
    assert changed(comparison).pair_id() != comparison.pair_id()


def test_pair_id_binds_exact_direct_prompt_text(monkeypatch: pytest.MonkeyPatch) -> None:
    before = benchmark_comparison()
    original = oolong.direct_instructions

    monkeypatch.setattr(
        oolong,
        "direct_instructions",
        lambda task, expected_count: original(task, expected_count) + "\nChanged wording.",
    )
    after = benchmark_comparison()

    assert before.prompts.direct.sha256 != after.prompts.direct.sha256
    assert before.pair_id() != after.pair_id()


def test_pair_id_binds_exact_rlm_prompt_text(monkeypatch: pytest.MonkeyPatch) -> None:
    before = benchmark_comparison()
    original = oolong.instructions

    monkeypatch.setattr(
        oolong,
        "instructions",
        lambda task, expected_count, *, mode: (
            original(task, expected_count, mode=mode) + "\nChanged wording."
        ),
    )
    after = benchmark_comparison()

    assert before.prompts.rlm.sha256 != after.prompts.rlm.sha256
    assert before.pair_id() != after.pair_id()


def test_direct_url_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--url",
                "http://rlm/responses",
                "--model",
                "qwen",
                "--rlm-harness-fingerprint",
                "a" * 64,
            ]
        )


def test_expected_rlm_harness_fingerprint_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--url",
                "http://rlm/responses",
                "--direct-url",
                "http://direct/responses",
                "--model",
                "qwen",
            ]
        )
```

- [ ] **Step 4: Run focused tests and verify failures**

Run: `uv run pytest tests/test_cli.py tests/test_server.py tests/test_benchmark.py -q`

Expected first: benchmark collection fails because the new typed declarations do not exist. Add
only those declarations and parsing signatures, then run each CLI, pairing, pair-ID, checkpoint-
resume, failure-accounting, conditioning, and server-attestation test separately. Observe the old
raw-string fallback, unsupported background option, absent identity headers, missing
provenance/deep-copy isolation, and non-resumable checkpoint behavior as independent failures
before implementing them.

- [ ] **Step 5: Make controller options strict and protocol-compatible**

`_parse_options` parses values with `strict_json_loads` and raises `ValueError` on
`StrictJSONError`; strings must be JSON quoted. Update CLI help accordingly. `ControllerConfig`
validates strict JSON and rejects keys that change the controller protocol or asynchronous
behavior:

```python
@dataclass(frozen=True, slots=True)
class ControllerRequestCore:
    model: str
    instructions: str
    input: list[dict[str, Any]]


PROTOCOL_OWNED_FIELDS = frozenset(field.name for field in fields(ControllerRequestCore))
UNSUPPORTED_CONTROLLER_FIELDS = frozenset(
    {"background", "stream", "modalities", "response_format", "text", "tools", "tool_choice"}
)
```

`ControllerConversation.request` builds its owned portion through `ControllerRequestCore`, so the
protected-key set is structurally derived rather than maintained in parallel. Export it from
`protocol.py` and import it in `config.py`. The unsupported set is one explicit controller-output
contract, not a fallback alias table.

Expose one pure `validate_controller_options(options)` in `config.py`; `ControllerConfig` calls it
rather than duplicating the sets. `BenchmarkComparison.from_options` rejects `seed` in the raw
sampling mapping, materializes `{**sampling, "seed": seed}`, and passes that exact mapping through
the same validator. Sampling therefore cannot contain `model`, `input`, `instructions`, async or
tool fields, or another protocol-owned key that would be ignored or override immutable comparison
data on one arm.

Add frozen `ControllerRunIdentity(model, options_sha256)` and include it plus
`harness_fingerprint` in `RunResult`. The engine derives both from its one immutable run snapshot;
`options_sha256` uses `strict_json_sha256(snapshot.controller.options)`, and the effective model is
the configured controller model or that run's public model. The FastAPI adapter exposes those
values and aggregate ledger counters through the exact `X-RLM-*` headers asserted above. Header
construction consumes only `RunResult`; it never rereads mutable `RLM.config` after completion.

- [ ] **Step 6: Implement structured benchmark calls and atomic checkpoints**

Define `ConditioningMode(str, Enum)` with `MINIMAL = "minimal"` and `ORACLE = "oracle"`; default
the CLI to `minimal`. Also define `BenchmarkCondition(str, Enum)` with `RLM = "rlm"` and
`DIRECT = "direct"`, a `PromptProfile` enum with `RLM_MINIMAL_V1`, `RLM_ORACLE_V1`, and
`DIRECT_V1`, plus a frozen `BenchmarkTarget(condition, url)`. Keep the pure signatures
`instructions(task, expected_count, *, mode)` and `direct_instructions(task, expected_count)` so
the derived prompt-pair constructor has one source for each arm. Minimal conditioning states the
aggregate question, expected record count, label taxonomy, and exact answer format, but supplies no
extraction code, batching strategy, or retry recipe. Oracle conditioning retains the explicit
scaffold below as a separately labeled stress condition.

Update the Oolong task-conditioning example to the structured batch ABI. It may recommend this
controller-authored retry, but the host never performs it automatically:

```python
batch = ask_batch(prompts, options={"temperature": 0})
labels = list(batch.texts)
if batch.failed_indexes:
    retry = ask_batch(
        [prompts[index] for index in batch.failed_indexes],
        options={"temperature": 0},
    )
    replacements = retry.require_texts()
    for index, text in zip(batch.failed_indexes, replacements):
        labels[index] = text
assert all(isinstance(label, str) and label.strip() for label in labels)
```

Add:

```python
@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    model: str
    context: str
    task: str
    expected_count: int
    source_id: str
    context_sha256: str

    @classmethod
    def from_context(
        cls,
        *,
        model: str,
        context: str,
        task: str,
        expected_count: int,
        source_id: str,
    ) -> BenchmarkExample: ...


@dataclass(frozen=True, slots=True)
class BenchmarkBudget:
    timeout_seconds: float
    max_model_calls: int
    max_total_tokens: int | None


@dataclass(frozen=True, slots=True)
class BenchmarkPrompt:
    profile: PromptProfile
    text: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BenchmarkPrompts:
    rlm: BenchmarkPrompt
    direct: BenchmarkPrompt


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    example: BenchmarkExample
    mode: ConditioningMode
    rlm_harness_fingerprint: str
    seed: int
    repetition: int
    sampling_json: str
    budget: BenchmarkBudget
    prompts: BenchmarkPrompts = field(init=False)

    @classmethod
    def from_options(
        cls,
        *,
        example: BenchmarkExample,
        mode: ConditioningMode,
        rlm_harness_fingerprint: str,
        seed: int,
        repetition: int,
        sampling: dict[str, Any],
        budget: BenchmarkBudget,
    ) -> BenchmarkComparison: ...

    def canonical_json(self) -> str: ...

    def pair_id(self) -> str: ...

    def model_options(self) -> dict[str, Any]: ...

    def model_options_sha256(self) -> str: ...


class BenchmarkFailureKind(str, Enum):
    HTTP = "http"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    DECODE = "decode"
    PROTOCOL = "protocol"
    PROVENANCE = "provenance"
    BUDGET = "budget"


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    kind: BenchmarkFailureKind
    exception_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCall:
    status: int | None
    response: dict[str, Any] | None
    headers: dict[str, str]
    failure: BenchmarkFailure | None
```

`BenchmarkComparison.from_options` validates sampling with the shared strict-JSON contract,
requires a 64-character lowercase hexadecimal expected RLM harness fingerprint, rejects a
duplicate `seed` key in sampling, validates the combined mapping with the shared
`validate_controller_options`, and stores only canonical JSON, so caller-owned nested mappings
cannot mutate a comparison. `model_options()` materializes exactly `{**sampling, "seed": seed}`;
its shared canonical SHA-256 is the controller-options attestation expected from the RLM server.
`BenchmarkPrompt.__post_init__` derives its `init=False` digest from exact text.
`BenchmarkComparison.__post_init__` derives its `init=False` `BenchmarkPrompts` from `instructions(...)` and
`direct_instructions(...)`; callers cannot supply a profile, text, or digest, and
`dataclasses.replace` recomputes them whenever the example or conditioning mode changes.
`BenchmarkExample.from_context` derives the context digest, and `__post_init__` rejects a digest
that does not match the exact context; provenance never relies on a caller-maintained parallel hash.
`pair_id()` is SHA-256 of the complete canonical comparison: example provenance and content digest,
both typed prompt profiles, both exact prompt texts and digests, RLM conditioning mode, expected
harness fingerprint, seed, repetition, sampling, and declared budget. `paired_requests` and
`run_pair` accept the comparison value, derive the ID internally, and deep-copy each arm's request;
they never accept a caller-supplied pair ID.

Expose the comparison controls as required/explicit CLI values: `--direct-url`,
`--rlm-harness-fingerprint`, `--timeout`, `--max-model-calls`, `--max-total-tokens`, and
`--repetitions`, with strict finite/range validation.
Sampling options use the same quoted strict-JSON syntax as controller options. Defaults, where
retained for convenience, are still materialized into `BenchmarkRunSpec` and the run fingerprint.

Decode response bytes with `strict_json_loads`. Catch only expected transport and decoding failures:
`HTTPError`, `URLError`, `TimeoutError`, `OSError`, `StrictJSONError`, and `UnicodeError`. Classify
them with `BenchmarkFailureKind` and record the concrete exception type and message. A decoded
success must be a strict terminal Responses object according to `validate_terminal_response`;
malformed envelopes and missing benchmark answer text become `PROTOCOL` failures. Do not catch
arbitrary programming exceptions.

Enforce declared provenance and budget at the observable boundary. The RLM arm requires
`X-RLM-Run-ID`, `X-RLM-Model-Calls`, `X-RLM-Input-Tokens`, `X-RLM-Output-Tokens`,
`X-RLM-Usage-Unreported-Calls`, `X-RLM-Controller-Model`,
`X-RLM-Controller-Options-SHA256`, and `X-RLM-Harness-Fingerprint`. Reject missing or malformed
headers, a controller model different from the comparison model, an options digest different from
`model_options_sha256()`, a harness fingerprint different from the expected value, unreported
usage, or a call count above `max_model_calls`. The server derives these headers from the immutable
run snapshot and typed `RunResult` identity, never from request echo fields. The direct arm has
exactly one model call. Audit RLM tokens only as the aggregate input-plus-output headers across all
controller and subcalls; a preserved leaf `FINAL_RESPONSE.usage` is not run usage. Audit direct
tokens from its one standard response usage object. A non-null `max_total_tokens` rejects either
audited total above budget. Both HTTP calls use the same declared timeout. Record observed calls,
tokens, elapsed time, and audit status; never label an unauditable arm as a valid paired score.

Keep the optional `harness_rl.oolong` import inside `main()` as established in Step 2. If the
sibling dependency is unavailable for an actual run, exit with one explicit prerequisite error
naming `harness-rl`; do not substitute another dataset or scorer.

Represent the checkpoint with validated `BenchmarkRunSpec`, `BenchmarkRecord`, and
`BenchmarkCheckpoint` dataclasses. Its run fingerprint covers dataset/selection digest, target
URLs, model, both exact prompt texts/profiles/digests, expected RLM harness fingerprint, effective
controller-options digest, seed policy, repetitions, sampling, declared budget, source revision,
and runner schema version. `atomic_checkpoint(path, checkpoint)` writes the complete strict payload
to a temporary file in `path.parent`, flushes and `fsync`s it, closes it, uses `os.replace`, then
opens and `fsync`s the parent directory so the replacement itself is crash-durable. Clean up only
the exact temporary path on a write failure and re-raise. Call it after every completed arm when
`--output` is supplied.

On startup, strictly decode an existing checkpoint, reject a mismatched run fingerprint, malformed
record, or duplicate `(pair_id, condition)` key, and schedule only missing arms. A partial pair
therefore resumes its missing target without rerunning or overwriting the completed target. Never
resume by source ID alone or merge checkpoints with different provenance.
`summarize([])` raises `ValueError("benchmark selected no examples")`. Every failed call has
`answer: null`, `score: null`, and a structured `failure`; HTTP and protocol failures retain their
status code, while failures that received no HTTP response use `status: null`. A failure is never
scored as an empty answer.

`summarize` reports total, scored, and failed counts. `mean_score` is over scored examples only and
is `null` when none were scored; the failure rate remains separate so missing work cannot improve
or masquerade as benchmark accuracy.

Every record stores `condition`, `prompt_profile`, exact conditioning text and its SHA-256, dataset
path, dataset file SHA-256, seed, and selection parameters. On resume, record validation requires
the profile, text, and digest to equal the matching immutable `comparison.prompts` arm before the
record can satisfy a `(pair_id, condition)` key. `rlm_conditioning_mode` is the selected
minimal/oracle enum on the RLM record and `null` on the direct record; the direct arm is always
`PromptProfile.DIRECT_V1`, so it cannot be mislabeled as oracle merely because of its comparison
partner. Add required `--direct-url`. Build both arms
from one immutable `BenchmarkExample`, so model, selected context, aggregate task, expected output,
seed, sampling, and provenance cannot drift. Both public requests receive the exact mapping from
`comparison.model_options()`. A valid RLM record additionally requires its controller-identity
headers to attest those same options and model. The RLM arm receives the selected minimal or oracle
prompt. The direct arm receives one versioned direct prompt containing the task, taxonomy, record
count, and answer format but no RLM ABI names, extraction code, or retry recipe; the same direct
condition is used when comparing minimal and oracle RLM arms. Deep-copy each derived request at the call
boundary, derive one deterministic `pair_id` from immutable example/comparison provenance, and
store `condition`, `pair_id`, exact prompt digest, and nullable `run_id` on both records. Checkpoint
after each target completes. Never substitute the direct answer for an RLM failure (or vice versa).

Summaries report each condition independently plus paired score deltas only for pairs with two
valid verifier scores; unpaired/failing counts remain explicit. The richer checkpoint/repetition
and verifier sidecar remains owned by `rlm-bootstrap`, keyed by experiment `episode_id`, with
`run_id` nullable for direct records and `pair_id` linking comparisons. Its concrete
acceptance contract is documented in `../rlm-bootstrap/rlm_harness_integration_contract.md`: split
by context group, run repeated paired rollouts, and compare the same task/context/model/budget
condition with attested seed and sampling, without sharing answers between arms.

Call the local output paired stress-test evidence, not a reproducible training/evaluation dataset.
It records declared controls and runner/source identity, but checkpoint digests, tokenizer/backend
revisions, controlled repetitions, split enforcement, and verifier provenance become publication-
quality evidence only after the `rlm-bootstrap` sidecar acceptance tests pass.

- [ ] **Step 7: Run CLI, server, and benchmark tests**

Run: `uv run pytest tests/test_cli.py tests/test_server.py tests/test_benchmark.py -q`

Expected: PASS.

- [ ] **Step 8: Review the operational-tooling checkpoint without staging**

```bash
git diff --check
git diff --stat -- src/rlm benchmarks tests
```

---

### Task 8: Documentation, Coverage, and Contract Audit

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `RESEARCH.md`
- Modify: `AGENTS.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_executor.py`
- Modify: `tests/test_backend.py`
- Modify: `tests/test_protocol.py`
- Modify: `tests/test_trace.py`
- External handoff artifact (sibling workspace, not staged here):
  `../rlm-bootstrap/rlm_harness_integration_contract.md`

**Interfaces:**
- Consumes: completed runtime contract.
- Produces: accurate user/research documentation and a measured coverage report.

- [ ] **Step 1: Add pytest-cov to the development environment**

Add `pytest-cov>=5.0` to `[project.optional-dependencies].dev`, then run:

While touching project metadata, replace the generic package author with the repository owner's
canonical identity: `Alexander Richard Towell <lex@metafunctor.com>`.

Run: `uv lock`

Expected: `uv.lock` records pytest-cov and its coverage dependency without changing runtime dependencies.

Run: `uv sync --extra dev`

Expected: the locked development tools, including pytest-cov, are installed before coverage and
static verification.

- [ ] **Step 2: Update public and architecture documentation**

Document the exact final behavior:

- nested configuration example using `HarnessSpec`, `RunLimits`, `ControllerConfig`, `ExecutionConfig`, and `TraceConfig`;
- Responses-only surface and explicit direct baseline;
- typed repairable/fatal boundary;
- strict one-cell protocol with no mandatory inspection;
- reasoning-item replay;
- structured `AskBatchResult` use, including `failed_indexes` and `require_texts()`;
- deadline-aware backend contract;
- strict trace schema, exact rendered-prompt digests, and harness fingerprint;
- task conditioning and verifier ownership outside the kernel;
- paired direct-model evaluation under identical task/context/model conditions, with failures kept
  separate from verifier scores;
- fixed-harness self-SFT, later RLVR, and symbolic harness-search boundary;
- IPython process isolation is not a security sandbox.

Delete stale claims about stderr indicating failure, mandatory first-turn inspection, global pending text errors, permissive controller code, or prompt-embedded benchmark recipes.

Cross-check `../rlm-bootstrap/rlm_harness_integration_contract.md` against the final public names.
It must retain the concrete paired-baseline, context-group split, repetition, failure-rate, and
provenance requirements. If a public name changed during implementation, update that companion
document in the sibling workspace in the same step; it is not part of this repository's commit.
The sibling directory is not currently a Git repository, so compute and report the document's
SHA-256 in the final handoff. `rlm-bootstrap` must add that exact reviewed artifact to version
control when its repository is initialized; do not imply it was committed here.

- [ ] **Step 3: Audit every fallback and broad exception site**

Run: `rg -n -i "fallback|except[[:space:]]*:|except BaseException|except Exception|except .*Error|contextlib\.suppress|default=str|repr\(value\)|pending_text_error|seen_actions|[[:space:]]or[[:space:]]+(backend|config|factory|policy|protocol)" src tests benchmarks README.md DESIGN.md RESEARCH.md`

For every match, classify it in review notes as one of:

1. explicit rethrow/cleanup that preserves the original fault;
2. expected typed boundary handling;
3. a remaining defect requiring a regression test and fix.

There must be no non-JSON-success conversion, invalid-JSON CLI conversion, trace `repr` conversion,
answer fallback, or singleton pending-error state. Cleanup-only exception suppression must name the
narrow OS/pipe exception and have no effect on run success. Inspect every optional-dependency
constructor directly for truthiness defaulting even when the grep is empty; targeted falsey
backend/factory tests are the executable guard. Record every broad catch and suppression
classification in the final handoff rather than treating a grep exit code as proof of semantics.

- [ ] **Step 4: Run coverage and add tests for material uncovered failure paths**

Run: `uv run pytest --cov=rlm --cov-report=term-missing`

Expected: PASS with a line-by-line missing coverage report.

Add focused tests only for uncovered protocol, deadline, error, cleanup, and serialization branches that can alter public success/failure semantics. Do not add tests merely to inflate a percentage. Re-run the same command and retain the final report for handoff.

- [ ] **Step 5: Verify there are no Codex-specific runtime references**

Run: `rg -n -i "codex" src tests README.md DESIGN.md RESEARCH.md pyproject.toml`

Expected: no maintained runtime, tests, or documentation references. Historical design/plan documents are excluded from this assertion.

- [ ] **Step 6: Run complete static and build verification**

Run: `uv run pytest -q`

Expected: all tests pass.

Run: `uv run ruff check src tests benchmarks`

Expected: exit 0 with no diagnostics.

Run: `uv run ruff format --check src tests benchmarks`

Expected: exit 0 with no files requiring formatting.

Run: `uv build`

Expected: source and wheel distributions are created successfully.

- [ ] **Step 7: Review the complete implementation without staging**

```bash
git diff --check
git status --short
git diff --stat
```

Present the complete reviewed diff and verification evidence to the user before any commit. If the
user asks for commits, partition them with interactive hunk selection and verify each staged diff;
never use full-path staging as a substitute for hunk review in this pre-existing dirty tree.

---

### Task 9: Live Qwen and Oolong Smoke Verification

**Files:**
- Create only runtime artifacts under ignored `runs/`; do not modify source unless a reproducible defect is first added as a failing test in the owning earlier task.

**Interfaces:**
- Consumes: built `rlm` server, upstream `http://192.168.0.204:11434/v1`, model `qwen3.5:latest`, and optional sibling Oolong dataset.
- Produces: trace-backed evidence that the live model can inspect, execute Python, repair a defective cell, and complete benchmark requests.

- [ ] **Step 1: Confirm the configured upstream and model explicitly**

Run: `curl --fail-with-body --max-time 10 http://192.168.0.204:11434/v1/models`

Expected: HTTP success with a JSON model catalog containing `qwen3.5:latest`. If unavailable, record the exact transport/status failure and continue only with local automated verification; do not substitute another model.

- [ ] **Step 2: Start the hardened server on an unused test port**

Run in a managed process session:

```bash
uv run rlm serve \
  --host 127.0.0.1 \
  --port 8011 \
  --upstream-base-url http://192.168.0.204:11434/v1 \
  --upstream-api-key ollama \
  --controller-model qwen3.5:latest \
  --controller-option temperature=0 \
  --controller-option seed=42 \
  --max-model-calls 64 \
  --max-total-tokens 20000 \
  --trace-dir runs/live-hardening \
  --trace-markdown
```

Expected: server starts without configuration warnings. Use a different explicitly checked free port if 8011 is occupied.

- [ ] **Step 3: Run three trace-backed live cases**

Run this verifier in a managed process session:

```bash
uv run python - <<'PY'
import json

import httpx

from rlm.protocol import extract_text
from rlm.json import strict_json_loads
from rlm.response import validate_terminal_response

cases = [
    (
        "xor-squares",
        "Compute the sum of i squared for integers i from 1 through 10000 where i is divisible "
        "by 7 or 11, but not both. Return only the decimal integer.",
        "69371782431",
    ),
    (
        "structured-distractors",
        "Parse only the signed value in the middle field of each DATA line; digits in notes are "
        "distractors. Sum by label and return compact JSON with alphabetically sorted keys.\n"
        "DATA:\nalpha|17|ref900\nbeta|-4|year2025\nalpha|29|v2\ngamma|1000|id7\n"
        "beta|8|n99\nalpha|-3|code42\ngamma|6|x123\nbeta|101|room8",
        '{"alpha":43,"beta":105,"gamma":1006}',
    ),
    (
        "prime-state",
        "Generate the first 300 primes. Join them with commas, compute the lowercase SHA-256 "
        "digest, and also sum every 17th prime using one-based positions 17, 34, and so on. "
        "Return exactly: <digest> <sum>.",
        "d00fcee57f01afd9e941dd9795bc8923a9655a128bc623308d259df8fdae37ec 15561",
    ),
]

for name, prompt, expected in cases:
    response = httpx.post(
        "http://127.0.0.1:8011/v1/responses",
        json={"model": "qwen3.5:latest", "input": prompt},
        timeout=650,
    )
    response.raise_for_status()
    payload = strict_json_loads(response.content)
    if not isinstance(payload, dict):
        raise AssertionError(f"{name}: response body is not an object")
    error = validate_terminal_response(payload)
    if error is not None:
        raise AssertionError(f"{name}: {error}")
    actual = extract_text(payload).strip()
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")
    print(json.dumps({
        "case": name,
        "run_id": response.headers["x-rlm-run-id"],
        "turns": response.headers["x-rlm-turns"],
        "model_calls": response.headers["x-rlm-model-calls"],
        "answer": actual,
    }), flush=True)
PY
```

The third case is stateful and exposes any natural controller mistake as a structured repair; do
not instruct the model to fail artificially.

For each response, require HTTP 200, validate the body with `validate_terminal_response`, and record
`X-RLM-Run-ID`, turns, model calls, final answer, and the causal event sequence. The deterministic
fault-matrix tests, rather than nondeterministic live behavior, are the acceptance proof for
self-correction. Do not accept a direct fallback event or a response with missing terminal output.

- [ ] **Step 4: Run a checkpointed Oolong sample when the sibling dataset is present**

Run:

```bash
RLM_HARNESS_FINGERPRINT="$(uv run python -c \
  'from rlm.prompts import default_harness_spec; print(default_harness_spec().fingerprint())')"
PYTHONPATH="$PWD/src" ../harness-rl/.venv/bin/python benchmarks/oolong.py \
  --url http://127.0.0.1:8011/v1/responses \
  --direct-url http://192.168.0.204:11434/v1/responses \
  --rlm-harness-fingerprint "$RLM_HARNESS_FINGERPRINT" \
  --model qwen3.5:latest \
  --context-length 1024 \
  --num-examples 3 \
  --seed 42 \
  --sampling temperature=0 \
  --repetitions 1 \
  --timeout 600 \
  --max-model-calls 64 \
  --max-total-tokens 20000 \
  --conditioning minimal \
  --output runs/oolong-hardening-1k.json
```

Expected: three RLM records and three direct records, plus per-condition and paired aggregates,
with the output file atomically updated after each target. Failures remain structured records and
never borrow the other arm's answer. If the pinned dataset or environment is absent, report that
exact missing prerequisite and do not replace it with a different dataset.

- [ ] **Step 5: Stop the managed server and inspect cleanup**

Terminate only the server process started in Step 2. Confirm the port is closed and no executor descendant from these runs remains. Preserve traces and benchmark outputs under ignored `runs/`.

- [ ] **Step 6: Re-run final automated verification after any live-discovered fix**

If live verification exposed a reproducible code defect, return to the owning task, add a failing regression test, implement the minimal fix, and run:

```bash
uv run pytest -q
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
uv build
```

Expected: all commands exit 0 before completion is claimed.

No trace or benchmark artifact is staged. If live verification requires a source fix, keep its
regression test and implementation visible in the final reviewed diff; do not commit it unless the
user later authorizes the commit strategy for the dirty worktree.
