"""The request-preserving Recursive Language Model loop."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from rlm.abi import (
    ENVIRONMENT_ABI,
    HostBatchPayload,
    HostOperation,
    HostPayload,
    HostRequestPayload,
)
from rlm.backend import CloseableBackend, ModelBackend
from rlm.config import RLMConfig
from rlm.errors import (
    BackendCloseError,
    BackendCloseFailure,
    BackendProtocolError,
    BackendResponseError,
    ErrorRecord,
    ExecutionTimeoutError,
    HarnessContractError,
    InvalidRequestError,
    LimitExceededError,
    ProtocolError,
    RecoveryAbortedError,
    RemoteRLMError,
    RLMError,
    TraceCausalParent,
    error_record_for_exception,
)
from rlm.executor import ExecutionEnvironmentFactory, IPythonExecutor
from rlm.json import json_compatibility_error, strict_json_sha256
from rlm.ledger import Ledger
from rlm.observation import controller_observation
from rlm.prompts import bootstrap_message, render_prompt, validate_harness_contract
from rlm.protocol import ControllerConversation
from rlm.recovery import Abort, ControllerFault, FaultKind, RecoveryPolicy
from rlm.response import (
    requires_complete_response,
    text_response,
    validate_response_envelope,
    validate_terminal_response,
)
from rlm.trace import (
    TRACE_SCHEMA_VERSION,
    BranchCompletedPayload,
    BranchStartedPayload,
    ControllerActionPayload,
    ControllerExecutionPayload,
    ControllerObservationPayload,
    EventRef,
    HarnessFingerprint,
    ModelFailedPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    RecoveryDecisionPayload,
    RepairPayload,
    RunCompletedPayload,
    RunFailedPayload,
    RunStartedPayload,
    RunTraceStatus,
    TraceManifest,
    TraceSink,
    make_trace_sink,
)
from rlm.types import (
    ControllerRunIdentity,
    ExecutionFaultKind,
    ExecutionResult,
    ModelCallContext,
    ModelRole,
    ResponseSubmission,
    RunResult,
    TextSubmission,
)


@dataclass(frozen=True, slots=True)
class _BranchResult:
    response: dict[str, Any]
    stop_reason: str
    turns: int
    event_id: EventRef


@dataclass(frozen=True, slots=True)
class _ParallelOutcome:
    """One settled parallel child result, retained in input order."""

    value: Any
    error: BaseException | None


class RLM:
    """Wrap a Responses backend in one bounded controller-and-Python loop."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        config: RLMConfig | None = None,
        controller_backend: ModelBackend | None = None,
        execution_factory: ExecutionEnvironmentFactory | None = None,
    ) -> None:
        self.backend = backend
        self.controller_backend = backend if controller_backend is None else controller_backend
        self.config = RLMConfig() if config is None else config
        self.execution_factory = IPythonExecutor if execution_factory is None else execution_factory

    def run(self, request: Mapping[str, Any]) -> RunResult:
        """Run one Responses request and return its response plus RLM metadata."""

        public_request = _validate_request(request)
        config = self.config.snapshot()
        _validate_harness_abi(config)
        run_id = uuid.uuid4().hex
        started = time.monotonic()
        trace = make_trace_sink(config.tracing, run_id=run_id)
        ledger = Ledger(config)
        stop_reason = "error"
        turns = 0
        error: BaseException | None = None

        run_started_event_id = trace.event(
            RunStartedPayload(
                request=public_request,
                effective_config=config,
                harness=config.harness,
                harness_fingerprint=HarnessFingerprint(config.harness),
                trace_schema_version=TRACE_SCHEMA_VERSION,
                environment_abi=ENVIRONMENT_ABI.to_dict(),
                environment_abi_digest=ENVIRONMENT_ABI.digest(),
            ),
            branch_id="branch-root",
            depth=0,
        )
        try:

            def note_root_turn(turn: int) -> None:
                nonlocal turns
                turns = turn

            result = self._run_branch(
                request=public_request,
                config=config,
                trace=trace,
                ledger=ledger,
                run_id=run_id,
                depth=0,
                branch_id="branch-root",
                parent_event_id=run_started_event_id,
                on_root_turn=note_root_turn,
            )
            stop_reason = result.stop_reason
            turns = result.turns
            trace.event(
                RunCompletedPayload(stop_reason=stop_reason, response=result.response),
                branch_id="branch-root",
                depth=0,
                parent_event_id=result.event_id,
            )
            return RunResult(
                response=result.response,
                run_id=run_id,
                stop_reason=stop_reason,
                usage=ledger.snapshot(),
                turns=turns,
                duration_seconds=time.monotonic() - started,
                trace_directory=trace.directory,
                controller_identity=ControllerRunIdentity(
                    model=config.controller.model or str(public_request["model"]),
                    options_sha256=strict_json_sha256(config.controller.options),
                ),
                harness_fingerprint=config.harness.fingerprint(),
            )
        except BaseException as exc:
            error = exc
            trace.event(
                RunFailedPayload(error=_error_record(exc)),
                branch_id="branch-root",
                depth=0,
                parent_event_id=_failure_parent_event_id(exc) or run_started_event_id,
            )
            raise
        finally:
            trace.close(
                manifest_factory=lambda: _trace_manifest(
                    run_id=run_id,
                    error=error,
                    stop_reason=stop_reason,
                    turns=turns,
                    started=started,
                    ledger=ledger,
                    config=config,
                ),
                cause=error,
            )

    def complete(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run one request and return only its Responses object."""

        return self.run(request).response

    responses = complete

    def direct(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run the one-call direct baseline and return its Responses object."""

        return self.run_direct(request).response

    def run_direct(self, request: Mapping[str, Any]) -> RunResult:
        """Run the deadline-aware public one-call baseline with normal tracing."""

        public_request = _validate_request(request)
        config = self.config.snapshot()
        _validate_harness_abi(config)
        run_id = uuid.uuid4().hex
        started = time.monotonic()
        trace = make_trace_sink(config.tracing, run_id=run_id)
        ledger = Ledger(config)
        error: BaseException | None = None
        run_started_event_id = trace.event(
            RunStartedPayload(
                request=public_request,
                effective_config=config,
                harness=config.harness,
                harness_fingerprint=HarnessFingerprint(config.harness),
                trace_schema_version=TRACE_SCHEMA_VERSION,
                environment_abi=ENVIRONMENT_ABI.to_dict(),
                environment_abi_digest=ENVIRONMENT_ABI.digest(),
            ),
            branch_id="branch-root",
            depth=0,
        )
        try:
            response, response_event_id = self._call_model(
                backend=self.backend,
                request=public_request,
                trace=trace,
                ledger=ledger,
                run_id=run_id,
                branch_id="branch-root",
                depth=0,
                parent_event_id=run_started_event_id,
                role=ModelRole.PUBLIC,
            )
            terminal_error = validate_terminal_response(response)
            if terminal_error is not None:
                raise BackendProtocolError(terminal_error)
            trace.event(
                RunCompletedPayload(stop_reason="direct", response=response),
                branch_id="branch-root",
                depth=0,
                parent_event_id=response_event_id,
            )
            return RunResult(
                response=response,
                run_id=run_id,
                stop_reason="direct",
                usage=ledger.snapshot(),
                turns=0,
                duration_seconds=time.monotonic() - started,
                trace_directory=trace.directory,
                controller_identity=ControllerRunIdentity(
                    model=config.controller.model or str(public_request["model"]),
                    options_sha256=strict_json_sha256(config.controller.options),
                ),
                harness_fingerprint=config.harness.fingerprint(),
            )
        except BaseException as exc:
            error = exc
            trace.event(
                RunFailedPayload(error=_error_record(exc)),
                branch_id="branch-root",
                depth=0,
                parent_event_id=_failure_parent_event_id(exc) or run_started_event_id,
            )
            raise
        finally:
            trace.close(
                manifest_factory=lambda: _trace_manifest(
                    run_id=run_id,
                    error=error,
                    stop_reason="direct" if error is None else "error",
                    turns=0,
                    started=started,
                    ledger=ledger,
                    config=config,
                ),
                cause=error,
            )

    def close(self) -> None:
        """Close each distinct optional backend and surface every failure."""

        seen: set[int] = set()
        failures: list[BackendCloseFailure] = []
        first_error: Exception | None = None
        for backend in (self.backend, self.controller_backend):
            if id(backend) in seen:
                continue
            seen.add(id(backend))
            if not isinstance(backend, CloseableBackend):
                continue
            try:
                backend.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                failures.append(
                    BackendCloseFailure(
                        backend_name=_backend_name(backend),
                        exception_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        if failures:
            assert first_error is not None
            raise BackendCloseError(failures) from first_error

    def __enter__(self) -> RLM:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run_branch(
        self,
        *,
        request: dict[str, Any],
        config: RLMConfig,
        trace: TraceSink,
        ledger: Ledger,
        run_id: str,
        depth: int,
        branch_id: str,
        parent_event_id: EventRef,
        on_root_turn: Callable[[int], None] | None = None,
        action_deadline: float | None = None,
    ) -> _BranchResult:
        """Run the small controller -> Python -> observation loop."""

        ledger.check()
        _check_action_deadline(action_deadline)
        allow_recursion = depth < config.limits.max_depth
        controller_model = config.controller.model or str(request["model"])
        system = render_prompt(config.harness, allow_recursion=allow_recursion)
        branch_event_id = trace.event(
            BranchStartedPayload(
                request=request,
                controller_model=controller_model,
                prompt=system,
                harness=config.harness,
                harness_fingerprint=HarnessFingerprint(config.harness),
            ),
            branch_id=branch_id,
            depth=depth,
            parent_event_id=parent_event_id,
        )
        conversation = ControllerConversation(
            system=system,
            first_user=bootstrap_message(config.harness, request, depth=depth),
            context=config.harness.context,
        )
        recovery_policy = RecoveryPolicy(config.harness.recovery)
        conversation_parent_id = branch_event_id

        with self.execution_factory(
            request=request,
            allow_recursion=allow_recursion,
            working_directory=config.execution.working_directory,
            startup_timeout=ledger.timeout(_action_timeout_cap(20.0, action_deadline)),
            max_output_chars=config.limits.max_execution_output_chars,
        ) as executor:
            if executor.abi_version != config.harness.abi_version:
                raise HarnessContractError(
                    "execution environment ABI version does not match the configured harness"
                )
            for turn in range(1, config.limits.max_turns + 1):
                if depth == 0 and on_root_turn is not None:
                    on_root_turn(turn)
                controller_response, response_event_id = self._call_model(
                    backend=self.controller_backend,
                    request=conversation.request(
                        model=controller_model,
                        options=config.controller.options,
                    ),
                    trace=trace,
                    ledger=ledger,
                    run_id=run_id,
                    branch_id=branch_id,
                    depth=depth,
                    parent_event_id=conversation_parent_id,
                    role=ModelRole.CONTROLLER,
                    action_deadline=action_deadline,
                )
                conversation.append_response(controller_response)
                if controller_response["status"] == "incomplete":
                    fault = ControllerFault(
                        FaultKind.MODEL_OUTPUT,
                        "controller response is incomplete",
                        details={
                            "incomplete_details": controller_response.get("incomplete_details"),
                        },
                    )
                    conversation_parent_id = self._repair(
                        config=config,
                        recovery_policy=recovery_policy,
                        fault=fault,
                        execution=None,
                        turn=turn,
                        conversation=conversation,
                        trace=trace,
                        causal=response_event_id,
                        branch_id=branch_id,
                        depth=depth,
                    )
                    continue
                try:
                    action = conversation.parse(controller_response)
                except ProtocolError as exc:
                    fault = ControllerFault(FaultKind.CONTROLLER_PROTOCOL, str(exc))
                    conversation_parent_id = self._repair(
                        config=config,
                        recovery_policy=recovery_policy,
                        fault=fault,
                        execution=None,
                        turn=turn,
                        conversation=conversation,
                        trace=trace,
                        causal=response_event_id,
                        branch_id=branch_id,
                        depth=depth,
                    )
                    continue
                action_event_id = trace.event(
                    ControllerActionPayload(code=action.code),
                    branch_id=branch_id,
                    depth=depth,
                    parent_event_id=response_event_id,
                )

                def handle(
                    operation: HostOperation,
                    payload: HostPayload,
                    timeout: float,
                    parent_id: EventRef = action_event_id,
                ) -> Any:
                    value = self._handle_action(
                        operation=operation,
                        payload=payload,
                        public_request=request,
                        config=config,
                        trace=trace,
                        ledger=ledger,
                        run_id=run_id,
                        depth=depth,
                        branch_id=branch_id,
                        parent_event_id=parent_id,
                        on_root_turn=on_root_turn,
                        action_deadline=time.monotonic() + timeout,
                    )
                    ledger.check()
                    return value

                execution = executor.execute(
                    action.code,
                    action_handler=handle,
                    timeout=ledger.timeout(
                        _action_timeout_cap(
                            config.limits.execution_timeout_seconds,
                            action_deadline,
                        )
                    ),
                )
                execution_event_id = trace.event(
                    ControllerExecutionPayload(execution=execution),
                    branch_id=branch_id,
                    depth=depth,
                    parent_event_id=action_event_id,
                )
                _raise_host_failure(execution)

                if execution.exception is not None:
                    fault_kind = _fault_kind_for_execution_exception(execution.exception.fault_kind)
                    fault = ControllerFault(
                        fault_kind,
                        f"{execution.exception.type}: {execution.exception.message}",
                        details=execution.exception.details,
                    )
                    conversation_parent_id = self._repair(
                        config=config,
                        recovery_policy=recovery_policy,
                        fault=fault,
                        execution=execution,
                        turn=turn,
                        conversation=conversation,
                        trace=trace,
                        causal=execution_event_id,
                        branch_id=branch_id,
                        depth=depth,
                    )
                    continue

                try:
                    final = self._submitted_final(
                        request=request,
                        execution=execution,
                        trace=trace,
                        ledger=ledger,
                        branch_id=branch_id,
                        depth=depth,
                        parent_event_id=execution_event_id,
                        turn=turn,
                    )
                except ProtocolError as exc:
                    fault = ControllerFault(FaultKind.FINAL_SUBMISSION, str(exc))
                    conversation_parent_id = self._repair(
                        config=config,
                        recovery_policy=recovery_policy,
                        fault=fault,
                        execution=execution,
                        turn=turn,
                        conversation=conversation,
                        trace=trace,
                        causal=execution_event_id,
                        branch_id=branch_id,
                        depth=depth,
                    )
                    continue
                if final is not None:
                    return final

                observation = controller_observation(
                    spec=config.harness.observation,
                    fault=None,
                    execution=execution,
                    max_chars=config.limits.max_observation_chars,
                )
                conversation.append_observation(observation)
                conversation_parent_id = trace.event(
                    ControllerObservationPayload(observation=observation),
                    branch_id=branch_id,
                    depth=depth,
                    parent_event_id=execution_event_id,
                )

        raise LimitExceededError(
            f"controller did not submit a final response within {config.limits.max_turns} turns",
            code="max_turns_exceeded",
        )

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
        """Apply the one sealed transition for every recoverable controller fault."""

        decision = recovery_policy.decide(fault, turn=turn)
        decision_event = trace.event(
            RecoveryDecisionPayload(turn=turn, decision=decision),
            branch_id=branch_id,
            depth=depth,
            parent_event_id=causal,
        )
        if isinstance(decision, Abort):
            raise RecoveryAbortedError(
                fault,
                causal_parent=TraceCausalParent(decision_event),
            )
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

    def _submitted_final(
        self,
        *,
        request: dict[str, Any],
        execution: ExecutionResult,
        trace: TraceSink,
        ledger: Ledger,
        branch_id: str,
        depth: int,
        parent_event_id: EventRef,
        turn: int,
    ) -> _BranchResult | None:
        if execution.submission is None:
            return None
        if isinstance(execution.submission, ResponseSubmission):
            error = validate_terminal_response(execution.submission.response)
            if error:
                raise ProtocolError(error)
            response = copy.deepcopy(execution.submission.response)
            final_kind = "response"
        elif isinstance(execution.submission, TextSubmission):
            if requires_complete_response(request):
                raise ProtocolError("this request requires FINAL_RESPONSE, not FINAL_TEXT")
            response = text_response(request, execution.submission.text, usage=ledger.snapshot())
            final_kind = "text"
        else:
            raise ProtocolError("unknown execution submission")

        branch_event_id = trace.event(
            BranchCompletedPayload(
                stop_reason=f"final_{final_kind}",
                response=response,
            ),
            branch_id=branch_id,
            depth=depth,
            parent_event_id=parent_event_id,
        )
        return _BranchResult(response, f"final_{final_kind}", turn, branch_event_id)

    def _handle_action(
        self,
        *,
        operation: HostOperation,
        payload: HostPayload,
        public_request: dict[str, Any],
        config: RLMConfig,
        trace: TraceSink,
        ledger: Ledger,
        run_id: str,
        depth: int,
        branch_id: str,
        parent_event_id: EventRef,
        on_root_turn: Callable[[int], None] | None,
        action_deadline: float,
    ) -> Any:
        match operation, payload:
            case HostOperation.MODEL_COMPLETE, HostRequestPayload(request=supplied):
                ledger.reserve_subcalls()
                body = (
                    copy.deepcopy(public_request)
                    if supplied is None
                    else _validate_request(supplied)
                )
                response, _ = self._call_model(
                    backend=self.backend,
                    request=body,
                    trace=trace,
                    ledger=ledger,
                    run_id=run_id,
                    branch_id=branch_id,
                    depth=depth,
                    parent_event_id=parent_event_id,
                    role=ModelRole.PUBLIC if supplied is None else ModelRole.SUBCALL,
                    action_deadline=action_deadline,
                )
                return response
            case HostOperation.MODEL_COMPLETE_BATCH, HostBatchPayload(requests=requests):
                bodies = _request_list(requests)
                ledger.reserve_subcalls(len(bodies))

                def complete_one(body: dict[str, Any]) -> dict[str, Any]:
                    response, _ = self._call_model(
                        backend=self.backend,
                        request=body,
                        trace=trace,
                        ledger=ledger,
                        run_id=run_id,
                        branch_id=branch_id,
                        depth=depth,
                        parent_event_id=parent_event_id,
                        role=ModelRole.SUBCALL,
                        action_deadline=action_deadline,
                    )
                    return response

                return self._parallel_map(complete_one, bodies, config=config)
            case HostOperation.RLM_COMPLETE, HostRequestPayload(request=supplied):
                self._ensure_recursion(depth, config=config)
                ledger.reserve_subcalls()
                body = (
                    copy.deepcopy(public_request)
                    if supplied is None
                    else _validate_request(supplied)
                )
                return self._run_branch(
                    request=body,
                    config=config,
                    trace=trace,
                    ledger=ledger,
                    run_id=run_id,
                    depth=depth + 1,
                    branch_id=_child_branch_id(),
                    parent_event_id=parent_event_id,
                    on_root_turn=on_root_turn,
                    action_deadline=action_deadline,
                ).response
            case HostOperation.RLM_COMPLETE_BATCH, HostBatchPayload(requests=requests):
                self._ensure_recursion(depth, config=config)
                bodies = _request_list(requests)
                ledger.reserve_subcalls(len(bodies))

                def run_child(body: dict[str, Any]) -> dict[str, Any]:
                    return self._run_branch(
                        request=body,
                        config=config,
                        trace=trace,
                        ledger=ledger,
                        run_id=run_id,
                        depth=depth + 1,
                        branch_id=_child_branch_id(),
                        parent_event_id=parent_event_id,
                        on_root_turn=on_root_turn,
                        action_deadline=action_deadline,
                    ).response

                return self._parallel_map(run_child, bodies, config=config)
            case _:
                raise HarnessContractError(
                    f"invalid typed host operation/payload combination: {operation!r}"
                )

    def _call_model(
        self,
        *,
        backend: ModelBackend,
        request: Mapping[str, Any],
        trace: TraceSink,
        ledger: Ledger,
        run_id: str,
        branch_id: str,
        depth: int,
        parent_event_id: EventRef,
        role: ModelRole,
        action_deadline: float | None = None,
    ) -> tuple[dict[str, Any], EventRef]:
        ledger.reserve_model_call()
        call_id = uuid.uuid4().hex
        context = ModelCallContext(
            run_id=run_id,
            branch_id=branch_id,
            call_id=call_id,
            role=role,
            depth=depth,
        )
        request_copy = copy.deepcopy(dict(request))
        request_event_id = trace.event(
            ModelRequestPayload(context=context, request=request_copy),
            branch_id=branch_id,
            depth=depth,
            parent_event_id=parent_event_id,
        )
        started = time.monotonic()
        response_event_id: EventRef = None
        try:
            with ledger.model_slot():
                response = backend.complete(
                    request_copy,
                    timeout=ledger.timeout(_remaining_action_timeout(action_deadline)),
                    context=context,
                )
            _require_response_object(response)
            response_event_id = trace.event(
                ModelResponsePayload(
                    context=context,
                    duration_seconds=time.monotonic() - started,
                    response=response,
                ),
                branch_id=branch_id,
                depth=depth,
                parent_event_id=request_event_id,
            )
            _check_action_deadline(action_deadline)
            ledger.check()
            envelope_error = validate_response_envelope(response)
            if envelope_error is not None:
                raise BackendProtocolError(envelope_error)
            ledger.record_response(response)
            status = response["status"]
            if status in {"queued", "in_progress"}:
                raise BackendProtocolError(f"backend response has nonterminal status {status!r}")
            if status in {"failed", "cancelled"}:
                error = response.get("error")
                message = (
                    error.get("message")
                    if isinstance(error, Mapping) and isinstance(error.get("message"), str)
                    else f"backend response was {status}"
                )
                raise BackendResponseError(message, details={"response_error": error})
        except BaseException as exc:
            failure_record = ErrorRecord.from_dict(copy.deepcopy(_error_record(exc).to_dict()))
            failed_event_id = trace.event(
                ModelFailedPayload(
                    context=context,
                    duration_seconds=time.monotonic() - started,
                    error=failure_record,
                ),
                branch_id=branch_id,
                depth=depth,
                parent_event_id=response_event_id or request_event_id,
            )
            if isinstance(exc, Exception):
                raise RemoteRLMError.from_record(
                    failure_record,
                    causal_parent=TraceCausalParent(failed_event_id),
                ) from exc
            raise
        return response, response_event_id

    def _parallel_map(
        self, function: Any, values: Sequence[dict[str, Any]], *, config: RLMConfig
    ) -> list[Any]:
        """Run every submitted child to an outcome before propagating an ordered failure.

        Child operations trace their own typed model, recovery, and terminal events. Waiting for
        every ordinary child failure preserves those causal records even if an earlier input
        fails. The surfaced error remains the first failure in request order, matching the public
        behavior of the former ordered map while keeping its exception-local causal parent intact.
        Infrastructure ``BaseException`` values cancel queued work and propagate after cleanup.
        """

        if not values:
            return []
        workers = min(config.limits.max_parallel_model_calls, len(values))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rlm-subcall")
        try:
            futures = [pool.submit(function, value) for value in values]
            outcomes: list[_ParallelOutcome] = []
            for future in futures:
                try:
                    outcomes.append(_ParallelOutcome(value=future.result(), error=None))
                except Exception as exc:
                    outcomes.append(_ParallelOutcome(value=None, error=exc))
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

        for outcome in outcomes:
            if outcome.error is not None:
                raise outcome.error
        return [outcome.value for outcome in outcomes]

    def _ensure_recursion(self, depth: int, *, config: RLMConfig) -> None:
        if depth >= config.limits.max_depth:
            raise LimitExceededError(
                f"recursive depth {depth + 1} exceeds max_depth={config.limits.max_depth}",
                code="max_depth_exceeded",
            )


def _trace_manifest(
    *,
    run_id: str,
    error: BaseException | None,
    stop_reason: str,
    turns: int,
    started: float,
    ledger: Ledger,
    config: RLMConfig,
) -> TraceManifest:
    """Build terminal trace metadata only when an enabled sink asks for it."""

    return TraceManifest(
        run_id=run_id,
        status=RunTraceStatus.FAILED if error is not None else RunTraceStatus.COMPLETED,
        stop_reason=stop_reason,
        turns=turns,
        duration_seconds=time.monotonic() - started,
        usage=ledger.stats(),
        error=None if error is None else _error_record(error),
        harness=config.harness.to_dict(),
        harness_fingerprint=config.harness.fingerprint(),
        effective_config=config.to_dict(),
        environment_abi=ENVIRONMENT_ABI.to_dict(),
        environment_abi_digest=ENVIRONMENT_ABI.digest(),
    )


def _error_record(error: BaseException) -> ErrorRecord:
    return error_record_for_exception(error, source="engine")


def _failure_parent_event_id(error: BaseException) -> EventRef:
    if isinstance(error, RLMError) and error.causal_parent is not None:
        return error.causal_parent.event_id
    return None


def _remaining_action_timeout(action_deadline: float | None) -> float | None:
    """Return a fresh positive remainder from one executor callback deadline."""

    if action_deadline is None:
        return None
    remaining = action_deadline - time.monotonic()
    if remaining <= 0:
        raise ExecutionTimeoutError(0.0)
    return remaining


def _action_timeout_cap(cap: float, action_deadline: float | None) -> float:
    remaining = _remaining_action_timeout(action_deadline)
    return cap if remaining is None else min(cap, remaining)


def _check_action_deadline(action_deadline: float | None) -> None:
    _remaining_action_timeout(action_deadline)


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    body = _request_object(request)
    json_error = json_compatibility_error(body)
    if json_error:
        raise InvalidRequestError(
            f"request must be strict JSON: {json_error}",
            code="invalid_json_value",
            param="request",
        )
    if not isinstance(body.get("model"), str) or not body["model"].strip():
        raise InvalidRequestError("a model is required", param="model")
    if "stream" in body and body["stream"] is not False:
        raise InvalidRequestError(
            "RLM does not support streaming",
            code="streaming_not_supported",
            param="stream",
        )
    if body.get("background") is True:
        raise InvalidRequestError(
            "RLM does not support background responses",
            code="background_not_supported",
            param="background",
        )
    return body


def _validate_harness_abi(config: RLMConfig) -> None:
    validate_harness_contract(config.harness)


def _request_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidRequestError(
            f"model request must be an object, got {type(value).__name__}",
            param="request",
        )
    return copy.deepcopy(dict(value))


def _request_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise InvalidRequestError("requests must be a list of objects", param="requests")
    return [_validate_request(item) for item in value]


def _require_response_object(response: Any) -> None:
    if not isinstance(response, dict):
        raise BackendProtocolError(
            f"backend returned {type(response).__name__}, expected an object"
        )
    json_error = json_compatibility_error(response)
    if json_error:
        raise BackendProtocolError(f"backend returned invalid JSON: {json_error}")


def _backend_name(backend: object) -> str:
    """Return an optional diagnostic label without using it for identity."""

    value = getattr(backend, "name", None)
    return value if isinstance(value, str) and value else type(backend).__name__


def _raise_host_failure(execution: ExecutionResult) -> None:
    if execution.host_failure is not None:
        parent = execution.host_failure.causal_parent_event_id
        raise RemoteRLMError.from_record(
            execution.host_failure.error,
            causal_parent=TraceCausalParent(parent) if parent is not None else None,
        )


def _fault_kind_for_execution_exception(exception_kind: ExecutionFaultKind) -> FaultKind:
    """Exhaustively translate the closed worker discriminator to recovery state."""

    match exception_kind:
        case ExecutionFaultKind.MODEL_OUTPUT:
            return FaultKind.MODEL_OUTPUT
        case ExecutionFaultKind.FINAL_SUBMISSION:
            return FaultKind.FINAL_SUBMISSION
        case ExecutionFaultKind.EXECUTION:
            return FaultKind.EXECUTION
    raise AssertionError(f"unhandled execution fault kind: {exception_kind!r}")


def _truncate(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = f"\n\n... <{len(value) - max_chars} characters omitted> ...\n\n"
    available = max_chars - len(marker)
    if available <= 0:
        return value[:max_chars]
    head = available // 2
    return value[:head] + marker + value[-(available - head) :]


def _child_branch_id() -> str:
    return f"branch-{uuid.uuid4().hex[:12]}"
