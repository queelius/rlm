"""The request-preserving RLM control loop."""

from __future__ import annotations

import copy
import dataclasses
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from rlm.backend import ModelBackend
from rlm.config import RLMConfig
from rlm.errors import InvalidRequestError, LimitExceededError, ProtocolError, RLMError
from rlm.executor import IPythonExecutor
from rlm.ledger import Ledger
from rlm.prompts import metadata_message, requires_complete_response
from rlm.protocol import ControllerConversation
from rlm.response import json_compatibility_error, text_response, validate_response
from rlm.trace import TraceRecorder
from rlm.types import API, ExecutionResult, RunResult, normalize_api


@dataclass(slots=True)
class _BranchResult:
    response: dict[str, Any]
    stop_reason: str
    turns: int


class RLM:
    """Decorate model backends with one bounded, programmable RLM loop.

    ``backend`` is the public model being wrapped. The controller and explicit
    worker subcalls use it too unless separate backends are supplied.
    """

    def __init__(
        self,
        backend: ModelBackend,
        *,
        config: RLMConfig | None = None,
        controller_backend: ModelBackend | None = None,
        worker_backend: ModelBackend | None = None,
    ) -> None:
        selected_config = config or RLMConfig()
        if getattr(backend, "controller_only", False):
            raise ValueError("a controller-only backend cannot be used as the public backend")
        if worker_backend is not None and getattr(worker_backend, "controller_only", False):
            raise ValueError("a controller-only backend cannot be used as the worker backend")
        selected_controller = controller_backend or backend
        required_action_mode = getattr(selected_controller, "required_action_mode", None)
        if required_action_mode is not None and selected_config.action_mode != required_action_mode:
            raise ValueError(
                f"the selected controller backend requires action_mode={required_action_mode!r}"
            )
        self.backend = backend
        self.controller_backend = selected_controller
        self.worker_backend = worker_backend or backend
        self.config = selected_config

    def run(self, api: API | str, request: Mapping[str, Any]) -> RunResult:
        """Run the RLM and return the response plus wrapper metadata."""

        try:
            public_api = normalize_api(str(api))
        except ValueError as exc:
            raise InvalidRequestError(str(exc), param="api") from exc
        if not isinstance(request, Mapping):
            raise InvalidRequestError(
                f"request must be a JSON object, got {type(request).__name__}",
                param="request",
            )
        public_request = copy.deepcopy(dict(request))
        json_error = json_compatibility_error(public_request)
        if json_error is not None:
            raise InvalidRequestError(
                f"request must be strict JSON: {json_error}",
                code="invalid_json_value",
                param="request",
            )
        if public_request.get("stream") is True:
            raise InvalidRequestError(
                "RLM supports only non-streaming requests; set stream to false or omit it",
                code="streaming_not_supported",
                param="stream",
            )
        if public_api == "responses" and public_request.get("background") is True:
            raise InvalidRequestError(
                "RLM does not support asynchronous background responses",
                code="background_not_supported",
                param="background",
            )
        if not self.config.controller_model and not public_request.get("model"):
            raise InvalidRequestError(
                "a model is required in the request when controller_model is not configured",
                param="model",
            )

        run_id = uuid.uuid4().hex
        started = time.monotonic()
        trace = TraceRecorder(self.config.debug, run_id=run_id)
        ledger = Ledger(self.config)
        root_branch = "branch-root"
        stop_reason = "error"
        turns = 0
        error: BaseException | None = None

        trace.event(
            "run.started",
            {
                "api": public_api,
                "request": public_request,
                "config": _public_config(self.config),
            },
            branch_id=root_branch,
            depth=0,
        )
        try:
            result = self._run_branch(
                api=public_api,
                request=public_request,
                trace=trace,
                ledger=ledger,
                depth=0,
                branch_id=root_branch,
                parent_event_id=None,
            )
            stop_reason = result.stop_reason
            turns = result.turns
            trace.event(
                "run.completed",
                {"stop_reason": stop_reason, "response": result.response},
                branch_id=root_branch,
                depth=0,
            )
            return RunResult(
                response=result.response,
                run_id=run_id,
                stop_reason=stop_reason,
                usage=ledger.snapshot(),
                turns=turns,
                duration_seconds=time.monotonic() - started,
                trace_directory=trace.directory,
            )
        except BaseException as exc:
            error = exc
            trace.event(
                "run.failed",
                {"error_type": type(exc).__name__, "message": str(exc)},
                branch_id=root_branch,
                depth=0,
            )
            raise
        finally:
            trace.close(
                manifest={
                    "run_id": run_id,
                    "status": "failed" if error is not None else "completed",
                    "stop_reason": stop_reason,
                    "turns": turns,
                    "duration_seconds": time.monotonic() - started,
                    "ledger": ledger.stats(),
                    "error": (
                        {"type": type(error).__name__, "message": str(error)}
                        if error is not None
                        else None
                    ),
                }
            )

    def complete(self, api: API | str, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run the RLM and return only the OpenAI-compatible response object."""

        return self.run(api, request).response

    def chat_completions(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.complete("chat.completions", request)

    def responses(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.complete("responses", request)

    def direct(self, api: API | str, request: Mapping[str, Any]) -> dict[str, Any]:
        """Call the wrapped public model unchanged for a baseline comparison."""

        try:
            target_api = normalize_api(str(api))
        except ValueError as exc:
            raise InvalidRequestError(str(exc), param="api") from exc
        if not isinstance(request, Mapping):
            raise InvalidRequestError(
                f"request must be a JSON object, got {type(request).__name__}",
                param="request",
            )
        public_request = copy.deepcopy(dict(request))
        json_error = json_compatibility_error(public_request)
        if json_error is not None:
            raise InvalidRequestError(
                f"request must be strict JSON: {json_error}",
                code="invalid_json_value",
                param="request",
            )
        if public_request.get("stream") is True:
            raise InvalidRequestError(
                "RLM supports only non-streaming requests; set stream to false or omit it",
                code="streaming_not_supported",
                param="stream",
            )
        if target_api == "responses" and public_request.get("background") is True:
            raise InvalidRequestError(
                "RLM does not support asynchronous background responses",
                code="background_not_supported",
                param="background",
            )
        response = self.backend.complete(target_api, public_request)
        _require_wire_response(response)
        return response

    def close(self) -> None:
        """Close each distinct backend that exposes a close method."""

        seen: set[int] = set()
        for backend in (self.backend, self.controller_backend, self.worker_backend):
            if id(backend) in seen:
                continue
            seen.add(id(backend))
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> RLM:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run_branch(
        self,
        *,
        api: API,
        request: dict[str, Any],
        trace: TraceRecorder,
        ledger: Ledger,
        depth: int,
        branch_id: str,
        parent_event_id: str | None,
    ) -> _BranchResult:
        ledger.check()
        allow_recursion = depth < self.config.max_depth
        controller_model = str(self.config.controller_model or request.get("model") or "")
        system = self.config.prompt.render(
            action_mode=self.config.action_mode,
            allow_recursion=allow_recursion,
        )
        prompt_hash = self.config.prompt.fingerprint(
            action_mode=self.config.action_mode,
            allow_recursion=allow_recursion,
        )
        branch_start_id = trace.event(
            "branch.started",
            {
                "api": api,
                "request": request,
                "controller_api": self.config.controller_api,
                "controller_model": controller_model,
                "prompt": system,
                "prompt_name": self.config.prompt.name,
                "prompt_version": self.config.prompt.version,
                "prompt_sha256": prompt_hash,
            },
            branch_id=branch_id,
            depth=depth,
            parent_event_id=parent_event_id,
        )
        conversation = ControllerConversation(
            self.config.controller_api,
            system=system,
            first_user=metadata_message(api, request, depth=depth),
        )
        worker_api = self.config.worker_api or api
        consecutive_errors = 0
        conversation_parent_id = branch_start_id

        executor_started = False
        try:
            with IPythonExecutor(
                api=api,
                request=request,
                worker_api=worker_api,
                worker_model=self.config.worker_model,
                worker_options=self.config.worker_options,
                allow_recursion=allow_recursion,
                working_directory=self.config.working_directory,
                startup_timeout=ledger.timeout(20.0),
                max_output_chars=self.config.max_execution_output_chars,
            ) as executor:
                executor_started = True
                for turn in range(1, self.config.max_turns + 1):
                    ledger.check()
                    controller_request = conversation.request(
                        model=controller_model,
                        options=self.config.controller_options,
                        action_mode=self.config.action_mode,
                    )
                    try:
                        response, response_event_id = self._backend_complete(
                            backend=self.controller_backend,
                            api=self.config.controller_api,
                            request=controller_request,
                            trace=trace,
                            ledger=ledger,
                            branch_id=branch_id,
                            depth=depth,
                            parent_event_id=conversation_parent_id,
                            role="controller",
                        )
                    except LimitExceededError:
                        raise
                    except RLMError as exc:
                        return self._direct_fallback(
                            api=api,
                            request=request,
                            trace=trace,
                            ledger=ledger,
                            depth=depth,
                            branch_id=branch_id,
                            parent_event_id=conversation_parent_id,
                            turns=turn,
                            reason=f"controller_{exc.code}",
                        )
                    action = conversation.parse(response, action_mode=self.config.action_mode)
                    action_event_id = trace.event(
                        "controller.action",
                        dataclasses.asdict(action),
                        branch_id=branch_id,
                        depth=depth,
                        parent_event_id=response_event_id,
                    )

                    if action.kind == "protocol_error":
                        consecutive_errors += 1
                        feedback = (
                            f"Controller protocol error: {action.error}. "
                            "Repair the action and continue."
                        )
                        if action.text is not None:
                            conversation.append_response(response)
                        conversation.append_feedback(feedback)
                        conversation_parent_id = trace.event(
                            "controller.feedback",
                            {"feedback": feedback, "consecutive_errors": consecutive_errors},
                            branch_id=branch_id,
                            depth=depth,
                            parent_event_id=action_event_id,
                        )
                        if consecutive_errors >= self.config.max_consecutive_errors:
                            return self._direct_fallback(
                                api=api,
                                request=request,
                                trace=trace,
                                ledger=ledger,
                                depth=depth,
                                branch_id=branch_id,
                                parent_event_id=conversation_parent_id,
                                turns=turn,
                                reason="consecutive_protocol_errors",
                            )
                        continue

                    if action.kind != "execute" or action.code is None:
                        raise ProtocolError(f"unhandled controller action: {action.kind}")

                    conversation.append_response(response)
                    execution_event_id = trace.event(
                        "ipython.started",
                        {"code": action.code},
                        branch_id=branch_id,
                        depth=depth,
                        parent_event_id=action_event_id,
                    )

                    def handle(
                        operation: str,
                        payload: dict[str, Any],
                        parent_id: str = execution_event_id,
                    ) -> Any:
                        return self._handle_host_action(
                            operation=operation,
                            payload=payload,
                            public_api=api,
                            public_request=request,
                            worker_api=worker_api,
                            trace=trace,
                            ledger=ledger,
                            depth=depth,
                            branch_id=branch_id,
                            parent_event_id=parent_id,
                        )

                    try:
                        execution = executor.execute(
                            action.code,
                            action_handler=handle,
                            timeout=ledger.timeout(self.config.execution_timeout),
                        )
                    except RLMError as exc:
                        trace.event(
                            "ipython.failed",
                            {"error_type": type(exc).__name__, "message": str(exc)},
                            branch_id=branch_id,
                            depth=depth,
                            parent_event_id=execution_event_id,
                        )
                        return self._direct_fallback(
                            api=api,
                            request=request,
                            trace=trace,
                            ledger=ledger,
                            depth=depth,
                            branch_id=branch_id,
                            parent_event_id=execution_event_id,
                            turns=turn,
                            reason=exc.code,
                        )

                    result_event_id = trace.event(
                        "ipython.completed",
                        execution.to_dict(),
                        branch_id=branch_id,
                        depth=depth,
                        parent_event_id=execution_event_id,
                    )
                    limit_error = _host_limit_error(execution)
                    if limit_error is not None:
                        code, message = limit_error
                        if code in {"model_call_limit", "deadline_exceeded"}:
                            raise LimitExceededError(message, code=code)
                        return self._direct_fallback(
                            api=api,
                            request=request,
                            trace=trace,
                            ledger=ledger,
                            depth=depth,
                            branch_id=branch_id,
                            parent_event_id=result_event_id,
                            turns=turn,
                            reason=code,
                        )

                    final_result = self._submitted_final(
                        api=api,
                        request=request,
                        execution=execution,
                        trace=trace,
                        ledger=ledger,
                        branch_id=branch_id,
                        depth=depth,
                        parent_event_id=result_event_id,
                        turn=turn,
                    )
                    if final_result is not None:
                        return final_result

                    final_rejection = _final_rejection_reason(api, request, execution)
                    observation = _visible_observation(
                        execution,
                        max_chars=self.config.max_observation_chars,
                        final_rejection=final_rejection,
                    )
                    conversation.append_observation(action, observation)
                    if (
                        execution.stderr
                        or execution.final_kind is not None
                        or execution.host_errors
                    ):
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0
                    conversation_parent_id = trace.event(
                        "controller.observation",
                        {
                            "observation": observation,
                            "raw_characters": len(execution.output),
                            "consecutive_errors": consecutive_errors,
                        },
                        branch_id=branch_id,
                        depth=depth,
                        parent_event_id=result_event_id,
                    )
                    if consecutive_errors >= self.config.max_consecutive_errors:
                        return self._direct_fallback(
                            api=api,
                            request=request,
                            trace=trace,
                            ledger=ledger,
                            depth=depth,
                            branch_id=branch_id,
                            parent_event_id=result_event_id,
                            turns=turn,
                            reason="consecutive_execution_errors",
                        )
        except LimitExceededError:
            raise
        except RLMError as exc:
            # RLM errors raised after entering the executor are handled at the
            # operation that caused them. Only an error while constructing or
            # starting the kernel is eligible for this exact-request fallback.
            if executor_started:
                raise
            reason = exc.code if exc.code.startswith("executor_") else f"executor_{exc.code}"
            return self._direct_fallback(
                api=api,
                request=request,
                trace=trace,
                ledger=ledger,
                depth=depth,
                branch_id=branch_id,
                parent_event_id=conversation_parent_id,
                turns=0,
                reason=reason,
            )

        return self._direct_fallback(
            api=api,
            request=request,
            trace=trace,
            ledger=ledger,
            depth=depth,
            branch_id=branch_id,
            parent_event_id=conversation_parent_id,
            turns=self.config.max_turns,
            reason="max_turns",
        )

    def _submitted_final(
        self,
        *,
        api: API,
        request: dict[str, Any],
        execution: ExecutionResult,
        trace: TraceRecorder,
        ledger: Ledger,
        branch_id: str,
        depth: int,
        parent_event_id: str,
        turn: int,
    ) -> _BranchResult | None:
        if execution.final_kind is None or execution.host_errors:
            return None
        rejection = _final_rejection_reason(api, request, execution)
        if rejection is not None:
            trace.event(
                "final.rejected",
                {"kind": execution.final_kind, "reason": rejection},
                branch_id=branch_id,
                depth=depth,
                parent_event_id=parent_event_id,
            )
            return None
        if execution.final_kind == "response":
            final = copy.deepcopy(execution.final_value)
            assert isinstance(final, dict)
            trace.event(
                "branch.completed",
                {"stop_reason": "final_response", "response": final},
                branch_id=branch_id,
                depth=depth,
                parent_event_id=parent_event_id,
            )
            return _BranchResult(final, "final_response", turn)
        if execution.final_kind == "text":
            final = text_response(
                api,
                request,
                str(execution.final_value),
                usage=ledger.snapshot(),
            )
            trace.event(
                "branch.completed",
                {"stop_reason": "final_text", "response": final},
                branch_id=branch_id,
                depth=depth,
                parent_event_id=parent_event_id,
            )
            return _BranchResult(final, "final_text", turn)
        raise AssertionError("validated final kind was not handled")

    def _handle_host_action(
        self,
        *,
        operation: str,
        payload: dict[str, Any],
        public_api: API,
        public_request: dict[str, Any],
        worker_api: API,
        trace: TraceRecorder,
        ledger: Ledger,
        depth: int,
        branch_id: str,
        parent_event_id: str,
    ) -> Any:
        if operation == "model_complete":
            ledger.reserve_subcalls(1)
            supplied = payload.get("request")
            if supplied is None:
                target_api = public_api
                body = copy.deepcopy(public_request)
                backend = self.backend
                role = "public_identity"
            else:
                body = _request_object(supplied)
                target_api = _target_api(payload.get("api"), default=worker_api)
                backend = self.worker_backend
                role = "worker"
            response, _ = self._backend_complete(
                backend=backend,
                api=target_api,
                request=body,
                trace=trace,
                ledger=ledger,
                branch_id=branch_id,
                depth=depth,
                parent_event_id=parent_event_id,
                role=role,
            )
            return response

        if operation == "model_complete_batch":
            bodies = _request_list(payload.get("requests"))
            ledger.reserve_subcalls(len(bodies))
            target_api = _target_api(payload.get("api"), default=worker_api)

            def one(body: dict[str, Any]) -> dict[str, Any]:
                response, _ = self._backend_complete(
                    backend=self.worker_backend,
                    api=target_api,
                    request=body,
                    trace=trace,
                    ledger=ledger,
                    branch_id=branch_id,
                    depth=depth,
                    parent_event_id=parent_event_id,
                    role="worker_batch_item",
                )
                return response

            return self._parallel_map(one, bodies)

        if operation == "rlm_complete":
            self._ensure_recursion(depth)
            ledger.reserve_subcalls(1)
            supplied = payload.get("request")
            body = copy.deepcopy(public_request) if supplied is None else _request_object(supplied)
            target_api = (
                public_api
                if supplied is None
                else _target_api(payload.get("api"), default=worker_api)
            )
            child = self._run_branch(
                api=target_api,
                request=body,
                trace=trace,
                ledger=ledger,
                depth=depth + 1,
                branch_id=_child_branch_id(),
                parent_event_id=parent_event_id,
            )
            return child.response

        if operation == "rlm_complete_batch":
            self._ensure_recursion(depth)
            bodies = _request_list(payload.get("requests"))
            ledger.reserve_subcalls(len(bodies))
            target_api = _target_api(payload.get("api"), default=worker_api)

            def one_child(body: dict[str, Any]) -> dict[str, Any]:
                child = self._run_branch(
                    api=target_api,
                    request=body,
                    trace=trace,
                    ledger=ledger,
                    depth=depth + 1,
                    branch_id=_child_branch_id(),
                    parent_event_id=parent_event_id,
                )
                return child.response

            return self._parallel_map(one_child, bodies)

        raise ProtocolError(f"unknown host operation: {operation!r}")

    def _backend_complete(
        self,
        *,
        backend: ModelBackend,
        api: API,
        request: Mapping[str, Any],
        trace: TraceRecorder,
        ledger: Ledger,
        branch_id: str,
        depth: int,
        parent_event_id: str | None,
        role: str,
    ) -> tuple[dict[str, Any], str]:
        ledger.reserve_model_call()
        call_id = uuid.uuid4().hex
        request_copy = copy.deepcopy(dict(request))
        request_event_id = trace.event(
            "model.request",
            {"call_id": call_id, "role": role, "api": api, "request": request_copy},
            branch_id=branch_id,
            depth=depth,
            parent_event_id=parent_event_id,
        )
        started = time.monotonic()
        try:
            with ledger.model_slot():
                response = backend.complete(api, request_copy)
            ledger.check()
        except BaseException as exc:
            trace.event(
                "model.failed",
                {
                    "call_id": call_id,
                    "role": role,
                    "api": api,
                    "duration_seconds": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                branch_id=branch_id,
                depth=depth,
                parent_event_id=request_event_id,
            )
            raise
        _require_wire_response(response)
        ledger.record_response(api, response)
        response_event_id = trace.event(
            "model.response",
            {
                "call_id": call_id,
                "role": role,
                "api": api,
                "duration_seconds": time.monotonic() - started,
                "response": response,
            },
            branch_id=branch_id,
            depth=depth,
            parent_event_id=request_event_id,
        )
        return response, response_event_id

    def _direct_fallback(
        self,
        *,
        api: API,
        request: dict[str, Any],
        trace: TraceRecorder,
        ledger: Ledger,
        depth: int,
        branch_id: str,
        parent_event_id: str | None,
        turns: int,
        reason: str,
    ) -> _BranchResult:
        fallback_event_id = trace.event(
            "fallback.started",
            {"reason": reason, "policy": "direct_public_model"},
            branch_id=branch_id,
            depth=depth,
            parent_event_id=parent_event_id,
        )
        response, response_event_id = self._backend_complete(
            backend=self.backend,
            api=api,
            request=request,
            trace=trace,
            ledger=ledger,
            branch_id=branch_id,
            depth=depth,
            parent_event_id=fallback_event_id,
            role="fallback",
        )
        stop_reason = f"{reason}_direct_fallback"
        trace.event(
            "branch.completed",
            {"stop_reason": stop_reason, "response": response},
            branch_id=branch_id,
            depth=depth,
            parent_event_id=response_event_id,
        )
        return _BranchResult(response, stop_reason, turns)

    def _parallel_map(self, function: Any, values: Sequence[dict[str, Any]]) -> list[Any]:
        if not values:
            return []
        workers = min(self.config.max_parallel_subcalls, len(values))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rlm-subcall") as pool:
            return list(pool.map(function, values))

    def _ensure_recursion(self, depth: int) -> None:
        if depth >= self.config.max_depth:
            raise LimitExceededError(
                f"recursive depth {depth + 1} exceeds max_depth={self.config.max_depth}",
                code="max_depth_exceeded",
            )


def _target_api(value: Any, *, default: API) -> API:
    if value is None:
        return default
    try:
        return normalize_api(str(value))
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="api") from exc


def _require_wire_response(response: Any) -> None:
    if not isinstance(response, dict):
        raise ProtocolError(
            f"backend returned {type(response).__name__}; expected a response object"
        )
    json_error = json_compatibility_error(response)
    if json_error is not None:
        raise ProtocolError(f"backend returned a non-JSON response object: {json_error}")


def _request_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidRequestError(
            f"model request must be an object, got {type(value).__name__}",
            param="request",
        )
    return copy.deepcopy(dict(value))


def _request_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise InvalidRequestError("requests must be a list of request objects", param="requests")
    return [_request_object(item) for item in value]


def _child_branch_id() -> str:
    return f"branch-{uuid.uuid4().hex[:12]}"


def _visible_observation(
    execution: ExecutionResult,
    *,
    max_chars: int,
    final_rejection: str | None,
) -> str:
    sections: list[str] = []
    if execution.stdout:
        sections.append(f"stdout:\n{execution.stdout.rstrip()}")
    if execution.stderr:
        sections.append(f"stderr:\n{execution.stderr.rstrip()}")
    if execution.display:
        sections.append(f"display:\n{execution.display.rstrip()}")
    if execution.host_errors:
        sections.append(f"host errors:\n{execution.host_errors!r}")
    if not sections:
        sections.append("<no output>")
    sections.append(f"variables: {execution.namespace!r}")
    if final_rejection is not None:
        sections.append(
            f"The {execution.final_kind!r} final value was rejected: {final_rejection}. "
            "Correct it and submit again."
        )
    return _truncate("\n\n".join(sections), max_chars=max_chars)


def _final_rejection_reason(
    api: API,
    request: Mapping[str, Any],
    execution: ExecutionResult,
) -> str | None:
    if execution.final_kind is None:
        return None
    if execution.final_kind == "response":
        return validate_response(api, execution.final_value)
    if execution.final_kind == "text":
        if requires_complete_response(api, request):
            return "the public request requires a complete response object"
        return None
    return "unknown final submission kind"


def _host_limit_error(execution: ExecutionResult) -> tuple[str, str] | None:
    limit_codes = {
        "deadline_exceeded",
        "max_depth_exceeded",
        "model_call_limit",
        "subcall_limit",
    }
    for error in execution.host_errors:
        code = error.get("code")
        if code in limit_codes:
            return str(code), str(error.get("message") or code)
    return None


def _truncate(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = f"\n\n... <{len(value) - max_chars} characters omitted> ...\n\n"
    available = max_chars - len(marker)
    if available <= 0:
        return value[:max_chars]
    head = available // 2
    tail = available - head
    return value[:head] + marker + value[-tail:]


def _public_config(config: RLMConfig) -> dict[str, Any]:
    value = dataclasses.asdict(config)
    value["prompt"] = {
        "name": config.prompt.name,
        "version": config.prompt.version,
    }
    return value
