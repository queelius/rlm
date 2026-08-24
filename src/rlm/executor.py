"""Persistent IPython execution in a disposable child process."""

from __future__ import annotations

import copy
import io
import multiprocessing
import os
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Protocol

from rlm.abi import (
    ENVIRONMENT_ABI,
    EnvironmentABI,
    HostOperation,
    HostPayload,
    validate_environment_bindings,
)
from rlm.errors import (
    ErrorRecord,
    ExecutionTimeoutError,
    FinalSubmissionFault,
    HostProtocolError,
    InvalidRequestError,
    ModelOutputFault,
    RLMError,
    error_record_for_exception,
)
from rlm.json import json_compatibility_error, strict_json_dumps, strict_json_loads
from rlm.protocol import extract_text
from rlm.types import (
    AskBatchResult,
    ExecutionException,
    ExecutionFaultKind,
    ExecutionResult,
    HostFailure,
    HostFailureKind,
    OutputChannel,
    OutputTruncation,
    ResponseSubmission,
    TextFailure,
    TextFailureKind,
    TextSubmission,
)

ActionHandler = Callable[[HostOperation, HostPayload, float], Any]
HostActionHandler = ActionHandler
_MAX_IPC_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExecutorCleanupFailure:
    """One cleanup-stage failure retained without lossy string formatting."""

    stage: str
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.stage, self.exception_type, self.message)
        ):
            raise ValueError("executor cleanup failure fields must be non-empty strings")

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "exception_type": self.exception_type,
            "message": self.message,
        }


class ExecutorCloseError(RLMError):
    """Typed aggregate cleanup failure chained from the first exact cause."""

    def __init__(
        self,
        failures: Sequence[ExecutorCleanupFailure],
        *,
        cleanup_interruptions: Sequence[BaseException] = (),
    ) -> None:
        if not failures:
            raise ValueError("executor close error requires at least one cleanup failure")
        self.cleanup_failures = tuple(failures)
        self.cleanup_interruptions = tuple(cleanup_interruptions)
        summary = "; ".join(
            f"{item.stage}: {item.exception_type}: {item.message}" for item in self.cleanup_failures
        )
        super().__init__(f"IPython executor cleanup failed: {summary}", code="executor_close")

    def _record_details(self) -> dict[str, Any]:
        return {"cleanup_failures": [item.to_dict() for item in self.cleanup_failures]}


class _ExecutorCleanupInterruptions(Exception):
    """Structured causal detail for additional non-ordinary cleanup failures."""

    def __init__(self, interruptions: Sequence[BaseException]) -> None:
        self.interruptions = tuple(interruptions)
        super().__init__("multiple non-ordinary executor cleanup interruptions")


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


class IPCMessageKind(str, Enum):
    READY = "ready"
    STARTUP_ERROR = "startup_error"
    SHUTDOWN = "shutdown"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    EXECUTE = "execute"
    EXECUTION_RESULT = "execution_result"
    HOST_REQUEST = "host_request"
    HOST_RESPONSE = "host_response"
    WORKER_ERROR = "worker_error"


def _host_error_record(value: Any) -> ErrorRecord:
    if not isinstance(value, Mapping):
        raise ValueError("host error frame must be an object")
    allowed = {"record", "causal_parent_event_id"}
    if set(value).difference(allowed) or "record" not in value:
        raise ValueError("host error frame must contain only a record and optional causal parent")
    parent = value.get("causal_parent_event_id")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise ValueError("host error causal parent must be a non-empty string or null")
    return ErrorRecord.from_dict(value["record"])


def _host_errors(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RLMError("executor host errors must be a list", code="executor_protocol")
    normalized: list[dict[str, Any]] = []
    for item in value:
        try:
            record = _host_error_record(item)
        except ValueError as exc:
            raise RLMError(
                f"invalid executor host error frame: {exc}", code="executor_protocol"
            ) from exc
        assert isinstance(item, Mapping)
        frame: dict[str, Any] = {"record": record.to_dict()}
        if "causal_parent_event_id" in item:
            frame["causal_parent_event_id"] = item["causal_parent_event_id"]
        normalized.append(frame)
    return normalized


def _execution_exception(value: Any) -> ExecutionException | None:
    if value is None:
        return None
    fields = {"type", "message", "fault_kind", "code", "traceback", "details"}
    if not isinstance(value, dict) or set(value) != fields:
        raise HostProtocolError("execution exception frame has an invalid field set")
    if not isinstance(value["type"], str) or not isinstance(value["message"], str):
        raise HostProtocolError("execution exception frame requires text identity fields")
    try:
        fault_kind = ExecutionFaultKind(value["fault_kind"])
    except (TypeError, ValueError) as exc:
        raise HostProtocolError("execution exception fault_kind is invalid") from exc
    if value["code"] is not None and not isinstance(value["code"], str):
        raise HostProtocolError("execution exception code must be text or null")
    if value["traceback"] is not None and not isinstance(value["traceback"], str):
        raise HostProtocolError("execution exception traceback must be text or null")
    if value["details"] is not None and not isinstance(value["details"], dict):
        raise HostProtocolError("execution exception details must be an object or null")
    return ExecutionException(
        type=value["type"],
        message=value["message"],
        fault_kind=fault_kind,
        code=value["code"],
        traceback=value["traceback"],
        details=value["details"],
    )


def _execution_fault_kind(error: BaseException) -> ExecutionFaultKind:
    """Classify actual worker exception objects before their type is serialized."""

    if isinstance(error, ModelOutputFault):
        return ExecutionFaultKind.MODEL_OUTPUT
    if isinstance(error, FinalSubmissionFault):
        return ExecutionFaultKind.FINAL_SUBMISSION
    return ExecutionFaultKind.EXECUTION


def _host_failure(value: Any) -> HostFailure | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"kind", "error"}:
        raise HostProtocolError("host failure frame has an invalid field set")
    raw_error = value["error"]
    if not isinstance(raw_error, dict) or set(raw_error) not in (
        {"record"},
        {"record", "causal_parent_event_id"},
    ):
        raise HostProtocolError("host failure error frame has an invalid field set")
    try:
        kind = HostFailureKind(value["kind"])
        record = ErrorRecord.from_dict(raw_error["record"])
    except (TypeError, ValueError) as exc:
        raise HostProtocolError(f"invalid host failure frame: {exc}") from exc
    parent = raw_error.get("causal_parent_event_id")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise HostProtocolError("host failure causal parent must be text or null")
    return HostFailure(kind=kind, error=record, causal_parent_event_id=parent)


def _submission(value: Any) -> TextSubmission | ResponseSubmission | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise HostProtocolError("execution submission must be an object or null")
    if set(value) == {"text"} and isinstance(value["text"], str):
        return TextSubmission(value["text"])
    if set(value) == {"response"} and isinstance(value["response"], dict):
        json_error = json_compatibility_error(value["response"])
        if json_error is None:
            return ResponseSubmission(value["response"])
    raise HostProtocolError("execution submission has an invalid shape")


def _truncations(value: Any) -> tuple[OutputTruncation, ...]:
    if not isinstance(value, list):
        raise HostProtocolError("execution truncations must be a list")
    decoded: list[OutputTruncation] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "channel",
            "original_chars",
            "retained_chars",
            "omitted_chars",
        }:
            raise HostProtocolError("execution truncation has an invalid field set")
        try:
            decoded.append(
                OutputTruncation(
                    channel=OutputChannel(item["channel"]),
                    original_chars=item["original_chars"],
                    retained_chars=item["retained_chars"],
                    omitted_chars=item["omitted_chars"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise HostProtocolError(f"invalid execution truncation: {exc}") from exc
    return tuple(decoded)


class _OutputBudget:
    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self.retained_chars = 0

    def take(self, count: int) -> int:
        accepted = min(count, max(0, self.max_chars - self.retained_chars))
        self.retained_chars += accepted
        return accepted


class _BoundedStringIO(io.StringIO):
    """Text capture that retains a prefix while counting discarded output."""

    def __init__(self, budget: _OutputBudget, *, channel: OutputChannel) -> None:
        super().__init__()
        self.budget = budget
        self.channel = channel
        self.total_chars = 0
        self.retained_chars = 0

    @property
    def truncated(self) -> bool:
        return self.total_chars > self.retained_chars

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError(f"write() argument must be str, not {type(value).__name__}")
        length = len(value)
        room = self.budget.take(length)
        if room:
            # Pipe messages are strict UTF-8 JSON. Replace lone surrogates in
            # only the retained prefix, avoiding a copy of discarded output.
            prefix = value[:room].encode("utf-8", errors="replace").decode("utf-8")
            super().write(prefix)
            self.retained_chars += len(prefix)
        self.total_chars += length
        # Text streams conventionally report the input length even when a
        # bounded sink deliberately discards the suffix.
        return len(value)

    def value(self) -> str:
        return super().getvalue()

    def truncation(self) -> OutputTruncation | None:
        if not self.truncated:
            return None
        return OutputTruncation(
            channel=self.channel,
            original_chars=self.total_chars,
            retained_chars=self.retained_chars,
            omitted_chars=self.total_chars - self.retained_chars,
        )


def _send_message(connection: Connection, message: dict[str, Any]) -> None:
    """Send one strict-JSON IPC frame without invoking pickle."""

    json_error = json_compatibility_error(message)
    if json_error is not None:
        raise TypeError(f"executor IPC message requires strict JSON: {json_error}")
    encoded = strict_json_dumps(message, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_IPC_BYTES:
        raise OSError(
            f"executor IPC message is {len(encoded)} bytes; limit is {_MAX_IPC_BYTES} bytes"
        )
    connection.send_bytes(encoded)


def _receive_message(connection: Connection) -> dict[str, Any]:
    """Receive one strict-JSON IPC frame without deserializing Python objects."""

    try:
        encoded = connection.recv_bytes(_MAX_IPC_BYTES)
        value = strict_json_loads(encoded)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        # Existing controller loops already treat OSError as a broken protocol
        # channel and reliably tear the worker down on that path.
        raise OSError(f"invalid executor IPC JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise OSError(f"invalid executor IPC JSON object: got {type(value).__name__}")
    json_error = json_compatibility_error(value)
    if json_error is not None:
        raise OSError(f"invalid executor IPC JSON: {json_error}")
    return value


def _message_kind(message: Mapping[str, Any]) -> IPCMessageKind:
    raw_kind = message.get("type")
    if not isinstance(raw_kind, str):
        raise HostProtocolError("executor IPC message requires a string type")
    try:
        return IPCMessageKind(raw_kind)
    except ValueError as exc:
        raise HostProtocolError(f"unknown executor IPC message type: {raw_kind!r}") from exc


def _sanitize_environment(working_directory: str) -> None:
    """Remove host credentials before any model-authored code can run."""

    safe_names = {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "TERM",
        "TZ",
        "VIRTUAL_ENV",
    }
    inherited = {name: os.environ[name] for name in safe_names if name in os.environ}
    os.environ.clear()
    os.environ.update(inherited)
    os.environ.update(
        {
            "HOME": working_directory,
            "PWD": working_directory,
            "TMPDIR": tempfile.gettempdir(),
        }
    )


def _environment_namespace_contract(abi: EnvironmentABI) -> tuple[frozenset[str], frozenset[str]]:
    """Return ABI-owned and ignored IPython names from one environment descriptor."""

    reserved = frozenset((*[item.name for item in abi.functions], abi.request_name))
    ignored = frozenset({"In", "Out", "exit", "quit", "get_ipython", "open", *reserved})
    return reserved, ignored


def _worker_main(
    connection: Connection,
    working_directory: str,
    public_request_json: bytes,
    allow_recursion: bool,
    max_output_chars: int,
) -> None:
    # Put the kernel and model-spawned descendants in their own POSIX process
    # group so the host can tear down the complete branch.
    if os.name == "posix":
        try:
            os.setsid()
        except OSError as exc:
            _send_message(
                connection,
                {
                    "type": IPCMessageKind.STARTUP_ERROR.value,
                    "error": (
                        f"could not establish POSIX process group: {type(exc).__name__}: {exc}"
                    ),
                },
            )
            connection.close()
            return
    _sanitize_environment(working_directory)
    try:
        public_request = strict_json_loads(public_request_json)
    except (TypeError, ValueError) as exc:
        _send_message(
            connection,
            {"type": IPCMessageKind.STARTUP_ERROR.value, "error": f"invalid request JSON: {exc}"},
        )
        connection.close()
        return
    if not isinstance(public_request, dict):
        _send_message(
            connection,
            {"type": IPCMessageKind.STARTUP_ERROR.value, "error": "request JSON must be an object"},
        )
        connection.close()
        return
    try:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
    except Exception as exc:
        _send_message(
            connection,
            {"type": IPCMessageKind.STARTUP_ERROR.value, "error": f"{type(exc).__name__}: {exc}"},
        )
        connection.close()
        return

    os.chdir(working_directory)
    shell = InteractiveShell()
    shell.run_line_magic("colors", "NoColor")
    state: dict[str, Any] = {
        # Spawn already gave this process a private request snapshot. Reuse it
        # across cells instead of copying a potentially huge context each turn.
        "public_request": public_request,
        "submission": None,
        "host_failure": None,
        "next_id": 0,
    }
    cell_thread_ident = threading.get_ident()
    bridge_lock = threading.Lock()

    def require_cell_thread(operation: str) -> None:
        if threading.get_ident() != cell_thread_ident:
            raise RuntimeError(
                f"{operation} may only run on the active IPython cell thread; "
                "use a provided batch helper for parallel calls"
            )

    def host_call(operation: HostOperation, payload: dict[str, Any]) -> Any:
        require_cell_thread(operation.value)
        if state["host_failure"] is not None:
            raise RuntimeError("fatal host failure is already latched")
        with bridge_lock:
            request_id = int(state["next_id"])
            state["next_id"] = request_id + 1
            _send_message(
                connection,
                {
                    "type": IPCMessageKind.HOST_REQUEST.value,
                    "request_id": request_id,
                    "operation": operation.value,
                    "payload": payload,
                },
            )
            reply = _receive_message(connection)
        if (
            _message_kind(reply) is not IPCMessageKind.HOST_RESPONSE
            or reply.get("request_id") != request_id
        ):
            raise RuntimeError(f"invalid host response: {reply!r}")
        if "error" in reply:
            if set(reply) != {"type", "request_id", "error"}:
                raise HostProtocolError("host response error frame has an invalid field set")
            error = reply["error"]
            try:
                record = _host_error_record(error)
            except ValueError as exc:
                raise HostProtocolError(f"invalid executor host error frame: {exc}") from exc
            # A malformed model-authored request is a controller coding error,
            # just like invalid Python. Surface it in the cell so the next turn
            # can repair it. Provider, limit, and transport failures stay fatal.
            if record.exception_type == "InvalidRequestError":
                raise InvalidRequestError(record.message, code=record.code)
            if state["host_failure"] is None:
                state["host_failure"] = {
                    "kind": (
                        HostFailureKind.RLM_ERROR.value
                        if record.code != "internal_error"
                        else HostFailureKind.INFRASTRUCTURE.value
                    ),
                    "error": copy.deepcopy(error),
                }
            raise RuntimeError(record.message)
        if set(reply) != {"type", "request_id", "value"}:
            raise HostProtocolError("host response value frame has an invalid field set")
        return reply.get("value")

    def model_complete(request_obj: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if request_obj is None else copy.deepcopy(dict(request_obj))
        value = host_call(HostOperation.MODEL_COMPLETE, {"request": body})
        if not isinstance(value, dict):
            raise RuntimeError("model_complete host returned a non-object")
        return value

    def model_complete_batch(
        requests: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        bodies = []
        for index, item in enumerate(requests):
            if not isinstance(item, Mapping):
                raise TypeError(
                    "model_complete_batch expects full request mappings; "
                    f"item {index} is {type(item).__name__}. Use ask_batch for text prompts."
                )
            bodies.append(copy.deepcopy(dict(item)))
        value = host_call(HostOperation.MODEL_COMPLETE_BATCH, {"requests": bodies})
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise RuntimeError("model_complete_batch host returned an invalid value")
        return value

    def _text_request(
        prompt: str,
        *,
        system: str | None,
        model: str | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body = copy.deepcopy(dict(options or {}))
        body["model"] = model or public_request.get("model")
        if not body.get("model"):
            raise ValueError("ask requires a model")
        if system is not None:
            body["instructions"] = str(system)
        body["input"] = str(prompt)
        return body

    def _text_failure(index: int, response: dict[str, Any]) -> TextFailure:
        kind = (
            TextFailureKind.INCOMPLETE_RESPONSE
            if response.get("status") == "incomplete"
            else TextFailureKind.EMPTY_TEXT
        )
        return TextFailure(index=index, kind=kind, response=response)

    def ask(
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        body = _text_request(str(prompt), system=system, model=model, options=options)
        response = model_complete(body)
        text = extract_text(response)
        if response.get("status") != "completed" or not text.strip():
            failure = _text_failure(0, response)
            raise ModelOutputFault(
                "ask returned no usable text",
                details={"failures": [failure.summary()]},
            )
        return text

    def ask_batch(
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AskBatchResult:
        requests = [
            _text_request(str(prompt), system=system, model=model, options=options)
            for prompt in prompts
        ]
        responses = model_complete_batch(requests)
        if len(responses) != len(requests):
            raise RuntimeError("model_complete_batch returned the wrong number of responses")
        texts: list[str | None] = []
        failures: list[TextFailure] = []
        for index, response in enumerate(responses):
            text = extract_text(response)
            if response.get("status") != "completed" or not text.strip():
                texts.append(None)
                failures.append(_text_failure(index, response))
            else:
                texts.append(text)
        return AskBatchResult(tuple(texts), tuple(responses), tuple(failures))

    def rlm_complete(request_obj: dict[str, Any] | None = None) -> dict[str, Any]:
        if not allow_recursion:
            raise RuntimeError("recursive RLM calls are disabled for this branch")
        body = None if request_obj is None else copy.deepcopy(dict(request_obj))
        value = host_call(HostOperation.RLM_COMPLETE, {"request": body})
        if not isinstance(value, dict):
            raise RuntimeError("rlm_complete host returned a non-object")
        return value

    def rlm_complete_batch(
        requests: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not allow_recursion:
            raise RuntimeError("recursive RLM calls are disabled for this branch")
        bodies = [copy.deepcopy(dict(item)) for item in requests]
        value = host_call(HostOperation.RLM_COMPLETE_BATCH, {"requests": bodies})
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise RuntimeError("rlm_complete_batch host returned an invalid value")
        return value

    def FINAL_RESPONSE(response: dict[str, Any]) -> None:
        require_cell_thread("FINAL_RESPONSE")
        if state["submission"] is not None:
            raise FinalSubmissionFault("only one final submission is allowed", details={})
        if not isinstance(response, Mapping):
            raise FinalSubmissionFault(
                "FINAL_RESPONSE requires an object",
                details={"received_type": type(response).__name__},
            )
        value = copy.deepcopy(dict(response))
        json_error = json_compatibility_error(value)
        if json_error is not None:
            raise FinalSubmissionFault(
                "FINAL_RESPONSE requires strict JSON",
                details={"reason": json_error},
            )
        state["submission"] = {"response": value}

    def FINAL_TEXT(text: str) -> None:
        require_cell_thread("FINAL_TEXT")
        if state["submission"] is not None:
            raise FinalSubmissionFault("only one final submission is allowed", details={})
        if not isinstance(text, str):
            raise FinalSubmissionFault(
                "FINAL_TEXT requires a string",
                details={"received_type": type(text).__name__},
            )
        state["submission"] = {"text": text}

    _, ignored = _environment_namespace_contract(ENVIRONMENT_ABI)

    def _variable_summary(value: Any) -> str:
        if isinstance(value, AskBatchResult):
            return f"AskBatchResult(len={len(value.texts)}, failures={len(value.failures)})"
        kind = type(value).__name__
        if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set, frozenset)):
            return f"{kind}(len={len(value)})"
        return kind

    def SHOW_VARS() -> dict[str, str]:
        return {
            name: _variable_summary(value)
            for name, value in sorted(shell.user_ns.items())
            if not name.startswith("_") and name not in ignored
        }

    available_bindings: dict[str, Any] = {
        "model_complete": model_complete,
        "model_complete_batch": model_complete_batch,
        "ask": ask,
        "ask_batch": ask_batch,
        "FINAL_RESPONSE": FINAL_RESPONSE,
        "FINAL_TEXT": FINAL_TEXT,
        "SHOW_VARS": SHOW_VARS,
        "rlm_complete": rlm_complete,
        "rlm_complete_batch": rlm_complete_batch,
    }
    reserved = {
        item.name: available_bindings[item.name]
        for item in ENVIRONMENT_ABI.enabled_functions(allow_recursion=allow_recursion)
    }
    enabled_names = {
        item.name for item in ENVIRONMENT_ABI.enabled_functions(allow_recursion=allow_recursion)
    }
    if set(reserved).union({ENVIRONMENT_ABI.request_name}) != enabled_names.union(
        {ENVIRONMENT_ABI.request_name}
    ):
        raise HostProtocolError("environment ABI namespace contract is inconsistent")

    validate_environment_bindings(
        ENVIRONMENT_ABI,
        reserved,
        allow_recursion=allow_recursion,
    )

    def restore_namespace() -> None:
        shell.user_ns.update(reserved)
        shell.user_ns[ENVIRONMENT_ABI.request_name] = state["public_request"]

    restore_namespace()
    _send_message(
        connection,
        {
            "type": IPCMessageKind.READY.value,
            "process_group_id": os.getpgrp() if os.name == "posix" else None,
        },
    )

    while True:
        try:
            command = _receive_message(connection)
        except EOFError:
            connection.close()
            return
        try:
            command_type = _message_kind(command)
        except HostProtocolError as exc:
            _send_message(
                connection,
                {"type": IPCMessageKind.WORKER_ERROR.value, "error": exc.message},
            )
            continue
        if command_type is IPCMessageKind.SHUTDOWN:
            if set(command) != {"type"}:
                _send_message(
                    connection,
                    {"type": IPCMessageKind.WORKER_ERROR.value, "error": "invalid shutdown frame"},
                )
                continue
            _send_message(connection, {"type": IPCMessageKind.SHUTDOWN_COMPLETE.value})
            connection.close()
            return
        if command_type is not IPCMessageKind.EXECUTE or set(command) != {"type", "code"}:
            _send_message(
                connection,
                {"type": IPCMessageKind.WORKER_ERROR.value, "error": "invalid worker command"},
            )
            continue

        state["submission"] = None
        state["host_failure"] = None
        code = command["code"]
        if not isinstance(code, str):
            _send_message(
                connection,
                {"type": IPCMessageKind.WORKER_ERROR.value, "error": "execute code must be text"},
            )
            continue
        started = time.perf_counter()
        budget = _OutputBudget(max_output_chars)
        stdout_capture = _BoundedStringIO(budget, channel=OutputChannel.STDOUT)
        stderr_capture = _BoundedStringIO(budget, channel=OutputChannel.STDERR)
        display_capture = _BoundedStringIO(budget, channel=OutputChannel.DISPLAY)
        traceback_capture = _BoundedStringIO(budget, channel=OutputChannel.TRACEBACK)
        with capture_output() as captured:
            # Replace capture_output's unbounded StringIO instances while
            # retaining its reliable restoration of IPython global hooks.
            captured._stdout = stdout_capture
            captured._stderr = stderr_capture
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            def capture_display(
                data: Any,
                metadata: Any = None,
                source: Any = None,
                *,
                transient: Any = None,
                update: bool = False,
                _output: _BoundedStringIO = display_capture,
            ) -> None:
                del metadata, source, transient, update
                if isinstance(data, Mapping) and data.get("text/plain") is not None:
                    _output.write(str(data["text/plain"]))
                    _output.write("\n")

            def capture_displayhook(value: Any = None) -> None:
                if value is None:
                    return
                formatted, _ = shell.display_formatter.format(value)
                capture_display(formatted)

            shell.display_pub.publish = capture_display
            sys.displayhook = capture_displayhook
            original_showtraceback = shell.showtraceback
            shell.showtraceback = lambda *args, **kwargs: None
            try:
                result = shell.run_cell(code, store_history=True, silent=False)
            finally:
                shell.showtraceback = original_showtraceback
        restore_namespace()
        stdout = stdout_capture.value()
        stderr = stderr_capture.value()
        display = display_capture.value().rstrip("\n")
        error = result.error_before_exec or result.error_in_exec
        exception: dict[str, Any] | None = None
        if error is not None:
            details = getattr(error, "details", None)
            exception = {
                "type": type(error).__name__,
                "message": str(error),
                "fault_kind": _execution_fault_kind(error).value,
                "code": getattr(error, "code", None),
                "traceback": "".join(
                    shell.InteractiveTB.structured_traceback(
                        type(error),
                        error,
                        error.__traceback__,
                    )
                ),
                "details": copy.deepcopy(details) if isinstance(details, dict) else None,
            }
        if exception is not None and exception["traceback"]:
            traceback_capture.write(exception["traceback"])
            exception["traceback"] = traceback_capture.value()
        submission = state["submission"]
        if error is not None or state["host_failure"] is not None:
            submission = None
        truncations = [
            capture.truncation()
            for capture in (stdout_capture, stderr_capture, display_capture, traceback_capture)
            if capture.truncation() is not None
        ]
        _send_message(
            connection,
            {
                "type": IPCMessageKind.EXECUTION_RESULT.value,
                "stdout": stdout,
                "stderr": stderr,
                "display": display,
                "duration_seconds": time.perf_counter() - started,
                "exception": exception,
                "host_failure": copy.deepcopy(state["host_failure"]),
                "submission": copy.deepcopy(submission),
                "namespace": SHOW_VARS(),
                "truncations": [
                    {
                        "channel": item.channel.value,
                        "original_chars": item.original_chars,
                        "retained_chars": item.retained_chars,
                        "omitted_chars": item.omitted_chars,
                    }
                    for item in truncations
                ],
            },
        )


class IPythonExecutor:
    """Controller for one persistent, non-sandboxed IPython child process."""

    abi_version = ENVIRONMENT_ABI.version

    def __init__(
        self,
        *,
        request: Mapping[str, Any],
        allow_recursion: bool,
        working_directory: str | Path | None = None,
        startup_timeout: float = 20.0,
        max_output_chars: int = 1_000_000,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1")
        self.startup_timeout = startup_timeout
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        if working_directory is None:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="rlm-")
            self.working_directory = Path(self._temporary_directory.name)
        else:
            self.working_directory = Path(working_directory).expanduser().resolve()
            self.working_directory.mkdir(parents=True, exist_ok=True)

        request_copy = copy.deepcopy(dict(request))
        request_error = json_compatibility_error(request_copy)
        if request_error is not None:
            raise HostProtocolError(f"executor request must be strict JSON: {request_error}")
        request_json = strict_json_dumps(request_copy, separators=(",", ":")).encode("utf-8")
        context = multiprocessing.get_context("spawn")
        self._parent, self._child = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_worker_main,
            args=(
                self._child,
                str(self.working_directory),
                request_json,
                allow_recursion,
                max_output_chars,
            ),
            name="rlm-ipython",
            daemon=True,
        )
        self._started = False
        self._closed = False
        self._process_group_id: int | None = None
        self.exitcode: int | None = None

    def start(self) -> IPythonExecutor:
        if self._started:
            raise RLMError("IPython executor has already started", code="executor_state")
        try:
            self._process.start()
            self._started = True
            # The spawned worker owns its endpoint now. Keeping the duplicate
            # open in the host masks EOF and leaks one descriptor per executor.
            self._close_child_endpoint()
            message = self._receive(self.startup_timeout, "IPython startup")
        except BaseException as exc:
            if isinstance(exc, RLMError) or not isinstance(exc, Exception):
                self._raise_after_forced_close(exc)
            primary = RLMError(
                f"IPython failed to start: {type(exc).__name__}: {exc}", code="executor_startup"
            )
            self._raise_after_forced_close(primary)
        try:
            message_type = _message_kind(message)
        except HostProtocolError as exc:
            self._raise_after_forced_close(exc)
        if message_type is not IPCMessageKind.READY or set(message) != {
            "type",
            "process_group_id",
        }:
            self._raise_after_forced_close(
                RLMError(f"IPython failed to start: {message!r}", code="executor_startup")
            )
        process_group_id = message.get("process_group_id")
        if (
            os.name == "posix"
            and isinstance(process_group_id, int)
            and process_group_id == self._process.pid
        ):
            self._process_group_id = process_group_id
        return self

    def execute(
        self,
        code: str,
        *,
        action_handler: HostActionHandler,
        timeout: float,
    ) -> ExecutionResult:
        self._ensure_running()
        try:
            _send_message(self._parent, {"type": IPCMessageKind.EXECUTE.value, "code": code})
        except (OSError, TypeError) as exc:
            self._raise_after_forced_close(
                RLMError(f"could not send IPython cell: {exc}", code="executor_protocol")
            )
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._parent.poll(remaining):
                self._raise_after_forced_close(ExecutionTimeoutError(timeout))
            try:
                message = _receive_message(self._parent)
            except (EOFError, OSError):
                self._raise_after_forced_close(
                    RLMError(
                        f"IPython exited unexpectedly with code {self._process.exitcode}",
                        code="executor_exit",
                    )
                )
            try:
                message_type = _message_kind(message)
            except HostProtocolError as exc:
                self._raise_after_forced_close(exc)
            if message_type is IPCMessageKind.HOST_REQUEST:
                if set(message) != {"type", "request_id", "operation", "payload"}:
                    self._raise_after_forced_close(
                        HostProtocolError("host request frame has an invalid field set")
                    )
                request_id = message["request_id"]
                try:
                    try:
                        operation = HostOperation(message["operation"])
                    except ValueError as exc:
                        raise HostProtocolError(
                            f"unknown host operation discriminator: {message['operation']!r}"
                        ) from exc
                    payload = operation.decode_payload(message["payload"])
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ExecutionTimeoutError(timeout)
                    value = action_handler(operation, payload, remaining)
                    if deadline - time.monotonic() <= 0:
                        raise ExecutionTimeoutError(timeout)
                    reply = {
                        "type": IPCMessageKind.HOST_RESPONSE.value,
                        "request_id": request_id,
                        "value": value,
                    }
                except ExecutionTimeoutError:
                    self._raise_after_forced_close(ExecutionTimeoutError(timeout))
                except HostProtocolError as exc:
                    self._raise_after_forced_close(exc)
                except Exception as exc:
                    error: Any = {
                        "record": error_record_for_exception(exc, source="executor_host").to_dict()
                    }
                    if isinstance(exc, RLMError) and exc.causal_parent is not None:
                        error["causal_parent_event_id"] = exc.causal_parent.event_id
                    reply = {
                        "type": IPCMessageKind.HOST_RESPONSE.value,
                        "request_id": request_id,
                        "error": error,
                    }
                try:
                    _send_message(self._parent, reply)
                except (OSError, TypeError) as exc:
                    self._raise_after_forced_close(
                        RLMError(
                            f"could not send IPython host response: {exc}",
                            code="executor_protocol",
                        )
                    )
                continue
            if message_type is IPCMessageKind.EXECUTION_RESULT:
                fields = {
                    "type",
                    "stdout",
                    "stderr",
                    "display",
                    "duration_seconds",
                    "exception",
                    "host_failure",
                    "submission",
                    "namespace",
                    "truncations",
                }
                if set(message) != fields:
                    self._raise_after_forced_close(
                        HostProtocolError("execution result frame has an invalid field set")
                    )
                try:
                    exception = _execution_exception(message["exception"])
                    host_failure = _host_failure(message["host_failure"])
                    submission = _submission(message["submission"])
                    if host_failure is not None:
                        exception = None
                        submission = None
                    elif exception is not None:
                        submission = None
                    return ExecutionResult(
                        stdout=message["stdout"],
                        stderr=message["stderr"],
                        display=message["display"],
                        exception=exception,
                        host_failure=host_failure,
                        submission=submission,
                        namespace=message["namespace"],
                        truncations=_truncations(message["truncations"]),
                        duration_seconds=message["duration_seconds"],
                    )
                except HostProtocolError as exc:
                    self._raise_after_forced_close(exc)
            self._raise_after_forced_close(
                RLMError(f"unexpected IPython message: {message!r}", code="executor_protocol")
            )

    def _receive(self, timeout: float, operation: str) -> dict[str, Any]:
        if not self._parent.poll(timeout):
            raise ExecutionTimeoutError(timeout)
        try:
            return _receive_message(self._parent)
        except (EOFError, OSError) as exc:
            raise RLMError(
                f"IPython exited during {operation} with code {self._process.exitcode}",
                code="executor_exit",
            ) from exc

    def _ensure_running(self) -> None:
        if self._closed or not self._process.is_alive():
            raise RLMError("IPython executor is not running", code="executor_state")

    def _raise_after_forced_close(self, primary: BaseException) -> None:
        """Preserve one public failure while recording exhaustive forced cleanup."""

        try:
            self.close(force=True)
        except BaseException as close_error:
            raise primary from close_error
        raise primary

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[ExecutorCleanupFailure] = []
        causes: list[Exception] = []
        interruptions: list[BaseException] = []

        def attempt(stage: str, operation: Callable[[], Any], *, default: Any = None) -> Any:
            try:
                return operation()
            except BaseException as exc:
                if isinstance(exc, Exception):
                    failures.append(
                        ExecutorCleanupFailure(
                            stage,
                            type(exc).__name__,
                            str(exc) or type(exc).__name__,
                        )
                    )
                    causes.append(exc)
                else:
                    interruptions.append(exc)
                return default

        def process_is_alive() -> bool:
            return bool(attempt("process.is_alive", self._process.is_alive, default=False))

        shutdown_acknowledged = False
        if process_is_alive() and not force:

            def shutdown() -> bool:
                try:
                    _send_message(self._parent, {"type": IPCMessageKind.SHUTDOWN.value})
                    if not self._parent.poll(2.0):
                        return False
                    message = _receive_message(self._parent)
                    return _message_kind(message) is IPCMessageKind.SHUTDOWN_COMPLETE and set(
                        message
                    ) == {"type"}
                except (EOFError, ConnectionError):
                    return False

            shutdown_acknowledged = bool(attempt("shutdown", shutdown, default=False))

        if shutdown_acknowledged and self._started:
            attempt("process.join.graceful", lambda: self._process.join(timeout=0.5))

        group_signalled = bool(
            attempt("process_group.sigterm", lambda: self._signal_process_group(signal.SIGTERM))
        )
        if process_is_alive() and not group_signalled:
            attempt("process.terminate", self._process.terminate)
        if self._started:
            attempt("process.join.terminate", lambda: self._process.join(timeout=3.0))
            if process_is_alive() or bool(
                attempt("process_group.exists", self._process_group_exists, default=False)
            ):
                group_killed = (
                    bool(
                        attempt(
                            "process_group.sigkill",
                            lambda: self._signal_process_group(signal.SIGKILL),
                        )
                    )
                    if os.name == "posix"
                    else False
                )
                if process_is_alive() and not group_killed:
                    attempt("process.kill", self._process.kill)
                attempt("process.join.kill", lambda: self._process.join(timeout=1.0))
            self.exitcode = attempt("process.exitcode", lambda: self._process.exitcode)

        attempt("child.close", self._close_child_endpoint)
        attempt("parent.close", self._parent.close)
        if not self._started or not process_is_alive():
            # Release multiprocessing's sentinel descriptor. A retained closed
            # executor otherwise leaks one descriptor (two after start).
            attempt("process.close", self._process.close)
        if self._temporary_directory is not None:
            attempt("temporary_directory.cleanup", self._temporary_directory.cleanup)
        if interruptions:
            selected = interruptions[0]
            additional = tuple(interruptions[1:])
            if failures:
                aggregate = ExecutorCloseError(failures, cleanup_interruptions=additional)
                raise selected from aggregate
            if additional:
                raise selected from _ExecutorCleanupInterruptions(additional)
            raise selected
        if failures:
            raise ExecutorCloseError(failures) from causes[0]

    def _close_child_endpoint(self) -> None:
        child = self._child
        if child is None:
            return
        self._child = None
        child.close()

    def _signal_process_group(self, signal_number: int) -> bool:
        if os.name != "posix":
            return False
        process_group_id = self._known_process_group_id()
        if process_group_id is None:
            return False
        try:
            os.killpg(process_group_id, signal_number)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def _process_group_exists(self) -> bool:
        if os.name != "posix":
            return False
        process_group_id = self._known_process_group_id()
        if process_group_id is None:
            return False
        try:
            os.killpg(process_group_id, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def _known_process_group_id(self) -> int | None:
        if self._process_group_id is not None:
            return self._process_group_id
        pid = self._process.pid
        if os.name != "posix" or pid is None:
            return None
        try:
            process_group_id = os.getpgid(pid)
        except ProcessLookupError:
            return None
        # Never signal an inherited group, which could include the host.
        if process_group_id != pid:
            return None
        self._process_group_id = process_group_id
        return process_group_id

    def __enter__(self) -> IPythonExecutor:
        return self.start()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if exception is not None:
                raise exception from close_error
            raise
