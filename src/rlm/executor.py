"""Persistent IPython execution in a disposable child process."""

from __future__ import annotations

import copy
import io
import json
import multiprocessing
import os
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from rlm.errors import ExecutionTimeoutError, RLMError
from rlm.protocol import extract_text
from rlm.response import json_compatibility_error
from rlm.types import API, ExecutionResult, normalize_api

HostActionHandler = Callable[[str, dict[str, Any]], Any]
_MAX_IPC_BYTES = 64 * 1024 * 1024


class _BoundedStringIO(io.StringIO):
    """Text capture that retains a prefix while counting discarded output."""

    def __init__(self, max_chars: int, *, label: str) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.label = label
        self.total_chars = 0
        self.retained_chars = 0

    @property
    def truncated(self) -> bool:
        return self.total_chars > self.max_chars

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError(f"write() argument must be str, not {type(value).__name__}")
        length = len(value)
        room = max(0, self.max_chars - self.retained_chars)
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

    def bounded_value(self) -> str:
        value = super().getvalue()
        if not self.truncated:
            return value
        omitted = self.total_chars - self.max_chars
        return (
            f"{value}\n... [RLM {self.label} truncated after {self.max_chars} characters; "
            f"{omitted} omitted]\n"
        )


class _ActionHandlerTimeout(TimeoutError):
    pass


def _call_action_handler(
    handler: HostActionHandler,
    operation: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> Any:
    """Run one host callback without letting it block the cell deadline."""

    completed = threading.Event()
    outcome: list[tuple[bool, Any]] = []

    def invoke() -> None:
        try:
            outcome.append((True, handler(operation, payload)))
        except BaseException as exc:
            outcome.append((False, exc))
        finally:
            completed.set()

    thread = threading.Thread(
        target=invoke,
        name="rlm-host-action",
        daemon=True,
    )
    thread.start()
    if timeout <= 0 or not completed.wait(timeout):
        raise _ActionHandlerTimeout
    succeeded, value = outcome[0]
    if not succeeded:
        raise value
    return value


def _send_message(connection: Connection, message: dict[str, Any]) -> None:
    """Send one strict-JSON IPC frame without invoking pickle."""

    json_error = json_compatibility_error(message)
    if json_error is not None:
        raise TypeError(f"executor IPC message requires strict JSON: {json_error}")
    encoded = json.dumps(
        message,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_IPC_BYTES:
        raise OSError(
            f"executor IPC message is {len(encoded)} bytes; limit is {_MAX_IPC_BYTES} bytes"
        )
    connection.send_bytes(encoded)


def _receive_message(connection: Connection) -> dict[str, Any]:
    """Receive one strict-JSON IPC frame without deserializing Python objects."""

    try:
        encoded = connection.recv_bytes(_MAX_IPC_BYTES)
        value = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
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


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


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


def _worker_main(
    connection: Connection,
    working_directory: str,
    public_api: API,
    public_request: dict[str, Any],
    worker_api: API,
    worker_model: str | None,
    worker_options: dict[str, Any],
    allow_recursion: bool,
    max_output_chars: int,
) -> None:
    # Put the kernel and model-spawned descendants in their own POSIX process
    # group so the host can tear down the complete branch.
    if os.name == "posix":
        with suppress(OSError):
            os.setsid()
        # The parent signals only a group whose id matches this worker's pid,
        # so a failed setsid safely falls back to direct-process cleanup.
    _sanitize_environment(working_directory)
    try:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
    except Exception as exc:
        _send_message(
            connection,
            {"type": "startup_error", "error": f"{type(exc).__name__}: {exc}"},
        )
        connection.close()
        return

    os.chdir(working_directory)
    shell = InteractiveShell()
    shell.run_line_magic("colors", "NoColor")
    state: dict[str, Any] = {
        # Spawn already gave this process a private request snapshot. Reuse it
        # across cells instead of copying a potentially huge context each turn.
        "request": public_request,
        "final_kind": None,
        "final_value": None,
        "host_errors": [],
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

    def host_call(operation: str, payload: dict[str, Any]) -> Any:
        require_cell_thread(operation)
        with bridge_lock:
            request_id = int(state["next_id"])
            state["next_id"] = request_id + 1
            _send_message(
                connection,
                {
                    "type": "host_request",
                    "request_id": request_id,
                    "operation": operation,
                    "payload": payload,
                },
            )
            reply = _receive_message(connection)
        if reply.get("type") != "host_response" or reply.get("request_id") != request_id:
            raise RuntimeError(f"invalid host response: {reply!r}")
        error = reply.get("error")
        if error:
            if not isinstance(error, dict):
                error = {"kind": "exception", "message": str(error)}
            state["host_errors"].append(copy.deepcopy(error))
            code = error.get("code")
            prefix = f"{code}: " if code else ""
            raise RuntimeError(prefix + str(error.get("message") or error))
        return reply.get("value")

    def model_complete(
        request: Mapping[str, Any] | None = None,
        *,
        api: str | None = None,
    ) -> dict[str, Any]:
        body = None if request is None else copy.deepcopy(dict(request))
        value = host_call(
            "model_complete",
            {"api": api, "request": body},
        )
        if not isinstance(value, dict):
            raise RuntimeError("model_complete host returned a non-object")
        return value

    def model_complete_batch(
        requests: Sequence[Mapping[str, Any]],
        *,
        api: str | None = None,
    ) -> list[dict[str, Any]]:
        bodies = [copy.deepcopy(dict(item)) for item in requests]
        value = host_call("model_complete_batch", {"api": api, "requests": bodies})
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise RuntimeError("model_complete_batch host returned an invalid value")
        return value

    def _text_request(
        prompt: str,
        *,
        system: str | None,
        model: str | None,
        api: str | None,
    ) -> tuple[API, dict[str, Any]]:
        target_api = normalize_api(api) if api is not None else worker_api
        body = copy.deepcopy(worker_options)
        body["model"] = model or worker_model or public_request.get("model")
        if not body.get("model"):
            raise ValueError("ask requires a model in the request or RLM configuration")
        if target_api == "chat.completions":
            messages: list[dict[str, Any]] = []
            if system is not None:
                messages.append({"role": "system", "content": str(system)})
            messages.append({"role": "user", "content": str(prompt)})
            body["messages"] = messages
        else:
            if system is not None:
                body["instructions"] = str(system)
            body["input"] = str(prompt)
        return target_api, body

    def ask(
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        api: str | None = None,
    ) -> str:
        target_api, body = _text_request(str(prompt), system=system, model=model, api=api)
        response = model_complete(body, api=target_api)
        return extract_text(target_api, response)

    def ask_batch(
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        api: str | None = None,
    ) -> list[str]:
        requests: list[dict[str, Any]] = []
        target_api: API | None = None
        for prompt in prompts:
            item_api, body = _text_request(str(prompt), system=system, model=model, api=api)
            target_api = item_api
            requests.append(body)
        if target_api is None:
            return []
        responses = model_complete_batch(requests, api=target_api)
        return [extract_text(target_api, response) for response in responses]

    def rlm_complete(
        request: Mapping[str, Any] | None = None,
        *,
        api: str | None = None,
    ) -> dict[str, Any]:
        if not allow_recursion:
            raise RuntimeError("recursive RLM calls are disabled for this branch")
        body = None if request is None else copy.deepcopy(dict(request))
        value = host_call("rlm_complete", {"api": api, "request": body})
        if not isinstance(value, dict):
            raise RuntimeError("rlm_complete host returned a non-object")
        return value

    def rlm_complete_batch(
        requests: Sequence[Mapping[str, Any]],
        *,
        api: str | None = None,
    ) -> list[dict[str, Any]]:
        if not allow_recursion:
            raise RuntimeError("recursive RLM calls are disabled for this branch")
        bodies = [copy.deepcopy(dict(item)) for item in requests]
        value = host_call("rlm_complete_batch", {"api": api, "requests": bodies})
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise RuntimeError("rlm_complete_batch host returned an invalid value")
        return value

    def FINAL_RESPONSE(response: Mapping[str, Any]) -> None:
        require_cell_thread("FINAL_RESPONSE")
        value = copy.deepcopy(dict(response))
        json_error = json_compatibility_error(value)
        if json_error is not None:
            raise TypeError(f"FINAL_RESPONSE requires strict JSON: {json_error}")
        state["final_kind"] = "response"
        state["final_value"] = value

    def FINAL_TEXT(text: Any) -> None:
        require_cell_thread("FINAL_TEXT")
        state["final_kind"] = "text"
        state["final_value"] = str(text)

    ignored = {
        "In",
        "Out",
        "exit",
        "quit",
        "get_ipython",
        "open",
        "request",
        "api",
        "model_complete",
        "model_complete_batch",
        "ask",
        "ask_batch",
        "FINAL_RESPONSE",
        "FINAL_TEXT",
        "SHOW_VARS",
        "rlm_complete",
        "rlm_complete_batch",
    }

    def SHOW_VARS() -> dict[str, str]:
        return {
            name: type(value).__name__
            for name, value in sorted(shell.user_ns.items())
            if not name.startswith("_") and name not in ignored
        }

    reserved: dict[str, Any] = {
        "model_complete": model_complete,
        "model_complete_batch": model_complete_batch,
        "ask": ask,
        "ask_batch": ask_batch,
        "FINAL_RESPONSE": FINAL_RESPONSE,
        "FINAL_TEXT": FINAL_TEXT,
        "SHOW_VARS": SHOW_VARS,
    }
    if allow_recursion:
        reserved.update(
            {
                "rlm_complete": rlm_complete,
                "rlm_complete_batch": rlm_complete_batch,
            }
        )

    def restore_namespace() -> None:
        shell.user_ns.update(reserved)
        shell.user_ns["request"] = state["request"]
        shell.user_ns["api"] = public_api

    restore_namespace()
    _send_message(
        connection,
        {
            "type": "ready",
            "process_group_id": os.getpgrp() if os.name == "posix" else None,
        },
    )

    while True:
        try:
            command = _receive_message(connection)
        except EOFError:
            connection.close()
            return
        command_type = command.get("type")
        if command_type == "shutdown":
            _send_message(connection, {"type": "shutdown_complete"})
            connection.close()
            return
        if command_type != "execute":
            _send_message(
                connection,
                {"type": "worker_error", "error": f"unknown command: {command_type!r}"},
            )
            continue

        state["final_kind"] = None
        state["final_value"] = None
        state["host_errors"] = []
        code = str(command.get("code", ""))
        started = time.perf_counter()
        stdout_capture = _BoundedStringIO(max_output_chars, label="stdout")
        stderr_capture = _BoundedStringIO(max_output_chars, label="stderr")
        display_capture = _BoundedStringIO(max_output_chars, label="display")
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
            result = shell.run_cell(code, store_history=True, silent=False)
        restore_namespace()
        stdout = stdout_capture.bounded_value()
        stderr = stderr_capture.bounded_value()
        display = display_capture.bounded_value().rstrip("\n")
        error = result.error_before_exec or result.error_in_exec
        if error is not None and not stderr:
            stderr = f"{type(error).__name__}: {error}"
        final_value = state["final_value"]
        if state["final_kind"] is not None and (error is not None or state["host_errors"]):
            state["final_kind"] = None
            final_value = None
            stderr += "\nRuntimeError: final submission ignored because the cell failed"
        _send_message(
            connection,
            {
                "type": "execution_result",
                "stdout": stdout,
                "stderr": stderr,
                "display": display,
                "duration_seconds": time.perf_counter() - started,
                "final_kind": state["final_kind"],
                "final_value": final_value,
                "namespace": SHOW_VARS(),
                "host_errors": copy.deepcopy(state["host_errors"]),
            },
        )


class IPythonExecutor:
    """Controller for one persistent, non-sandboxed IPython child process."""

    def __init__(
        self,
        *,
        api: API,
        request: Mapping[str, Any],
        worker_api: API,
        worker_model: str | None,
        worker_options: Mapping[str, Any],
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

        context = multiprocessing.get_context("spawn")
        self._parent, self._child = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_worker_main,
            args=(
                self._child,
                str(self.working_directory),
                api,
                copy.deepcopy(dict(request)),
                worker_api,
                worker_model,
                copy.deepcopy(dict(worker_options)),
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
            self.close(force=True)
            if isinstance(exc, RLMError) or not isinstance(exc, Exception):
                raise
            raise RLMError(
                f"IPython failed to start: {type(exc).__name__}: {exc}",
                code="executor_startup",
            ) from exc
        if message.get("type") != "ready":
            self.close(force=True)
            raise RLMError(f"IPython failed to start: {message!r}", code="executor_startup")
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
            _send_message(self._parent, {"type": "execute", "code": code})
        except (OSError, TypeError) as exc:
            self.close(force=True)
            raise RLMError(f"could not send IPython cell: {exc}", code="executor_protocol") from exc
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._parent.poll(remaining):
                self.close(force=True)
                raise ExecutionTimeoutError(timeout)
            try:
                message = _receive_message(self._parent)
            except (EOFError, OSError) as exc:
                self.close(force=True)
                raise RLMError(
                    f"IPython exited unexpectedly with code {self._process.exitcode}",
                    code="executor_exit",
                ) from exc
            message_type = message.get("type")
            if message_type == "host_request":
                request_id = message.get("request_id")
                try:
                    value = _call_action_handler(
                        action_handler,
                        str(message.get("operation")),
                        dict(message.get("payload") or {}),
                        timeout=deadline - time.monotonic(),
                    )
                    reply = {"type": "host_response", "request_id": request_id, "value": value}
                except _ActionHandlerTimeout:
                    self.close(force=True)
                    raise ExecutionTimeoutError(timeout) from None
                except Exception as exc:
                    if isinstance(exc, RLMError):
                        error: Any = {
                            "kind": "rlm_error",
                            "type": type(exc).__name__,
                            "message": exc.message,
                            "code": exc.code,
                            "status_code": exc.status_code,
                        }
                    else:
                        error = {
                            "kind": "exception",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    reply = {
                        "type": "host_response",
                        "request_id": request_id,
                        "error": error,
                    }
                try:
                    _send_message(self._parent, reply)
                except (OSError, TypeError) as exc:
                    self.close(force=True)
                    raise RLMError(
                        f"could not send IPython host response: {exc}",
                        code="executor_protocol",
                    ) from exc
                continue
            if message_type == "execution_result":
                return ExecutionResult(
                    stdout=str(message.get("stdout", "")),
                    stderr=str(message.get("stderr", "")),
                    display=str(message.get("display", "")),
                    duration_seconds=float(message.get("duration_seconds", 0.0)),
                    final_kind=message.get("final_kind"),
                    final_value=message.get("final_value"),
                    namespace=dict(message.get("namespace") or {}),
                    host_errors=list(message.get("host_errors") or []),
                )
            self.close(force=True)
            raise RLMError(f"unexpected IPython message: {message!r}", code="executor_protocol")

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

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        shutdown_acknowledged = False
        if self._process.is_alive() and not force:
            try:
                _send_message(self._parent, {"type": "shutdown"})
                if self._parent.poll(2.0):
                    message = _receive_message(self._parent)
                    shutdown_acknowledged = message.get("type") == "shutdown_complete"
            except (BrokenPipeError, EOFError, OSError):
                pass

        # The acknowledgement is sent immediately before a normal worker
        # return. Give that return time to finish rather than always converting
        # a clean shutdown into SIGTERM.
        if shutdown_acknowledged and self._started:
            self._process.join(timeout=0.5)

        group_signalled = self._signal_process_group(signal.SIGTERM)
        if self._process.is_alive() and not group_signalled:
            self._process.terminate()
        if self._started:
            self._process.join(timeout=3.0)
            if self._process.is_alive() or self._process_group_exists():
                group_killed = (
                    self._signal_process_group(signal.SIGKILL) if os.name == "posix" else False
                )
                if self._process.is_alive() and not group_killed:
                    self._process.kill()
                self._process.join(timeout=1.0)
            self.exitcode = self._process.exitcode

        self._close_child_endpoint()
        self._parent.close()
        if not self._started or not self._process.is_alive():
            # Release multiprocessing's sentinel descriptor. A retained closed
            # executor otherwise leaks one descriptor (two after start).
            self._process.close()
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()

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

    def __exit__(self, *_: object) -> None:
        self.close()
