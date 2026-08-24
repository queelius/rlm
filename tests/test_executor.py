from __future__ import annotations

import inspect
import os
import signal
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rlm.abi import (
    ENVIRONMENT_ABI,
    HostBatchPayload,
    HostOperation,
    HostPayload,
    HostRequestPayload,
)
from rlm.errors import ExecutionTimeoutError, HostProtocolError, RLMError, UpstreamError
from rlm.executor import ExecutorCloseError, IPythonExecutor, _worker_main
from rlm.types import (
    ExecutionException,
    ExecutionFaultKind,
    ExecutionResult,
    HostFailure,
    HostFailureKind,
    OutputChannel,
    ResponseSubmission,
    TextSubmission,
)
from tests.fakes import request, responses_text


def test_background_thread_cannot_use_host_bridge_or_corrupt_next_cell() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handle(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((operation, payload))
        return {"unexpected": True}

    request = {
        "model": "public-model",
        "input": [{"role": "user", "content": "hello"}],
    }
    with IPythonExecutor(
        request=request,
        allow_recursion=False,
    ) as executor:
        first = executor.execute(
            """
import threading

thread_errors = []

def background_call():
    try:
        model_complete()
    except Exception as exc:
        thread_errors.append(str(exc))

thread = threading.Thread(target=background_call)
thread.start()
thread.join()
print(thread_errors[0])
""",
            action_handler=handle,
            timeout=10,
        )
        second = executor.execute(
            'print("kernel still usable")',
            action_handler=handle,
            timeout=10,
        )

    assert "model_complete may only run on the active IPython cell thread" in first.stdout
    assert second.stdout == "kernel still usable\n"
    assert calls == []


def test_final_is_discarded_when_python_execution_fails() -> None:
    with IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(
            'FINAL_TEXT("must not escape")\nraise ValueError("cell failed")',
            action_handler=lambda *_: None,
            timeout=10,
        )

    assert result.submission is None
    assert result.exception is not None
    assert result.exception.type == "ValueError"


def test_user_exception_named_like_a_model_output_fault_is_plain_execution() -> None:
    """Changing worker identity classification to a class-name lookup breaks this test."""

    with IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(
            "class ModelOutputFault(Exception):\n    pass\nraise ModelOutputFault('user')",
            action_handler=lambda *_: None,
            timeout=10,
        )

    assert result.exception is not None
    assert result.exception.type == "ModelOutputFault"
    assert result.exception.fault_kind is ExecutionFaultKind.EXECUTION


def test_final_response_requires_strict_json_values() -> None:
    code = r"""
invalid_values = [
    {"output": [{"payload": {1, 2}}]},
    {"output": [{"payload": float("nan")}]},
    {"output": [{1: "non-string key"}]},
    {"output": [{"payload": (1, 2)}]},
]
cycle = {}
cycle["self"] = cycle
invalid_values.append({"output": [cycle]})

validation_errors = []
for invalid in invalid_values:
    try:
        FINAL_RESPONSE(invalid)
    except Exception as exc:
        validation_errors.append(exc.details["reason"])

print("\n".join(validation_errors))
FINAL_RESPONSE(
    {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "valid"}],
                "numbers": [0, 1.25],
            }
        ]
    }
)
"""
    with IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(code, action_handler=lambda *_: None, timeout=10)

    assert result.submission is not None
    assert result.submission.response["output"][0]["content"][0]["text"] == "valid"
    assert "non-JSON value (set)" in result.stdout
    assert "non-finite number" in result.stdout
    assert "non-string object key (int)" in result.stdout
    assert "non-JSON value (tuple)" in result.stdout
    assert "reference cycle" in result.stdout


def test_model_authored_objects_cannot_cross_ipc_pickle_boundary(tmp_path: Path) -> None:
    marker = tmp_path / "parent-unpickled.txt"
    calls: list[tuple[str, dict[str, Any]]] = []

    def handle(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((operation, payload))
        return {"unexpected": True}

    expression = (
        "__import__('pathlib').Path("
        f"{str(marker)!r}"
        ").write_text('model-authored reducer ran in host')"
    )
    code = f"""
class ModelAuthoredValue:
    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (eval, ({expression!r},))

model_complete({{"payload": ModelAuthoredValue()}})
"""

    with IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(code, action_handler=handle, timeout=10)

    assert not marker.exists()
    assert calls == []
    assert result.submission is None
    assert result.exception is not None
    assert "executor IPC message requires strict JSON" in result.exception.message
    assert "non-JSON value (ModelAuthoredValue)" in result.exception.message


def test_model_complete_batch_rejects_text_prompts_with_helper_guidance() -> None:
    with IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(
            'model_complete_batch(["classify this"])',
            action_handler=lambda *_: None,
            timeout=10,
        )

    assert result.exception is not None
    assert "model_complete_batch expects full request mappings" in result.exception.message
    assert "Use ask_batch for text prompts" in result.exception.message


def test_request_snapshot_is_reused_between_cells() -> None:
    with IPythonExecutor(
        request={"model": "public-model", "input": [{"role": "user", "content": "x"}]},
        allow_recursion=False,
    ) as executor:
        executor.execute(
            "first_request_object = request",
            action_handler=lambda *_: None,
            timeout=10,
        )
        result = executor.execute(
            "print(request is first_request_object)",
            action_handler=lambda *_: None,
            timeout=10,
        )

    assert result.stdout == "True\n"


def test_namespace_summaries_include_builtin_collection_lengths() -> None:
    with IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(
            'empty = []\nitems = [1, 2]\nmapping = {"a": 1}\ntext = "abc"\nnumber = 7',
            action_handler=lambda *_: None,
            timeout=10,
        )

    assert result.namespace["empty"] == "list(len=0)"
    assert result.namespace["items"] == "list(len=2)"
    assert result.namespace["mapping"] == "dict(len=1)"
    assert result.namespace["text"] == "str(len=3)"
    assert result.namespace["number"] == "int"


def test_execution_output_is_bounded_before_ipc_and_kernel_remains_usable() -> None:
    with IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
        max_output_chars=64,
    ) as executor:
        bounded = executor.execute(
            """
import sys
print("s" * 10_000)
print("e" * 10_000, file=sys.stderr)
get_ipython().display_pub.publish({"text/plain": "d" * 10_000})
""",
            action_handler=lambda *_: None,
            timeout=10,
        )
        next_cell = executor.execute(
            'print("still usable")',
            action_handler=lambda *_: None,
            timeout=10,
        )

    assert len(bounded.stdout) < 200
    assert len(bounded.stderr) < 200
    assert len(bounded.display) < 200
    assert {item.channel for item in bounded.truncations} == {
        OutputChannel.STDOUT,
        OutputChannel.STDERR,
        OutputChannel.DISPLAY,
    }
    assert sum(item.retained_chars for item in bounded.truncations) <= 64
    assert next_cell.stdout == "still usable\n"


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
            executor.execute("model_complete({})", action_handler=slow_handler, timeout=0.02)

    assert seen_timeouts and 0 < seen_timeouts[0] <= 0.02
    assert time.monotonic() - started >= 0.06


def test_ordinary_host_exception_round_trips_as_a_validated_error_record() -> None:
    def handler(*_: Any) -> dict[str, Any]:
        raise ValueError("ordinary host failure")

    with IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ) as executor:
        result = executor.execute("model_complete()", action_handler=handler, timeout=10)

    assert result.host_failure is not None
    record = result.host_failure.error
    assert record.exception_type == "ValueError"
    assert record.message == "ordinary host failure"
    assert record.code == "internal_error"
    assert record.body == {
        "error": {
            "message": "ordinary host failure",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }


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

    assert result.exception is not None
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

    assert result.exception is not None
    assert result.exception.type == "FinalSubmissionFault"
    assert result.submission is None


def test_final_text_rejects_non_string_without_coercion() -> None:
    with IPythonExecutor(request=request(), allow_recursion=False) as executor:
        result = executor.execute(
            'FINAL_TEXT({"alpha": 43})',
            action_handler=lambda operation, payload, timeout: None,
            timeout=10,
        )

    assert result.exception is not None
    assert result.exception.type == "FinalSubmissionFault"
    assert result.exception.fault_kind is ExecutionFaultKind.FINAL_SUBMISSION
    assert result.exception.message == "FINAL_TEXT requires a string"
    assert result.exception.details == {"received_type": "dict"}
    assert result.submission is None


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


def test_fatal_host_failure_is_latched_when_the_cell_catches_it() -> None:
    calls: list[str] = []

    def fail(operation: HostOperation, payload: HostPayload, timeout: float) -> object:
        assert operation is HostOperation.MODEL_COMPLETE
        assert isinstance(payload, HostRequestPayload)
        calls.append(str(payload.request["input"]))
        raise UpstreamError(503, {"message": "first down"}, endpoint="/responses")

    with IPythonExecutor(request=request(), allow_recursion=False) as executor:
        result = executor.execute(
            """
try:
    ask("first")
except BaseException:
    pass
try:
    ask("second")
except BaseException:
    pass
FINAL_TEXT("must not escape")
""",
            action_handler=fail,
            timeout=10,
        )

    assert calls == ["first"]
    assert result.host_failure is not None
    assert result.host_failure.error.message == "first down"
    assert result.submission is None


def test_unknown_host_operation_is_fatal_before_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm.executor as executor_module

    original_receive = executor_module._receive_message
    handled: list[HostOperation] = []

    def corrupt_operation(connection: object) -> dict[str, object]:
        frame = original_receive(connection)
        if frame.get("type") == "host_request":
            return {**frame, "operation": "not_an_operation"}
        return frame

    monkeypatch.setattr(executor_module, "_receive_message", corrupt_operation)
    with (
        IPythonExecutor(request=request(), allow_recursion=False) as executor,
        pytest.raises(HostProtocolError, match="unknown host operation"),
    ):
        executor.execute(
            "model_complete()",
            action_handler=lambda operation, payload, timeout: handled.append(operation),
            timeout=10,
        )

    assert handled == []


def test_request_namespace_contract_uses_the_abi_request_name() -> None:
    import rlm.executor as executor_module

    changed = replace(ENVIRONMENT_ABI, request_name="context")
    reserved, ignored = executor_module._environment_namespace_contract(changed)

    assert "context" in reserved
    assert "context" in ignored
    assert "request" not in reserved
    assert "request" not in ignored


def test_execution_submission_runtime_invariants_are_closed() -> None:
    with pytest.raises(TypeError, match="text"):
        TextSubmission(1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="mapping"):
        ResponseSubmission(("not", "a mapping"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="strict JSON"):
        ResponseSubmission({1: "key"})  # type: ignore[dict-item]
    cycle: dict[str, object] = {}
    cycle["cycle"] = cycle
    with pytest.raises(TypeError, match="strict JSON"):
        ResponseSubmission(cycle)
    with pytest.raises(TypeError, match="strict JSON"):
        ResponseSubmission({"number": float("nan")})

    original = {"output": []}
    submission = ResponseSubmission(original)
    original["output"].append("mutated")
    assert submission.response == {"output": []}

    failure = HostFailure(
        HostFailureKind.INFRASTRUCTURE,
        UpstreamError(503, {}, endpoint="/").to_record(),
    )
    with pytest.raises(ValueError, match="submission"):
        ExecutionResult(submission=TextSubmission("done"), host_failure=failure)
    with pytest.raises(TypeError, match="submission"):
        ExecutionResult(submission=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exception and host failure"):
        ExecutionResult(
            exception=ExecutionException("ValueError", "bad"),
            host_failure=failure,
        )


def test_exception_traceback_uses_ipython_formatting_and_shared_budget() -> None:
    with IPythonExecutor(
        request=request(),
        allow_recursion=False,
        max_output_chars=220,
    ) as executor:
        result = executor.execute(
            'raise ValueError("cell failed")',
            action_handler=lambda operation, payload, timeout: None,
            timeout=10,
        )

    assert result.exception is not None
    assert "<ipython-input-" in result.exception.traceback
    assert any(item.channel is OutputChannel.TRACEBACK for item in result.truncations)


def test_graceful_close_reaps_worker_and_closes_descriptors() -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ).start()

    executor.close()

    assert executor.exitcode == 0
    assert executor._child is None
    assert executor._process._closed


def test_close_ignores_expected_peer_teardown_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ).start()
    original_send = __import__("rlm.executor", fromlist=["_send_message"])._send_message

    def broken_shutdown(parent: object, message: dict[str, object]) -> None:
        if message == {"type": "shutdown"}:
            raise BrokenPipeError("peer closed")
        original_send(parent, message)

    monkeypatch.setattr("rlm.executor._send_message", broken_shutdown)
    executor.close()
    executor.close()

    assert executor._child is None
    assert executor._process._closed


def test_close_reports_unexpected_os_error_after_resource_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ).start()
    working_directory = executor.working_directory
    original_send = __import__("rlm.executor", fromlist=["_send_message"])._send_message

    def denied_shutdown(parent: object, message: dict[str, object]) -> None:
        if message == {"type": "shutdown"}:
            raise PermissionError("shutdown denied")
        original_send(parent, message)

    monkeypatch.setattr("rlm.executor._send_message", denied_shutdown)
    with pytest.raises(RLMError, match="shutdown denied") as caught:
        executor.close()

    assert caught.value.code == "executor_close"
    assert isinstance(caught.value.__cause__, PermissionError)
    assert executor._child is None
    assert executor._process._closed
    assert not working_directory.exists()


def test_close_collects_every_cleanup_failure_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ).start()
    calls: list[str] = []

    class FailingParent:
        def close(self) -> None:
            calls.append("parent")
            raise PermissionError("parent close denied")

    class FailingTemporaryDirectory:
        def cleanup(self) -> None:
            calls.append("temporary_directory")
            raise OSError("temporary directory cleanup failed")

    def failing_process_close() -> None:
        calls.append("process")
        raise OSError("process close failed")

    monkeypatch.setattr(executor, "_parent", FailingParent())
    monkeypatch.setattr(executor._process, "close", failing_process_close)
    executor._temporary_directory = FailingTemporaryDirectory()  # type: ignore[assignment]

    with pytest.raises(RLMError, match="parent close denied") as caught:
        executor.close(force=True)

    assert caught.value.code == "executor_close"
    records = [
        (item.stage, item.exception_type, item.message) for item in caught.value.cleanup_failures
    ]
    assert records == [
        ("parent.close", "PermissionError", "parent close denied"),
        ("process.close", "OSError", "process close failed"),
        ("temporary_directory.cleanup", "OSError", "temporary directory cleanup failed"),
    ]
    assert calls == ["parent", "process", "temporary_directory"]
    assert isinstance(caught.value.__cause__, PermissionError)


def test_context_manager_preserves_active_error_and_chains_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    )

    with pytest.raises(ValueError, match="active operation failed") as caught, executor:
        original_send = __import__("rlm.executor", fromlist=["_send_message"])._send_message

        def denied_shutdown(parent: object, message: dict[str, object]) -> None:
            if message == {"type": "shutdown"}:
                raise PermissionError("shutdown denied")
            original_send(parent, message)

        monkeypatch.setattr("rlm.executor._send_message", denied_shutdown)
        raise ValueError("active operation failed")

    assert isinstance(caught.value.__cause__, RLMError)
    assert caught.value.__cause__.code == "executor_close"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_worker_startup_fails_when_process_group_cannot_be_established(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    messages: list[dict[str, Any]] = []

    class RecordingConnection:
        def send_bytes(self, value: bytes) -> None:
            messages.append(__import__("json").loads(value))

        def close(self) -> None:
            pass

    def denied_setsid() -> None:
        raise PermissionError("setsid denied")

    monkeypatch.setattr("rlm.executor.os.setsid", denied_setsid)
    _worker_main(
        RecordingConnection(),  # type: ignore[arg-type]
        str(tmp_path),
        b'{"model":"public-model","input":[]}',
        False,
        100,
    )

    assert messages == [
        {
            "type": "startup_error",
            "error": "could not establish POSIX process group: PermissionError: setsid denied",
        }
    ]


def test_startup_error_remains_primary_when_forced_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    )

    class FailingParent:
        def close(self) -> None:
            raise PermissionError("parent close denied")

    monkeypatch.setattr(
        executor._process,
        "start",
        lambda: (_ for _ in ()).throw(OSError("start denied")),
    )
    monkeypatch.setattr(executor, "_parent", FailingParent())

    with pytest.raises(RLMError, match="start denied") as caught:
        executor.start()

    assert caught.value.code == "executor_startup"
    assert isinstance(caught.value.__cause__, ExecutorCloseError)
    assert caught.value.__cause__.cleanup_failures[0].stage == "parent.close"


def test_protocol_error_remains_primary_when_forced_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ).start()

    class FailingParent:
        def __init__(self, parent: object) -> None:
            self.parent = parent

        def send_bytes(self, value: bytes) -> None:
            self.parent.send_bytes(value)  # type: ignore[attr-defined]

        def poll(self, timeout: float) -> bool:
            return self.parent.poll(timeout)  # type: ignore[attr-defined]

        def close(self) -> None:
            self.parent.close()  # type: ignore[attr-defined]
            raise PermissionError("parent close denied")

    monkeypatch.setattr(executor, "_parent", FailingParent(executor._parent))
    monkeypatch.setattr("rlm.executor._receive_message", lambda parent: {"type": "unknown"})

    with pytest.raises(HostProtocolError, match="unknown executor IPC message") as caught:
        executor.execute("print('unreachable')", action_handler=lambda *_: None, timeout=10)

    assert isinstance(caught.value.__cause__, ExecutorCloseError)
    assert caught.value.__cause__.cleanup_failures[0].stage == "parent.close"


def test_malformed_startup_kind_remains_primary_when_forced_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    )

    class FailingParent:
        def __init__(self, parent: object) -> None:
            self.parent = parent

        def close(self) -> None:
            self.parent.close()  # type: ignore[attr-defined]
            raise PermissionError("parent close denied")

    monkeypatch.setattr(executor, "_parent", FailingParent(executor._parent))
    monkeypatch.setattr(executor, "_receive", lambda timeout, operation: {"type": "unknown"})

    with pytest.raises(HostProtocolError, match="unknown executor IPC message") as caught:
        executor.start()

    assert isinstance(caught.value.__cause__, ExecutorCloseError)
    assert caught.value.__cause__.cleanup_failures[0].stage == "parent.close"


def test_malformed_execution_truncation_remains_primary_when_forced_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ).start()

    class FailingParent:
        def __init__(self, parent: object) -> None:
            self.parent = parent

        def send_bytes(self, value: bytes) -> None:
            self.parent.send_bytes(value)  # type: ignore[attr-defined]

        def poll(self, timeout: float) -> bool:
            return self.parent.poll(timeout)  # type: ignore[attr-defined]

        def close(self) -> None:
            self.parent.close()  # type: ignore[attr-defined]
            raise PermissionError("parent close denied")

    malformed_result = {
        "type": "execution_result",
        "stdout": "",
        "stderr": "",
        "display": "",
        "duration_seconds": 0.0,
        "exception": None,
        "host_failure": None,
        "submission": None,
        "namespace": {},
        "truncations": [{"channel": "not-a-channel"}],
    }
    monkeypatch.setattr(executor, "_parent", FailingParent(executor._parent))
    monkeypatch.setattr("rlm.executor._receive_message", lambda parent: malformed_result)

    with pytest.raises(HostProtocolError, match="execution truncation") as caught:
        executor.execute("print('unreachable')", action_handler=lambda *_: None, timeout=10)

    assert isinstance(caught.value.__cause__, ExecutorCloseError)
    assert caught.value.__cause__.cleanup_failures[0].stage == "parent.close"


def test_forced_close_paths_use_the_primary_error_transition() -> None:
    source = inspect.getsource(IPythonExecutor)
    start_source = inspect.getsource(IPythonExecutor.start)
    execute_source = inspect.getsource(IPythonExecutor.execute)

    assert source.count("self.close(force=True)") == 1
    assert "if _message_kind(message)" not in start_source
    assert "except HostProtocolError as exc:" in start_source
    assert 'truncations=_truncations(message["truncations"]),' in execute_source
    assert execute_source.count("except HostProtocolError as exc:") >= 3


def test_close_preserves_interruptions_and_keeps_ordinary_cleanup_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ).start()
    calls: list[str] = []
    first = KeyboardInterrupt()
    second = SystemExit(4)

    class InterruptingParent:
        def close(self) -> None:
            calls.append("parent")
            raise first

    class FailingTemporaryDirectory:
        def cleanup(self) -> None:
            calls.append("temporary_directory")
            raise OSError("temporary directory cleanup failed")

    def interrupting_process_close() -> None:
        calls.append("process")
        raise second

    monkeypatch.setattr(executor, "_parent", InterruptingParent())
    monkeypatch.setattr(executor._process, "close", interrupting_process_close)
    executor._temporary_directory = FailingTemporaryDirectory()  # type: ignore[assignment]

    with pytest.raises(KeyboardInterrupt) as caught:
        executor.close(force=True)

    assert caught.value is first
    assert calls == ["parent", "process", "temporary_directory"]
    assert isinstance(caught.value.__cause__, ExecutorCloseError)
    assert caught.value.__cause__.cleanup_interruptions == (second,)
    assert [(item.stage, item.message) for item in caught.value.__cause__.cleanup_failures] == [
        ("temporary_directory.cleanup", "temporary directory cleanup failed"),
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_close_terminates_model_spawned_descendants() -> None:
    executor = IPythonExecutor(
        request={"model": "public-model", "input": []},
        allow_recursion=False,
    ).start()
    descendant_pid: int | None = None
    try:
        result = executor.execute(
            "import subprocess\nchild = subprocess.Popen(['sleep', '60'])\nprint(child.pid)",
            action_handler=lambda *_: None,
            timeout=10,
        )
        descendant_pid = int(result.stdout.strip())
        assert _process_is_running(descendant_pid)

        executor.close()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and _process_is_running(descendant_pid):
            time.sleep(0.01)
        assert not _process_is_running(descendant_pid)
        assert executor.exitcode == 0
    finally:
        executor.close(force=True)
        if descendant_pid is not None and _process_is_running(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


def _process_is_running(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            # A zombie is already terminated; it merely awaits reaping.
            state = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        except (IndexError, OSError):
            pass
        else:
            return state != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
