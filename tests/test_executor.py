from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest

from rlm.errors import ExecutionTimeoutError
from rlm.executor import IPythonExecutor


def test_background_thread_cannot_use_host_bridge_or_corrupt_next_cell() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handle(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((operation, payload))
        return {"unexpected": True}

    request = {
        "model": "public-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with IPythonExecutor(
        api="chat.completions",
        request=request,
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
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
        api="chat.completions",
        request={"model": "public-model", "messages": []},
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(
            'FINAL_TEXT("must not escape")\nraise ValueError("cell failed")',
            action_handler=lambda *_: None,
            timeout=10,
        )

    assert result.final_kind is None
    assert result.final_value is None
    assert "ValueError: cell failed" in result.stderr
    assert "final submission ignored because the cell failed" in result.stderr


def test_final_response_requires_strict_json_values() -> None:
    code = r"""
invalid_values = [
    {"choices": [{"payload": {1, 2}}]},
    {"choices": [{"payload": float("nan")}]},
    {"choices": [{1: "non-string key"}]},
    {"choices": [{"payload": (1, 2)}]},
]
cycle = {}
cycle["self"] = cycle
invalid_values.append({"choices": [cycle]})

validation_errors = []
for invalid in invalid_values:
    try:
        FINAL_RESPONSE(invalid)
    except TypeError as exc:
        validation_errors.append(str(exc))

print("\n".join(validation_errors))
FINAL_RESPONSE(
    {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "valid"},
                "numbers": [0, 1.25],
            }
        ]
    }
)
"""
    with IPythonExecutor(
        api="chat.completions",
        request={"model": "public-model", "messages": []},
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(code, action_handler=lambda *_: None, timeout=10)

    assert result.final_kind == "response"
    assert result.final_value["choices"][0]["message"]["content"] == "valid"
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
        api="chat.completions",
        request={"model": "public-model", "messages": []},
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
        allow_recursion=False,
    ) as executor:
        result = executor.execute(code, action_handler=handle, timeout=10)

    assert not marker.exists()
    assert calls == []
    assert result.final_kind is None
    assert "executor IPC message requires strict JSON" in result.stderr
    assert "non-JSON value (ModelAuthoredValue)" in result.stderr


def test_request_snapshot_is_reused_between_cells() -> None:
    with IPythonExecutor(
        api="chat.completions",
        request={"model": "public-model", "messages": [{"role": "user", "content": "x"}]},
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
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


def test_execution_output_is_bounded_before_ipc_and_kernel_remains_usable() -> None:
    with IPythonExecutor(
        api="chat.completions",
        request={"model": "public-model", "messages": []},
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
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
    assert "RLM stdout truncated after 64 characters" in bounded.stdout
    assert "RLM stderr truncated after 64 characters" in bounded.stderr
    assert "RLM display truncated after 64 characters" in bounded.display
    assert next_cell.stdout == "still usable\n"


def test_host_callback_cannot_overrun_cell_timeout() -> None:
    executor = IPythonExecutor(
        api="chat.completions",
        request={"model": "public-model", "messages": []},
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
        allow_recursion=False,
    ).start()

    def slow_handler(*_: Any) -> dict[str, Any]:
        time.sleep(0.5)
        return {"too": "late"}

    started = time.monotonic()
    with pytest.raises(ExecutionTimeoutError):
        executor.execute(
            "model_complete({})",
            action_handler=slow_handler,
            timeout=0.05,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    assert executor._closed


def test_graceful_close_reaps_worker_and_closes_descriptors() -> None:
    executor = IPythonExecutor(
        api="chat.completions",
        request={"model": "public-model", "messages": []},
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
        allow_recursion=False,
    ).start()

    executor.close()

    assert executor.exitcode == 0
    assert executor._child is None
    assert executor._process._closed


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_close_terminates_model_spawned_descendants() -> None:
    executor = IPythonExecutor(
        api="chat.completions",
        request={"model": "public-model", "messages": []},
        worker_api="chat.completions",
        worker_model=None,
        worker_options={},
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
