import json
import subprocess
import sys

import psutil
import pytest


def test_failed_owner_release_does_not_imply_scientific_success(tmp_path):
    import run_independent_queue as queue

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    created = psutil.Process(child.pid).create_time()
    child.wait()
    (tmp_path / "OWNER-fixture.json").write_text(
        json.dumps({"pid": child.pid, "create_time": created})
    )
    (tmp_path / "TERMINAL-fixture.json").write_text(
        json.dumps({"failure": "exploratory cap", "stopped": True})
    )
    assert queue.resource_released(tmp_path)


def test_live_owner_blocks_even_with_terminal_receipt(tmp_path):
    import run_independent_queue as queue

    current = psutil.Process()
    (tmp_path / "OWNER-fixture.json").write_text(
        json.dumps({"pid": current.pid, "create_time": current.create_time()})
    )
    (tmp_path / "TERMINAL-fixture.json").write_text("{}")
    assert not queue.resource_released(tmp_path)


def test_independent_job_runs_after_failed_job_and_pins_are_checked(tmp_path):
    import run_independent_queue as queue

    first = {
        "name": "failure",
        "argv": [sys.executable, "-c", "raise SystemExit(3)"],
        "pins": {},
        "cap_seconds": 10,
    }
    second = {
        "name": "success",
        "argv": [sys.executable, "-c", "print('scientific fixture')"],
        "pins": {},
        "cap_seconds": 10,
    }
    records = queue.execute_jobs([first, second], tmp_path, deadline=queue.time.time() + 180)
    assert [r["returncode"] for r in records] == [3, 0]
    assert "scientific fixture" in (tmp_path / "success.log").read_text()
    pinned = tmp_path / "input.txt"
    pinned.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        queue.validate_pins({str(pinned): "incorrect"})
