import importlib
import json
import os

import psutil


def test_live_previous_supervisor_blocks_even_without_gpu_owners(tmp_path):
    gate = importlib.import_module("coordinate_independent_queue")
    process = psutil.Process(os.getpid())
    assert not gate.ready(process.pid, process.create_time(), [tmp_path / "058", tmp_path / "059"])


def test_released_queue_allows_ownerless_preflight_but_not_unresolved_owner(tmp_path):
    gate = importlib.import_module("coordinate_independent_queue")
    output = tmp_path / "058"
    output.mkdir()
    dead_pid = 1_000_000_000
    assert not psutil.pid_exists(dead_pid)
    (output / "OWNER-fixture.json").write_text(json.dumps({"pid": dead_pid, "create_time": 1}))
    assert not gate.ready(dead_pid, 1, [output, tmp_path / "059"])
    (output / "TERMINAL-fixture.json").write_text(json.dumps({"failure": "preflight failed"}))
    assert gate.ready(dead_pid, 1, [output, tmp_path / "059"])
