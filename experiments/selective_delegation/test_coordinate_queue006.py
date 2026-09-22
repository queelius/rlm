import importlib.util
import json
import os

import psutil


def test_live_queue005_blocks_without_any_scientific_owner(tmp_path):
    assert importlib.util.find_spec("coordinate_queue006") is not None
    import coordinate_queue006 as gate

    process = psutil.Process(os.getpid())
    assert not gate.ready(process.pid, process.create_time(), [tmp_path / "061", tmp_path / "062"])


def test_dead_queue_allows_ownerless_preflight_but_requires_existing_owner_release(tmp_path):
    import coordinate_queue006 as gate

    output = tmp_path / "061"
    output.mkdir()
    dead_pid = 1_000_000_000
    assert not psutil.pid_exists(dead_pid)
    (output / "OWNER-fixture.json").write_text(json.dumps({"pid": dead_pid, "create_time": 1}))
    assert not gate.ready(dead_pid, 1, [output, tmp_path / "062priv", tmp_path / "062pub"])
    (output / "TERMINAL-fixture.json").write_text(json.dumps({"failure": "preflight failed"}))
    assert gate.ready(dead_pid, 1, [output, tmp_path / "062priv", tmp_path / "062pub"])
