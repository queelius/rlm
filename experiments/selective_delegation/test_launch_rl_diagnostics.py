"""The queue waits for authenticated release, not merely a terminal filename."""

import json

import pytest


def test_release_requires_terminal_and_dead_authenticated_owner(tmp_path, monkeypatch):
    import launch_rl_diagnostics as queue

    assert not queue.released(tmp_path)
    (tmp_path / "OWNER-test.json").write_text(json.dumps({"pid": 42, "create_time": 10}))
    assert not queue.released(tmp_path)
    terminal = tmp_path / "TERMINAL-test.json"
    terminal.write_text(json.dumps({"failure": None}))

    class Process:
        def create_time(self):
            return 10

        def status(self):
            return "running"

    monkeypatch.setattr(queue.psutil, "Process", lambda pid: Process())
    assert not queue.released(tmp_path)
    monkeypatch.setattr(Process, "create_time", lambda self: 11)
    assert queue.released(tmp_path)  # PID reuse is not the authenticated owner.
    terminal.write_text(json.dumps({"failure": "observed failure"}))
    with pytest.raises(RuntimeError, match="predecessor"):
        queue.released(tmp_path)
