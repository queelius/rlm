"""A skipped final scientific job must not strand independent follow-on work."""

import importlib.util
import json
from pathlib import Path


def module():
    path = Path(__file__).with_name("run_followon_queue_20260928.py")
    spec = importlib.util.spec_from_file_location("followon", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_finished_queue_with_preflight_failure_releases_independent_followon(tmp_path):
    runner = module()
    (tmp_path / "ACCEPTED.json").write_text(
        json.dumps(
            {"status": "accepted", "jobs": [{"name": "last", "output": str(tmp_path / "failed")}]}
        )
    )
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue/INVOCATION.json").write_text(json.dumps({"pid": 123, "started": 1}))
    (tmp_path / "queue/last.json").write_text(json.dumps({"returncode": 1}))
    state = runner.predecessor_state(tmp_path, process_active=lambda _: False)
    assert state["settled"] is True
    assert state["successful_jobs"] == 0


def test_live_queue_or_unreleased_scientific_owner_blocks(tmp_path, monkeypatch):
    runner = module()
    owner = tmp_path / "scientific"
    owner.mkdir()
    (owner / "OWNER-x.json").write_text("{}")
    (tmp_path / "ACCEPTED.json").write_text(
        json.dumps({"status": "accepted", "jobs": [{"name": "last", "output": str(owner)}]})
    )
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue/INVOCATION.json").write_text(json.dumps({"pid": 123, "started": 1}))
    assert not runner.predecessor_state(tmp_path, process_active=lambda _: True)["settled"]
    monkeypatch.setattr(runner.generic, "resource_released", lambda _: False)
    assert not runner.predecessor_state(tmp_path, process_active=lambda _: False)["settled"]
