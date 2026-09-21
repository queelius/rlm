import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest


def fixture(root, status="proposed"):
    source = root / "source-028-qampari"
    source.mkdir()
    script = source / "qampari_probe.py"
    script.write_text("frozen")
    (root / "QAMPARI-DECISION-001.json").write_text(
        json.dumps(
            {
                "status": status,
                "source": source.name,
                "schema": "qampari-v1",
                "sha256": {
                    str(script.relative_to(root)): hashlib.sha256(script.read_bytes()).hexdigest()
                },
            }
        )
    )
    return script


def test_proposal_validates_but_does_not_launch(tmp_path):
    import launch_qampari as launch

    script = fixture(tmp_path)
    assert launch.main(Namespace(root=tmp_path, validate_only=True)) == 0
    with pytest.raises(RuntimeError, match="accepted"):
        launch.main(Namespace(root=tmp_path, validate_only=False))
    script.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        launch.main(Namespace(root=tmp_path, validate_only=True))


def test_predecessor_must_have_complete_physical_and_logical_accounting(tmp_path):
    import launch_qampari as launch

    assert not launch.complete_release(tmp_path)
    (tmp_path / "OWNER-fixture.json").write_text(json.dumps({"pid": 999999999, "create_time": 1}))
    (tmp_path / "TERMINAL-fixture.json").write_text(json.dumps({"failure": None, "stopped": False}))
    summary = {
        "planned_unique_direct_calls": 80,
        "recorded_unique_direct_calls": 79,
        "missing_unique_direct_calls": 1,
        "planned_logical_slots": 320,
        "mapped_logical_slots": 316,
        "status_counts": {"scored": 79},
    }
    (tmp_path / "SUMMARY.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="incomplete"):
        launch.complete_release(tmp_path)
    summary.update(
        recorded_unique_direct_calls=80,
        missing_unique_direct_calls=0,
        mapped_logical_slots=320,
        status_counts={"scored": 80},
    )
    (tmp_path / "SUMMARY.json").write_text(json.dumps(summary))
    assert launch.complete_release(tmp_path)


def test_one_hour_command_and_expired_lease_refuse_execution(tmp_path, monkeypatch):
    import launch_qampari as launch

    command = launch.command(Path("/study"))
    assert command[1] == "/study/source-028-qampari/qampari_probe.py"
    assert command[command.index("--cases") + 1] == "/study/qampari-inputs-001/cases.jsonl"
    assert command[-2:] == ["--hours", "1"]
    fixture(tmp_path, "accepted")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "0")
    with pytest.raises(RuntimeError, match="wait/lease"):
        launch.main(Namespace(root=tmp_path, validate_only=False))
