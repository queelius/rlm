import hashlib
import json
from argparse import Namespace
from pathlib import Path

import launch_sufficiency as launch
import pytest


def decision(root, status="proposed"):
    source = root / "source-024-sufficiency"
    source.mkdir()
    script = source / "sufficiency_probe.py"
    script.write_text("frozen")
    (root / "SUFFICIENCY-DECISION-001.json").write_text(
        json.dumps(
            {
                "status": status,
                "source": source.name,
                "schema": "sufficiency-v1",
                "sha256": {
                    str(script.relative_to(root)): hashlib.sha256(script.read_bytes()).hexdigest()
                },
            }
        )
    )
    return script


def test_proposal_can_validate_but_cannot_launch(tmp_path):
    decision(tmp_path)
    assert launch.main(Namespace(root=tmp_path, validate_only=True)) == 0
    with pytest.raises(RuntimeError, match="accepted"):
        launch.main(Namespace(root=tmp_path, validate_only=False))


def test_hash_and_existing_owner_are_checked(tmp_path):
    script = decision(tmp_path, "accepted")
    output = tmp_path / "sufficiency-001"
    output.mkdir()
    (output / "OWNER-prior.json").write_text("{}")
    with pytest.raises(RuntimeError, match="existing"):
        launch.main(Namespace(root=tmp_path, validate_only=True))
    script.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        launch.main(Namespace(root=tmp_path, validate_only=True))


def test_fixed_command_offline_environment_and_expired_wait(tmp_path, monkeypatch):
    cmd = launch.command(Path("/study"))
    assert cmd[1] == "/study/source-024-sufficiency/sufficiency_probe.py"
    assert cmd[cmd.index("--cases") + 1] == "/study/sufficiency-inputs-001/cases.jsonl"
    assert float(cmd[-1]) == 1 / 3
    decision(tmp_path, "accepted")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "0")
    with pytest.raises(RuntimeError, match="wait/lease"):
        launch.main(Namespace(root=tmp_path, validate_only=False))
    assert launch.child_environment()["HF_HUB_OFFLINE"] == "1"
