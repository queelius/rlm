import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest


def fixture_decision(root):
    source = root / "source-018"
    source.mkdir()
    script = source / "plan_only_probe.py"
    script.write_text("frozen collector fixture")
    (root / "PLAN-ONLY-DECISION-001.json").write_text(
        json.dumps(
            {
                "status": "accepted",
                "source": "source-018",
                "schema": "plan-only-decision",
                "sha256": {
                    "source-018/plan_only_probe.py": hashlib.sha256(script.read_bytes()).hexdigest()
                },
            }
        )
    )
    return script


def test_command_uses_sealed_collector_and_twenty_minute_cap():
    from launch_plan_only import command

    assert command(Path("/study"))[1:] == [
        "/study/source-018/plan_only_probe.py",
        "--root",
        "/study",
        "--output",
        "/study/plan-only-001",
        "--hours",
        ".3333333333",
    ]


def test_validation_is_cpu_only_and_rejects_changed_source(tmp_path, monkeypatch):
    from launch_plan_only import main

    script = fixture_decision(tmp_path)
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    args = Namespace(root=tmp_path, validate_only=True)
    assert main(args) == 0
    script.write_text("changed collector")
    with pytest.raises(RuntimeError, match="changed"):
        main(args)


def test_existing_owner_refused_before_wait_or_gpu_launch(tmp_path, monkeypatch):
    from launch_plan_only import main

    fixture_decision(tmp_path)
    output = tmp_path / "plan-only-001"
    output.mkdir()
    (output / "OWNER-prior.json").write_text("{}")
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    with pytest.raises(RuntimeError, match="existing"):
        main(Namespace(root=tmp_path, validate_only=False))


def test_expired_lease_refuses_launch_even_without_predecessor(tmp_path, monkeypatch):
    from launch_plan_only import main

    fixture_decision(tmp_path)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "0")
    with pytest.raises(RuntimeError, match="wait/lease bound"):
        main(Namespace(root=tmp_path, validate_only=False))
