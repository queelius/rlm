import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest


def fixture_decision(root):
    source = root / "source-019"
    source.mkdir()
    script = source / "eval_direct_adapted.py"
    script.write_text("frozen collector fixture")
    (root / "HOTPOT-DIRECT-DECISION-001.json").write_text(
        json.dumps(
            {
                "status": "accepted",
                "source": "source-019",
                "schema": "hotpot-direct",
                "sha256": {
                    "source-019/eval_direct_adapted.py": hashlib.sha256(
                        script.read_bytes()
                    ).hexdigest()
                },
            }
        )
    )
    return script


def test_commands_use_hotpot_panel_and_official_panel_analysis():
    from launch_hotpot_direct import commands

    collect, analyze = commands(Path("/study"))
    assert collect[1:] == [
        "/study/source-019/eval_direct_adapted.py",
        "--panel",
        "hotpot_explorer32",
        "--cases",
        "/study/hotpot-inputs-001/cases.jsonl",
        "--helper-adapter",
        "/study/helper-sft-001/checkpoint-0036",
        "--output",
        "/study/hotpot-direct-adapted-001",
    ]
    assert analyze[1:] == [
        "/study/source-019/analyze_direct_adapted.py",
        "--output",
        "/study/hotpot-direct-adapted-001",
        "--cases",
        "/study/hotpot-inputs-001/cases.jsonl",
        "--report",
        "/study/analysis-hotpot-direct-adapted-001.json",
    ]


def test_cpu_validation_rejects_modified_source(tmp_path, monkeypatch):
    from launch_hotpot_direct import main

    script = fixture_decision(tmp_path)
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    args = Namespace(root=tmp_path, validate_only=True)
    assert main(args) == 0
    script.write_text("modified")
    with pytest.raises(RuntimeError, match="changed"):
        main(args)


def test_existing_owner_and_expired_lease_prevent_launch(tmp_path, monkeypatch):
    from launch_hotpot_direct import main

    fixture_decision(tmp_path)
    args = Namespace(root=tmp_path, validate_only=False)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "0")
    with pytest.raises(RuntimeError, match="wait/lease bound"):
        main(args)
    output = tmp_path / "hotpot-direct-adapted-001"
    output.mkdir()
    (output / "OWNER-prior.json").write_text("{}")
    with pytest.raises(RuntimeError, match="existing"):
        main(args)
