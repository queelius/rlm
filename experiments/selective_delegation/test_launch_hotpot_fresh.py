"""Fixed replication commands, owner completion, and separate official regrades."""

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from subprocess import CompletedProcess

import pytest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed(output, condition, count):
    save(output / "OWNER-a.json", {"pid": 99999999, "create_time": 0})
    save(output / "TERMINAL-a.json", {"failure": None, "stopped": False})
    save(output / "PLAN.json", {})
    save(
        output / "SUMMARY.json",
        {
            "conditions": {
                condition: {
                    "planned_episodes": count,
                    "recorded_episodes": count,
                    "missing_episodes": 0,
                }
            },
            "unresolved_started_attempts": 0,
        },
    )


def accepted(root):
    source = root / "source-021/eval_planner.py"
    source.parent.mkdir()
    source.write_text("frozen fixture")
    save(
        root / "HOTPOT-FRESH-DECISION-001.json",
        {
            "status": "accepted",
            "source": "source-021",
            "schema": "hotpot-fresh",
            "sha256": {
                "source-021/eval_planner.py": hashlib.sha256(source.read_bytes()).hexdigest()
            },
        },
    )
    return source


def test_commands_match_fixed_policies_and_individual_official_regrades():
    from launch_hotpot_fresh import commands

    jobs = commands(Path("/study"))
    assert len(jobs) == 2
    for condition, collect, score in jobs:
        assert collect[1] == "/study/source-021/eval_planner.py"
        assert collect[collect.index("--cases") + 1] == "/study/hotpot-fresh-inputs-001/cases.jsonl"
        assert collect[collect.index("--limit") + 1] == "128"
        assert collect[collect.index("--repeats") + 1] == "2"
        assert collect[collect.index("--hours") + 1] == "1"
        assert score[1] == "/study/source-021/score_hotpot.py"
        assert "--comparison-output" not in score
        if condition == "sft":
            assert "--trained-only" in collect
            assert collect[collect.index("--helper-contract") + 1] == "trained_helper"
        else:
            assert collect[collect.index("--mode") + 1] == "direct"
            assert "--trained-only" not in collect and "--helper-adapter" not in collect


def test_predecessor_missing_or_capped_cannot_advance(tmp_path):
    from launch_hotpot_fresh import complete_release

    completed(tmp_path, "rl", 128)
    assert complete_release(tmp_path, "rl", 128)
    save(
        tmp_path / "SUMMARY.json",
        {
            "conditions": {
                "rl": {"planned_episodes": 128, "recorded_episodes": 127, "missing_episodes": 1}
            },
            "unresolved_started_attempts": 0,
        },
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        complete_release(tmp_path, "rl", 128)
    save(tmp_path / "TERMINAL-a.json", {"failure": None, "stopped": True})
    with pytest.raises(RuntimeError, match="stopped"):
        complete_release(tmp_path, "rl", 128)


@pytest.mark.parametrize("collected_count", [256, 255])
def test_validation_and_sequential_collections_then_two_regrades(
    tmp_path, monkeypatch, collected_count
):
    import launch_hotpot_fresh as launcher

    source = accepted(tmp_path)
    args = Namespace(root=tmp_path, validate_only=True)
    assert launcher.main(args) == 0
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        launcher.main(args)
    source.write_text("frozen fixture")
    completed(tmp_path / "fresh-contract-rl24-001", "rl", 128)
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(launcher.time.time()) + 20000))
    called = []

    def run(command, check):
        called.append(command)
        if command[1].endswith("eval_planner.py"):
            output = Path(command[command.index("--output") + 1])
            completed(output, "base" if "--mode" in command else "sft", collected_count)
        return CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main(Namespace(root=tmp_path, validate_only=False)) == int(
        collected_count != 256
    )
    assert [Path(c[1]).name for c in called] == (
        ["eval_planner.py", "eval_planner.py", "score_hotpot.py", "score_hotpot.py"]
        if collected_count == 256
        else ["eval_planner.py", "score_hotpot.py"]
    )
