"""CPU-only queue, fixed endpoint and native-owner release contracts."""

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from subprocess import CompletedProcess

import pytest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def accepted(root):
    source = root / "source-020/rl_planner.py"
    source.parent.mkdir()
    source.write_text("sealed fixture")
    save(
        root / "RL-CONTINUATION-DECISION-001.json",
        {
            "status": "accepted",
            "source": "source-020",
            "schema": "continuation",
            "sha256": {"source-020/rl_planner.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        },
    )
    return source


def complete(root, step=24):
    output = root / "rl-continuation-001"
    save(output / "OWNER-a.json", {"pid": 99999999, "create_time": 0})
    save(
        output / "TERMINAL-a.json",
        {
            "state": "completed_updates",
            "optimizer_steps": step,
            "failure": None,
            "additional_optimizer_steps": step - 16,
        },
    )
    helper = {"mode": "trained_helper", "adapter": str(root / "helper-sft-001/checkpoint-0036")}
    save(
        output / "PLAN.json",
        {
            "updates": 24,
            "helper_contract": helper,
            "continuation": {
                "ancestor_checkpoint": str(root / "rl-fullpass-001/checkpoint-0016"),
                "starting_step": 16,
                "additional_updates": 8,
            },
        },
    )
    checkpoint = output / f"checkpoint-{step:04d}"
    save(
        checkpoint / "STATE.json",
        {"step": step, "cursor": 0, "next_update": step + 1, "helper_contract": helper},
    )
    for name in ("optimizer.pt", "rng.pt", "adapter_model.safetensors", "adapter_config.json"):
        (checkpoint / name).write_text("fixture")
    save(
        checkpoint / "COMMIT.json",
        {
            "step": step,
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()
            },
        },
    )
    return checkpoint


def test_fixed_commands_preserve_continuation_and_matched_evaluation():
    from launch_rl_continuation import commands

    train, evaluate = commands(Path("/study"))
    opts = dict(zip(train[2::2], train[3::2], strict=True))
    assert train[1] == "/study/source-020/rl_planner.py"
    assert opts["--continue-from"] == "/study/rl-fullpass-001/checkpoint-0016"
    assert opts["--adapter"] == "/study/planner-sft-001/checkpoint-0048"
    assert opts["--updates"] == "24" and opts["--hours"] == "1.5"
    assert opts["--parent-schedule"] == "consecutive"
    assert opts["--helper-contract"] == "trained_helper"
    assert opts["--helper-adapter"] == "/study/helper-sft-001/checkpoint-0036"
    assert "--trained-only" in evaluate
    pairs = evaluate[2:]
    pairs.remove("--trained-only")
    opts = dict(zip(pairs[::2], pairs[1::2], strict=True))
    assert opts["--adapter"] == "/study/rl-continuation-001/checkpoint-0024"
    assert opts["--cases"] == "/study/fresh-dev-inputs-003/cases.jsonl"
    assert opts["--split"] == "development" and opts["--limit"] == "64"
    assert opts["--repeats"] == "2" and opts["--hours"] == "1"
    assert opts["--trained-condition"] == "rl" and opts["--execution"] == "isolated"


def test_validation_and_exact_endpoint_reject_mutation_or_substitution(tmp_path):
    from launch_rl_continuation import main, require_endpoint

    script = accepted(tmp_path)
    args = Namespace(root=tmp_path, validate_only=True)
    assert main(args) == 0
    script.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        main(args)
    complete(tmp_path, 23)
    with pytest.raises(RuntimeError, match="checkpoint24"):
        require_endpoint(tmp_path)


def test_success_gates_evaluation_and_endpoint_hashes(tmp_path, monkeypatch):
    import launch_rl_continuation as launcher

    accepted(tmp_path)
    save(tmp_path / "hotpot-direct-adapted-001/OWNER-a.json", {"pid": 99999999, "create_time": 0})
    save(tmp_path / "hotpot-direct-adapted-001/TERMINAL-a.json", {"failure": None})
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(launcher.time.time()) + 20000))
    called = []

    def run(command, check):
        called.append(command)
        if len(called) == 1:
            complete(tmp_path)
        return CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main(Namespace(root=tmp_path, validate_only=False)) == 0
    assert len(called) == 2
    checkpoint = tmp_path / "rl-continuation-001/checkpoint-0024"
    (checkpoint / "optimizer.pt").write_text("changed")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        launcher.require_endpoint(tmp_path)


def test_failed_training_never_launches_evaluation(tmp_path, monkeypatch):
    import launch_rl_continuation as launcher

    accepted(tmp_path)
    save(tmp_path / "hotpot-direct-adapted-001/OWNER-a.json", {"pid": 99999999, "create_time": 0})
    save(tmp_path / "hotpot-direct-adapted-001/TERMINAL-a.json", {"failure": None})
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(launcher.time.time()) + 20000))
    calls = []

    def run(command, check):
        calls.append(command)
        return CompletedProcess(command, 1)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main(Namespace(root=tmp_path, validate_only=False)) == 1
    assert len(calls) == 1


def test_existing_attempt_and_short_allocation_are_rejected(tmp_path, monkeypatch):
    import launch_rl_continuation as launcher

    accepted(tmp_path)
    args = Namespace(root=tmp_path, validate_only=False)
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(launcher.time.time()) + 100))
    with pytest.raises(RuntimeError, match="lease bound"):
        launcher.main(args)
    (tmp_path / "fresh-contract-rl24-001").mkdir()
    with pytest.raises(RuntimeError, match="existing attempt"):
        launcher.main(args)
