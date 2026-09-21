"""Conditional accepted RL16→24 continuation, then fixed matched checkpoint24 evaluation."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commands(root):
    source = root / "source-020"
    helper = [
        "--helper-contract",
        "trained_helper",
        "--helper-adapter",
        str(root / "helper-sft-001/checkpoint-0036"),
    ]
    train = [
        TRAIN_PYTHON,
        str(source / "rl_planner.py"),
        "--cases",
        str(root / "inputs-001/cases.jsonl"),
        "--adapter",
        str(root / "planner-sft-001/checkpoint-0048"),
        "--continue-from",
        str(root / "rl-fullpass-001/checkpoint-0016"),
        "--output",
        str(root / "rl-continuation-001"),
        "--parent-schedule",
        "consecutive",
        "--updates",
        "24",
        "--hours",
        "1.5",
        *helper,
    ]
    evaluate = [
        TRAIN_PYTHON,
        str(source / "eval_planner.py"),
        "--cases",
        str(root / "fresh-dev-inputs-003/cases.jsonl"),
        "--adapter",
        str(root / "rl-continuation-001/checkpoint-0024"),
        "--output",
        str(root / "fresh-contract-rl24-001"),
        "--split",
        "development",
        "--start",
        "0",
        "--limit",
        "64",
        "--repeats",
        "2",
        "--execution",
        "isolated",
        "--trained-condition",
        "rl",
        "--trained-only",
        "--hours",
        "1",
        *helper,
    ]
    return train, evaluate


def refuse_existing_attempt(root):
    for name in ("rl-continuation-001", "fresh-contract-rl24-001"):
        if (root / name).exists():
            raise RuntimeError("existing attempt requires explicit resume review: " + name)


def require_endpoint(root):
    output = root / "rl-continuation-001"
    if not released(output):
        raise RuntimeError("training owner has not authenticated release")
    terminal = json.loads(next(output.glob("TERMINAL-*.json")).read_text())
    checkpoint = output / "checkpoint-0024"
    if (
        terminal.get("state") != "completed_updates"
        or terminal.get("optimizer_steps") != 24
        or terminal.get("additional_optimizer_steps") != 8
        or not (checkpoint / "COMMIT.json").exists()
    ):
        raise RuntimeError("successful exact committed checkpoint24 required; no substitution")
    commit = json.loads((checkpoint / "COMMIT.json").read_text())
    required = {
        "STATE.json",
        "optimizer.pt",
        "rng.pt",
        "adapter_model.safetensors",
        "adapter_config.json",
    }
    if commit.get("step") != 24 or not required <= commit.get("files", {}).keys():
        raise RuntimeError("incomplete checkpoint24 commit")
    for name, expected in commit["files"].items():
        if sha(checkpoint / name) != expected:
            raise RuntimeError("checkpoint changed: " + name)
    state = json.loads((checkpoint / "STATE.json").read_text())
    plan = json.loads((output / "PLAN.json").read_text())
    ancestry, helper = plan.get("continuation", {}), plan.get("helper_contract", {})
    if (
        state.get("step") != 24
        or state.get("cursor") != 0
        or state.get("next_update") != 25
        or plan.get("updates") != 24
        or ancestry.get("starting_step") != 16
        or ancestry.get("additional_updates") != 8
        or ancestry.get("ancestor_checkpoint") != str(root / "rl-fullpass-001/checkpoint-0016")
        or helper.get("mode") != "trained_helper"
        or helper.get("adapter") != str(root / "helper-sft-001/checkpoint-0036")
        or state.get("helper_contract") != helper
    ):
        raise RuntimeError("checkpoint24 continuation/helper identity differs")
    return checkpoint


def finish(root, status, **details):
    output = root / "rl-continuation-001"
    output.mkdir(exist_ok=True)
    receipt = {"status": status, "ended": time.time(), **details}
    with (output / "QUEUE-RESULT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "RL-CONTINUATION-DECISION-001.json").read_text())
    if decision.get("status") != "accepted" or decision.get("source") != "source-020":
        raise RuntimeError("accepted sealed RL continuation decision required")
    if not decision.get("sha256"):
        raise RuntimeError("accepted source/input hashes required")
    for relative, expected in decision["sha256"].items():
        if sha(root / relative) != expected:
            raise RuntimeError("accepted source/input changed: " + relative)
    print(json.dumps({"validated": time.time(), "decision": decision["schema"]}), flush=True)
    refuse_existing_attempt(root)
    if args.validate_only:
        return 0
    # Reserve both declared job caps and the normal allocation margin before starting.
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 9600)
    while True:
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release within accepted wait/lease bound")
        if released(root / "hotpot-direct-adapted-001"):
            break
        time.sleep(5)
    refuse_existing_attempt(root)
    train, evaluate = commands(root)
    print(json.dumps({"started": time.time(), "command": train}), flush=True)
    result = subprocess.run(train, check=False)
    if result.returncode:
        finish(root, "training_failed_or_capped_no_evaluation", returncode=result.returncode)
        return 1
    try:
        checkpoint = require_endpoint(root)
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        finish(root, "endpoint_rejected_no_evaluation", error=str(exc))
        return 1
    print(json.dumps({"started": time.time(), "command": evaluate}), flush=True)
    result = subprocess.run(evaluate, check=False)
    finish(root, "evaluation_returned", checkpoint=str(checkpoint), returncode=result.returncode)
    return int(result.returncode != 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
