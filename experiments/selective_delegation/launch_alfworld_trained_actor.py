"""Proposed fixed-checkpoint actor readout after clean SFT and 035 baseline release."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-042b-alfworld-trained-actor"
OUTPUT = "alfworld-trained-actor-001"
DECISION = "ALFWORLD-TRAINED-ACTOR-DECISION-002.json"
SFT = "alfworld-action-sft-002"
BASELINE = "alfworld-unseen-001"


def read(path):
    return json.loads(path.read_text())


def prerequisites_complete(root):
    if not released(root / SFT) or not released(root / BASELINE):
        return False
    terminal = read(next((root / SFT).glob("TERMINAL-*.json")))
    checkpoint = root / SFT / "checkpoint-0033" / "COMMIT.json"
    summary = read(root / BASELINE / "SUMMARY.json")
    return (
        terminal.get("complete") is True
        and checkpoint.exists()
        and summary.get("planned_episodes") == 72
        and summary.get("recorded_episodes") == 72
        and summary.get("all_slots_recorded") is True
    )


def command(root, prepare_only=False):
    value = [
        TRAIN_PYTHON,
        str(root / SOURCE / "alfworld_trained_actor.py"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "1.5",
    ]
    return value + (["--prepare-only"] if prepare_only else [])


def validate(root, decision):
    if (
        decision.get("schema") != "alfworld-trained-actor-decision-v1"
        or decision.get("source") != SOURCE
        or decision.get("status") not in ("proposed", "accepted")
    ):
        raise ValueError("recognized proposed/accepted trained-actor decision required")
    required = {
        f"{SOURCE}/SOURCE.json",
        f"{SOURCE}/alfworld_trained_actor.py",
        f"{SOURCE}/launch_alfworld_trained_actor.py",
        f"{SOURCE}/test_alfworld_trained_actor.py",
        f"{SOURCE}/test_alfworld_trained_actor_peft.py",
        f"{SFT}/PLAN.json",
        f"{BASELINE}/PLAN.json",
        "source-041b-alfworld-one-epoch-sft/train_alfworld_sft.py",
        "source-035-alfworld-unseen/alfworld_unseen.py",
    }
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("source, frozen baseline, and fixed-SFT contract hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed source/input changed: " + relative)
    source = read(root / SOURCE / "SOURCE.json")
    for name, expected in source.get("files", {}).items():
        if hashlib.sha256((root / SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed source dependency changed: " + name)


def complete(summary):
    groups = summary.get("groups", {})
    return (
        summary.get("planned_episodes") == 48
        and summary.get("recorded_episodes") == 48
        and set(groups) == {"flat", "manager_worker"}
        and all(
            group.get("observed") == 24 and group.get("missing_or_unobserved") == 0
            for group in groups.values()
        )
    )


def main(args):
    root = args.root.resolve()
    decision = read(root / DECISION)
    validate(root, decision)
    if args.validate_only:
        print(json.dumps({"status": decision["status"], "planned_episodes": 48, "hours": 1.5}))
        return 0
    if decision["status"] != "accepted":
        raise ValueError("proposal cannot launch; main acceptance required")
    output = root / OUTPUT
    if list(output.glob("OWNER-*.json")) or any(
        (output / name).exists() for name in ("calls", "episodes")
    ):
        raise RuntimeError("existing scientific attempt requires explicit review")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 9600)
    while not prerequisites_complete(root):
        if time.time() >= deadline:
            raise RuntimeError("fixed SFT or baseline release/allocation bound exhausted")
        time.sleep(5)
    preflight = subprocess.run(command(root, prepare_only=True), check=False)
    if preflight.returncode:
        raise RuntimeError("fixed checkpoint preflight failed")
    result = subprocess.run(command(root), check=False)
    released_ok = released(output)
    summary = read(output / "SUMMARY.json") if (output / "SUMMARY.json").exists() else {}
    success = not result.returncode and released_ok and complete(summary)
    with (output / "QUEUE-RESULT.json").open("x") as stream:
        json.dump(
            {
                "returncode": result.returncode,
                "released": released_ok,
                "all48observed": complete(summary),
                "ended": time.time(),
                "status": "complete" if success else "failed_or_incomplete",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return int(not success)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
