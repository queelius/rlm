"""Conditional launcher for the proposed 80-call TRAIN direct control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-029-training-direct-control-repair"
OUTPUT = "training-direct-control-002"
PREDECESSOR = "alfworld-closed-loop-001"


def complete_predecessor(summary: dict) -> bool:
    return all(
        summary.get("groups", {}).get(policy, {}).get("observed") == 16
        for policy in ("flat", "manager_worker")
    )


def command(root: Path) -> list[str]:
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "training_direct_control.py"),
        "--root",
        str(root),
        "--output",
        str(root / OUTPUT),
        "--hours",
        str(1 / 3),
    ]


def validate(root: Path, decision: dict) -> None:
    if (
        decision.get("schema") != "training-direct-control-decision-v2"
        or decision.get("source") != SOURCE
        or decision.get("predecessor") != PREDECESSOR
        or not decision.get("sha256")
    ):
        raise RuntimeError("recognized training-direct decision required")
    if decision.get("status") not in {"proposed", "accepted"}:
        raise RuntimeError("decision must be proposed or accepted")
    for relative, expected in decision.get("sha256", {}).items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("sealed dependency changed: " + relative)


def main(args) -> int:
    root = args.root.resolve()
    decision = json.loads((root / "TRAINING-DIRECT-CONTROL-DECISION-002.json").read_text())
    validate(root, decision)
    if args.validate_only:
        print(
            json.dumps(
                {"validated": time.time(), "status": decision["status"], "maximum_calls": 80}
            )
        )
        return 0
    if decision["status"] != "accepted":
        raise RuntimeError("proposed decision refuses GPU launch")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 1800)
    while not released(root / PREDECESSOR):
        if time.time() >= deadline:
            raise RuntimeError("predecessor wait or allocation bound exhausted")
        time.sleep(5)
    if not complete_predecessor(json.loads((root / PREDECESSOR / "SUMMARY.json").read_text())):
        raise RuntimeError("predecessor must complete all32 original episodes")
    output = root / OUTPUT
    if list(output.glob("OWNER-*.json")) or any(
        any((output / name).glob("*.json")) for name in ("calls", "episodes")
    ):
        raise RuntimeError("existing collector receipts require explicit review")
    return subprocess.run(command(root), check=False, env=os.environ).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
