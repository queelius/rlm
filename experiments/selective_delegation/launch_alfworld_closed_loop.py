"""Conditional indexed+feedback ALFWorld qualification after authenticated Hotpot vote."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-026-alfworld-closed-loop"
OUTPUT = "alfworld-closed-loop-001"
PREDECESSOR = "hotpot-fresh-direct-vote-001"


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "alfworld_closed_loop.py"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "1",
    ]


def validate(root, decision):
    if (
        decision.get("schema") != "alfworld-closed-loop-decision-v1"
        or decision.get("source") != SOURCE
        or decision.get("predecessor") != PREDECESSOR
        or decision.get("status") not in ("proposed", "accepted")
        or not decision.get("sha256")
    ):
        raise ValueError("recognized proposed/accepted closed-loop decision required")
    required = {
        SOURCE + "/alfworld_closed_loop.py",
        SOURCE + "/alfworld_probe.py",
        SOURCE + "/alfworld_bridge.py",
        SOURCE + "/launch_alfworld_closed_loop.py",
        OUTPUT + "/PLAN.json",
    }
    if not required <= decision["sha256"].keys():
        raise ValueError("collector/native/launcher/input hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("accepted source/input changed: " + relative)


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "ALFWORLD-CLOSED-LOOP-DECISION-001.json").read_text())
    validate(root, decision)
    if args.validate_only:
        print(
            json.dumps(
                {"status": decision["status"], "planned_episodes": 32, "maximum_seconds": 3600}
            )
        )
        return 0
    if decision["status"] != "accepted":
        raise ValueError("proposal cannot launch; main acceptance required")
    output = root / OUTPUT
    if list(output.glob("OWNER*")) or any(
        (output / name).exists() for name in ("calls", "episodes")
    ):
        raise RuntimeError("existing scientific attempt requires explicit review")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 4200)
    while not released(root / PREDECESSOR):
        if time.time() >= deadline:
            raise RuntimeError("predecessor wait/allocation bound exhausted")
        time.sleep(5)
    predecessor_summary = json.loads((root / PREDECESSOR / "SUMMARY.json").read_text())
    if (
        predecessor_summary.get("planned_new_calls") != 512
        or predecessor_summary.get("returned_new_calls") != 512
        or predecessor_summary.get("missing_new_calls") != 0
        or predecessor_summary.get("unavailable_new_calls") != 0
    ):
        raise RuntimeError("predecessor incomplete; no silent advancement")
    if time.time() >= deadline:
        raise RuntimeError("insufficient allocation for bounded qualification")
    # released authenticates PID/create_time and refuses failed/stopped predecessors.
    result = subprocess.run(command(root), check=False)
    terminal_ok = released(output)
    summary = (
        json.loads((output / "SUMMARY.json").read_text())
        if (output / "SUMMARY.json").exists()
        else {}
    )
    complete = all(
        summary.get("groups", {}).get(p, {}).get("observed") == 16
        for p in ("flat", "manager_worker")
    )
    receipt = {
        "returncode": result.returncode,
        "released": terminal_ok,
        "all32observed": complete,
        "ended": time.time(),
        "status": "complete"
        if not result.returncode and terminal_ok and complete
        else "failed_or_incomplete",
    }
    with (output / "QUEUE-RESULT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return int(bool(result.returncode) or not terminal_ok or not complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
