"""Run the accepted plan-only replay after the frozen-prefix screen owner releases."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TRAIN_PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / "source-018/plan_only_probe.py"),
        "--root",
        str(root),
        "--output",
        str(root / "plan-only-001"),
        "--hours",
        ".3333333333",
    ]


def refuse_existing_owner(root):
    if list((root / "plan-only-001").glob("OWNER-*.json")):
        raise RuntimeError("existing plan-only attempt requires explicit resume review")


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "PLAN-ONLY-DECISION-001.json").read_text())
    if decision["status"] != "accepted" or decision["source"] != "source-018":
        raise RuntimeError("accepted sealed plan-only decision required")
    if not decision["sha256"]:
        raise RuntimeError("accepted source/input hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("accepted source/input changed: " + relative)
    print(json.dumps({"validated": time.time(), "decision": decision["schema"]}), flush=True)
    refuse_existing_owner(root)
    if args.validate_only:
        return 0
    sys.path.insert(0, str(root / "source-018"))
    from launch_rl_diagnostics import released

    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 1800)
    while True:
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release within accepted wait/lease bound")
        if released(root / "next-question-screen-001"):
            break
        time.sleep(5)
    refuse_existing_owner(root)
    collect = command(root)
    print(json.dumps({"started": time.time(), "command": collect}), flush=True)
    result = subprocess.run(collect, check=False)
    print(json.dumps({"ended": time.time(), "returncode": result.returncode}), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
