"""Run the accepted Hotpot direct-adapter control after the plan-only owner releases."""

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
CPU_PYTHON = "/project/alex_phd/envs/prime-rl-5990b1b/bin/python"


def commands(root):
    output = root / "hotpot-direct-adapted-001"
    cases = root / "hotpot-inputs-001/cases.jsonl"
    collect = [
        TRAIN_PYTHON,
        str(root / "source-019/eval_direct_adapted.py"),
        "--panel",
        "hotpot_explorer32",
        "--cases",
        str(cases),
        "--helper-adapter",
        str(root / "helper-sft-001/checkpoint-0036"),
        "--output",
        str(output),
    ]
    analyze = [
        CPU_PYTHON,
        str(root / "source-019/analyze_direct_adapted.py"),
        "--output",
        str(output),
        "--cases",
        str(cases),
        "--report",
        str(root / "analysis-hotpot-direct-adapted-001.json"),
    ]
    return collect, analyze


def refuse_existing_owner(root):
    if list((root / "hotpot-direct-adapted-001").glob("OWNER-*.json")):
        raise RuntimeError("existing Hotpot direct attempt requires explicit resume review")


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "HOTPOT-DIRECT-DECISION-001.json").read_text())
    if decision["status"] != "accepted" or decision["source"] != "source-019":
        raise RuntimeError("accepted sealed Hotpot direct decision required")
    if not decision["sha256"]:
        raise RuntimeError("accepted source/input hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("accepted source/input changed: " + relative)
    print(json.dumps({"validated": time.time(), "decision": decision["schema"]}), flush=True)
    refuse_existing_owner(root)
    if args.validate_only:
        return 0
    sys.path.insert(0, str(root / "source-019"))
    from launch_rl_diagnostics import released

    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 1800)
    while True:
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release within accepted wait/lease bound")
        if released(root / "plan-only-001"):
            break
        time.sleep(5)
    refuse_existing_owner(root)
    collect, analyze = commands(root)
    print(json.dumps({"started": time.time(), "command": collect}), flush=True)
    result = subprocess.run(collect, check=False)
    print(json.dumps({"ended": time.time(), "returncode": result.returncode}), flush=True)
    analysis = subprocess.run(analyze, check=False)
    print(json.dumps({"analyzed": time.time(), "returncode": analysis.returncode}), flush=True)
    return int(result.returncode != 0 or analysis.returncode != 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
