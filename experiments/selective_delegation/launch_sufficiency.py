"""Launch the accepted full-context baseline only after authenticated predecessor release."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-024-sufficiency"


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "sufficiency_probe.py"),
        "--cases",
        str(root / "sufficiency-inputs-001/cases.jsonl"),
        "--output",
        str(root / "sufficiency-001"),
        "--hours",
        "0.3333333333333333",
    ]


def child_environment():
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def refuse_existing_owner(root):
    if list((root / "sufficiency-001").glob("OWNER-*.json")):
        raise RuntimeError("existing sufficiency owner requires explicit review")


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "SUFFICIENCY-DECISION-001.json").read_text())
    if decision.get("source") != SOURCE or decision.get("status") not in {"proposed", "accepted"}:
        raise RuntimeError("recognized sealed sufficiency decision required")
    if not args.validate_only and decision["status"] != "accepted":
        raise RuntimeError("accepted decision required; proposal cannot launch")
    if not decision.get("sha256"):
        raise RuntimeError("source/input hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("sealed source/input changed: " + relative)
    refuse_existing_owner(root)
    print(
        json.dumps(
            {"validated": time.time(), "status": decision["status"], "decision": decision["schema"]}
        ),
        flush=True,
    )
    if args.validate_only:
        return 0
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 1800)
    while True:
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release within wait/lease bound")
        if released(root / "training-execution-credit-001"):
            break
        time.sleep(5)
    refuse_existing_owner(root)
    collect = command(root)
    print(json.dumps({"started": time.time(), "command": collect}), flush=True)
    result = subprocess.run(collect, check=False, env=child_environment())
    print(json.dumps({"ended": time.time(), "returncode": result.returncode}), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
