"""Accepted QAMPARI screen after authenticated and complete TRAIN direct-control release."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-028-qampari"


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "qampari_probe.py"),
        "--cases",
        str(root / "qampari-inputs-001/cases.jsonl"),
        "--output",
        str(root / "qampari-reading-001"),
        "--hours",
        "1",
    ]


def complete_release(output):
    if not released(output):
        return False
    summary = json.loads((output / "SUMMARY.json").read_text())
    expected = {
        "planned_unique_direct_calls": 80,
        "recorded_unique_direct_calls": 80,
        "missing_unique_direct_calls": 0,
        "planned_logical_slots": 320,
        "mapped_logical_slots": 320,
    }
    if (
        any(summary.get(k) != v for k, v in expected.items())
        or summary.get("status_counts", {}).get("scored") != 80
    ):
        raise RuntimeError("TRAIN direct predecessor incomplete; inspect before advance")
    return True


def refuse_existing_owner(root):
    if list((root / "qampari-reading-001").glob("OWNER-*.json")):
        raise RuntimeError("existing QAMPARI owner requires explicit review")


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "QAMPARI-DECISION-001.json").read_text())
    if decision.get("source") != SOURCE or decision.get("status") not in {"proposed", "accepted"}:
        raise RuntimeError("recognized sealed decision required")
    if not args.validate_only and decision["status"] != "accepted":
        raise RuntimeError("accepted decision required; proposal cannot launch")
    if not decision.get("sha256"):
        raise RuntimeError("source and input hash map required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("sealed source/input changed: " + relative)
    refuse_existing_owner(root)
    print(
        json.dumps(
            {"validated": time.time(), "status": decision["status"], "schema": decision["schema"]}
        ),
        flush=True,
    )
    if args.validate_only:
        return 0
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 4200)
    while True:
        if time.time() >= deadline:
            raise RuntimeError("predecessor did not release within wait/lease bound")
        if complete_release(root / "training-direct-control-002"):
            break
        time.sleep(5)
    refuse_existing_owner(root)
    cmd = command(root)
    print(json.dumps({"started": time.time(), "command": cmd}), flush=True)
    result = subprocess.run(
        cmd,
        check=False,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    print(json.dumps({"ended": time.time(), "returncode": result.returncode}), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
