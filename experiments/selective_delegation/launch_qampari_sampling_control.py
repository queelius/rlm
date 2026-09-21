"""Conditional32-slot QAMPARI sampling-package control after complete fresh ALF."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released
from qampari_sampling_control import complete

SOURCE = "source-036-qampari-sampling-control"
OUTPUT = "qampari-sampling-control-001"
PREDECESSOR = "alfworld-unseen-001"


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "qampari_sampling_control.py"),
        "--cases",
        str(root / "qampari-inputs-001/cases.jsonl"),
        "--baseline",
        str(root / "qampari-reading-001"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "0.5",
    ]


def predecessor_complete(summary):
    groups = summary.get("groups", {})
    return (
        summary.get("planned_episodes") == 72
        and summary.get("recorded_episodes") == 72
        and summary.get("all_slots_recorded") is True
        and set(groups) == {"flat", "manager_worker", "local_reason"}
        and all(
            g.get("observed") == 24 and g.get("missing_or_unobserved") == 0 for g in groups.values()
        )
    )


def validate(root, decision):
    if (
        decision.get("schema") != "qampari-sampling-control-decision-v1"
        or decision.get("source") != SOURCE
        or decision.get("predecessor") != PREDECESSOR
        or decision.get("status") not in ("proposed", "accepted")
    ):
        raise ValueError("recognized immutable proposal/acceptance required")
    required = {
        f"{SOURCE}/qampari_sampling_control.py",
        f"{SOURCE}/launch_qampari_sampling_control.py",
        f"{OUTPUT}/PLAN.json",
        "qampari-inputs-001/cases.jsonl",
        "qampari-reading-001/PLAN.json",
    }
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("source, prepared PLAN and input hashes required")
    for relative, sha in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != sha:
            raise ValueError("sealed source/input changed: " + relative)
    if decision["status"] == "accepted" and not decision.get("predecessor_plan_sha256"):
        raise ValueError("accepted decision must pin predecessor PLAN")


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "QAMPARI-SAMPLING-CONTROL-DECISION-001.json").read_text())
    validate(root, decision)
    if args.validate_only:
        print(
            json.dumps({"status": decision["status"], "planned_calls": 32, "maximum_seconds": 1800})
        )
        return 0
    if decision["status"] != "accepted":
        raise ValueError("proposal cannot launch; main acceptance required")
    output = root / OUTPUT
    if list(output.glob("OWNER-*.json")) or (output / "calls").exists():
        raise RuntimeError("existing attempt requires review; no implicit retry")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 2400)
    while not released(root / PREDECESSOR):
        if time.time() >= deadline:
            raise RuntimeError("predecessor wait/allocation margin exhausted")
        time.sleep(5)
    prior = root / PREDECESSOR
    if (
        hashlib.sha256((prior / "PLAN.json").read_bytes()).hexdigest()
        != decision["predecessor_plan_sha256"]
    ):
        raise ValueError("fresh ALF predecessor PLAN differs")
    if not predecessor_complete(json.loads((prior / "SUMMARY.json").read_text())):
        raise RuntimeError("fresh ALF incomplete; no silent advancement")
    if time.time() >= deadline:
        raise RuntimeError("insufficient allocation for bounded control")
    cmd = command(root)
    print(json.dumps({"started": time.time(), "command": cmd}), flush=True)
    result = subprocess.run(
        cmd,
        check=False,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    terminal_ok = released(output)
    summary = (
        json.loads((output / "SUMMARY.json").read_text())
        if (output / "SUMMARY.json").exists()
        else {}
    )
    receipt = {
        "returncode": result.returncode,
        "released": terminal_ok,
        "all32observed": complete(summary),
        "ended": time.time(),
    }
    with (output / "QUEUE-RESULT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    return int(bool(result.returncode) or not terminal_ok or not complete(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
