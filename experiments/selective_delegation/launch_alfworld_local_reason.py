"""Proposed local deliberation control after authenticated sufficiency readout release."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-032-alfworld-local-reason"
OUTPUT = "alfworld-local-reason-001"
PREDECESSOR = "sufficiency-readout-positive_only-001"


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "alfworld_local_reason.py"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "1",
    ]


def validate(root, decision):
    if (
        decision.get("schema") != "alfworld-local-reason-decision-v1"
        or decision.get("source") != SOURCE
        or decision.get("predecessor") != PREDECESSOR
        or decision.get("status") not in ("proposed", "accepted")
    ):
        raise ValueError("recognized proposed/accepted local reason decision required")
    required = {
        f"{SOURCE}/{name}"
        for name in (
            "alfworld_local_reason.py",
            "alfworld_closed_loop.py",
            "alfworld_probe.py",
            "alfworld_bridge.py",
            "launch_alfworld_local_reason.py",
        )
    }
    required.update({f"{OUTPUT}/PLAN.json", "alfworld-closed-loop-001/PLAN.json"})
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("collector/native/launcher/input hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("accepted source/input changed: " + relative)


def require_complete(summary):
    cost = summary.get("physical_cost", {})
    if (
        summary.get("planned_variant_attempts") != 128
        or summary.get("planned_pair_attempts") != 64
        or summary.get("returned_valid", 0) + summary.get("returned_protocol_invalid", 0) != 128
        or summary.get("missing_or_inference_unavailable") != 0
        or cost.get("calls") != 128
        or cost.get("failed_calls") != 0
        or cost.get("unknown_usage_calls") != 0
    ):
        raise RuntimeError("sufficiency predecessor incomplete; no silent advancement")


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "ALFWORLD-LOCAL-REASON-DECISION-001.json").read_text())
    validate(root, decision)
    if args.validate_only:
        print(
            json.dumps(
                {"status": decision["status"], "planned_episodes": 16, "maximum_seconds": 3600}
            )
        )
        return 0
    if decision["status"] != "accepted":
        raise ValueError("proposal cannot launch; main acceptance required")
    output = root / OUTPUT
    if list(output.glob("OWNER-*.json")) or any(
        (output / p).exists() for p in ("calls", "episodes")
    ):
        raise RuntimeError("existing scientific attempt requires explicit review")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 4200)
    while not released(root / PREDECESSOR):
        if time.time() >= deadline:
            raise RuntimeError("predecessor wait/allocation bound exhausted")
        time.sleep(5)
    require_complete(json.loads((root / PREDECESSOR / "SUMMARY.json").read_text()))
    if time.time() >= deadline:
        raise RuntimeError("insufficient allocation for bounded control")
    result = subprocess.run(
        command(root),
        check=False,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    terminal_ok = released(output)
    summary = (
        json.loads((output / "SUMMARY.json").read_text())
        if (output / "SUMMARY.json").exists()
        else {}
    )
    complete = summary.get("observed") == 16 and summary.get("missing_or_unobserved") == 0
    receipt = {
        "returncode": result.returncode,
        "released": terminal_ok,
        "all16observed": complete,
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
