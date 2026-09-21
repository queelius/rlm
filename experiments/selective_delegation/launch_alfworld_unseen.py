"""Proposed unchanged-policy unseen ALFWorld readout after canonical sufficiency."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released

SOURCE = "source-035-alfworld-unseen"
OUTPUT = "alfworld-unseen-001"
PREDECESSOR = "sufficiency-canonical-positive_only-001"


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "alfworld_unseen.py"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "2",
    ]


def validate(root, decision):
    if (
        decision.get("schema") != "alfworld-unseen-decision-v1"
        or decision.get("source") != SOURCE
        or decision.get("status") not in ("proposed", "accepted")
        or decision.get("predecessor") != PREDECESSOR
    ):
        raise ValueError("recognized proposed/accepted unseen decision required")
    required = {
        f"{SOURCE}/{name}"
        for name in (
            "alfworld_unseen.py",
            "alfworld_local_reason.py",
            "alfworld_closed_loop.py",
            "alfworld_probe.py",
            "alfworld_bridge.py",
            "launch_alfworld_unseen.py",
        )
    }
    required.update({f"{OUTPUT}/PLAN.json", "alfworld-unseen-inputs-001/MANIFEST.json"})
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("source/input hashes required")
    for relative, expected in decision["sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed source/input changed: " + relative)


def require_predecessor(summary):
    if (
        summary.get("planned_variant_attempts") != 128
        or summary.get("missing_or_inference_unavailable") != 0
        or summary.get("returned_valid", 0) + summary.get("returned_protocol_invalid", 0) != 128
        or summary.get("physical_cost", {}).get("calls") != 128
        or summary.get("physical_cost", {}).get("failed_calls") != 0
    ):
        raise RuntimeError("canonical sufficiency predecessor incomplete")


def complete(summary):
    return (
        summary.get("planned_episodes") == 72
        and summary.get("recorded_episodes") == 72
        and summary.get("all_slots_recorded") is True
        and set(summary.get("groups", {})) == {"flat", "manager_worker", "local_reason"}
        and all(
            g.get("planned") == 24
            and g.get("observed") == 24
            and g.get("missing_or_unobserved") == 0
            for g in summary["groups"].values()
        )
    )


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / "ALFWORLD-UNSEEN-DECISION-001.json").read_text())
    validate(root, decision)
    if args.validate_only:
        print(
            json.dumps(
                {"status": decision["status"], "planned_episodes": 72, "maximum_seconds": 7200}
            )
        )
        return 0
    if decision["status"] != "accepted":
        raise ValueError("proposal cannot launch; main acceptance required")
    output = root / OUTPUT
    if list(output.glob("OWNER-*.json")) or any(
        (output / p).exists() for p in ("calls", "episodes")
    ):
        raise RuntimeError("existing attempt requires explicit review")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 7800)
    while not released(root / PREDECESSOR):
        if time.time() >= deadline:
            raise RuntimeError("predecessor wait/allocation bound exhausted")
        time.sleep(5)
    require_predecessor(json.loads((root / PREDECESSOR / "SUMMARY.json").read_text()))
    if time.time() >= deadline:
        raise RuntimeError("insufficient allocation for bounded readout")
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
    released_ok = released(output)
    summary = (
        json.loads((output / "SUMMARY.json").read_text())
        if (output / "SUMMARY.json").exists()
        else {}
    )
    success = not result.returncode and released_ok and complete(summary)
    with (output / "QUEUE-RESULT.json").open("x") as stream:
        json.dump(
            {
                "returncode": result.returncode,
                "released": released_ok,
                "all72observed": complete(summary),
                "ended": time.time(),
                "status": "complete" if success else "failed_or_incomplete",
            },
            stream,
            indent=2,
        )
    return int(not success)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
