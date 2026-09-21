"""Conditional 046 readout after authenticated 044 completion; proposed until accepted."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released
from launch_sufficiency_heldout import complete, read, zero_skip_matches

SOURCE = "source-046-sufficiency-compositional-readout"
OUTPUT = "sufficiency-compositional-readout-001"
PREDECESSOR = "textcraft-pilot-001"
DECISION = "SUFFICIENCY-COMPOSITIONAL-READOUT-DECISION-001.json"
RESOLVED = "SUFFICIENCY-COMPOSITIONAL-READOUT-RESOLVED-001.json"


def predecessor_complete(summary):
    groups = summary.get("groups", {})
    return (
        summary.get("planned_episodes") == 32
        and summary.get("recorded_episodes") == 32
        and set(groups) == {"flat", "recursive"}
        and all(
            g.get("observed") == 16 and g.get("missing_or_unknown") == 0 for g in groups.values()
        )
        and not summary.get("unresolved_starts")
    )


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "eval_sufficiency_compositional.py"),
        "--cases",
        str(root / "sufficiency-compositional-inputs-001/cases.jsonl"),
        "--warm-adapter",
        str(root / "sufficiency-sft-joint-001/checkpoint-0032"),
        "--rl-output",
        str(root / "sufficiency-rl-001"),
        "--sft-output",
        str(root / "sufficiency-extra-sft-001"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "1",
    ]


def validate(root, decision):
    if (
        decision.get("source") != SOURCE
        or decision.get("output") != OUTPUT
        or decision.get("predecessor") != PREDECESSOR
        or decision.get("status") not in ("proposed", "accepted")
    ):
        raise ValueError("recognized conditional decision required")
    required = {
        f"{SOURCE}/eval_sufficiency_compositional.py",
        f"{SOURCE}/eval_sufficiency_heldout.py",
        f"{SOURCE}/launch_sufficiency_compositional.py",
        f"{SOURCE}/COMPOSITIONAL-PROFILE.json",
        "sufficiency-compositional-inputs-001/cases.jsonl",
        "sufficiency-compositional-inputs-001/MANIFEST.json",
        "sufficiency-rl-001/PLAN.json",
        "source-037-sufficiency-rl/rl_sufficiency.py",
        "source-038-sufficiency-extra-sft/rl_sufficiency.py",
        "sufficiency-sft-joint-001/checkpoint-0032/COMMIT.json",
        f"{PREDECESSOR}/PLAN.json",
    }
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("panel/source/endpoint contract/predecessor identities required")
    for path, expected in decision["sha256"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError("sealed dependency changed: " + path)


def main(args):
    root = args.root.resolve()
    decision = read(root / DECISION)
    validate(root, decision)
    if args.validate_only:
        print(json.dumps({"status": decision["status"], "planned_calls": 384}))
        return 0
    if decision["status"] != "accepted":
        raise ValueError("main acceptance required")
    if list((root / OUTPUT).glob("OWNER-*.json")) or (root / RESOLVED).exists():
        raise ValueError("existing owner/resolution; no retry")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 4200)
    zero = False
    while True:
        if time.time() >= deadline:
            raise RuntimeError("bounded wait/allocation margin exhausted")
        if released(root / PREDECESSOR):
            if not predecessor_complete(read(root / PREDECESSOR / "SUMMARY.json")):
                raise ValueError("044 incomplete; no automatic advance")
            skip = root / "SUFFICIENCY-ZERO-UPDATE-SKIP-001.json"
            if skip.exists() and released(root / "sufficiency-rl-001"):
                if not zero_skip_matches(
                    read(skip), read(root / "sufficiency-rl-001/SUMMARY.json")
                ):
                    raise ValueError("zero-update skip unauthenticated")
                zero = True
                break
            if released(root / "sufficiency-extra-sft-001"):
                break
        time.sleep(5)
    cmd = command(root) + (["--prepare-only"] if zero else [])
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
    if result.returncode:
        raise RuntimeError("readout failed; no downstream advance")
    if zero:
        receipt = read(root / OUTPUT / "SKIPPED.json")
        if receipt["endpoint"]["step"] != 0 or receipt["scientific_readout"]:
            raise ValueError("reader skip unauthenticated")
    elif not released(root / OUTPUT) or not complete(read(root / OUTPUT / "SUMMARY.json")):
        raise ValueError("readout incomplete")
    with (root / RESOLVED).open("x") as stream:
        json.dump(
            {
                "chain_resolved": True,
                "scientific_readout": not zero,
                "output": OUTPUT,
                "reason": "zero_real_updates" if zero else "complete384",
                "ended": time.time(),
                "decision_sha256": hashlib.sha256((root / DECISION).read_bytes()).hexdigest(),
            },
            stream,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
