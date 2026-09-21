"""Conditional source044 native TextCraft qualification after accepted042b release."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from launch_rl_diagnostics import TRAIN_PYTHON, released
from prepare_textcraft import save, sha

SOURCE = "source-044-textcraft-pilot"
OUTPUT = "textcraft-pilot-001"
PREDECESSOR = "alfworld-trained-actor-001"
PREDECESSOR_SOURCE = "source-042b-alfworld-trained-actor"
PREDECESSOR_DECISION = "ALFWORLD-TRAINED-ACTOR-DECISION-002.json"
DECISION = "TEXTCRAFT-PILOT-DECISION-001.json"


def command(root):
    return [
        TRAIN_PYTHON,
        str(root / SOURCE / "eval_textcraft.py"),
        "--prepared",
        str(root / "textcraft-inputs-001"),
        "--output",
        str(root / OUTPUT),
        "--hours",
        "1",
    ]


def predecessor_complete(summary):
    groups = summary.get("groups", {})
    return (
        summary.get("planned_episodes") == 48
        and summary.get("recorded_episodes") == 48
        and set(groups) == {"flat", "manager_worker"}
        and all(
            g.get("observed") == 24 and g.get("missing_or_unobserved") == 0 for g in groups.values()
        )
    )


def complete(summary):
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


def validate(root, decision):
    if (
        decision.get("schema") != "textcraft-pilot-decision-v1"
        or decision.get("source") != SOURCE
        or decision.get("output") != OUTPUT
        or decision.get("predecessor") != PREDECESSOR
        or decision.get("status") not in ("proposed", "accepted")
    ):
        raise ValueError("recognized proposed/accepted immutable decision required")
    required = {
        f"{SOURCE}/eval_textcraft.py",
        f"{SOURCE}/launch_textcraft.py",
        f"{OUTPUT}/PLAN.json",
        "textcraft-inputs-001/MANIFEST.json",
        PREDECESSOR_DECISION,
        f"{PREDECESSOR_SOURCE}/alfworld_trained_actor.py",
    }
    if not required <= decision.get("sha256", {}).keys():
        raise ValueError("source/input/PLAN/accepted predecessor hashes required")
    for relative, digest in decision["sha256"].items():
        if sha(root / relative) != digest:
            raise ValueError("accepted source/input changed: " + relative)
    prior = json.loads((root / PREDECESSOR_DECISION).read_text())
    if prior["status"] != "accepted" or prior["source"] != PREDECESSOR_SOURCE:
        raise ValueError("accepted042b predecessor required")


def main(args):
    root = args.root.resolve()
    decision = json.loads((root / DECISION).read_text())
    validate(root, decision)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": decision["status"],
                    "planned_episodes": 32,
                    "max_native_calls": 3072,
                    "maximum_seconds": 3600,
                }
            )
        )
        return 0
    if decision["status"] != "accepted":
        raise ValueError("proposal cannot launch; main acceptance required")
    output, prior = root / OUTPUT, root / PREDECESSOR
    if list(output.glob("OWNER-*.json")) or any(
        (output / d).exists() for d in ("calls", "episodes")
    ):
        raise ValueError("existing attempt; no implicit retry")
    deadline = min(time.time() + 21600, int(os.environ["SLURM_JOB_END_TIME"]) - 4200)
    while not released(prior):
        if time.time() >= deadline:
            raise RuntimeError("042b wait/allocation bound exhausted")
        time.sleep(5)
    owner = json.loads(next(prior.glob("OWNER-*.json")).read_text())
    if (
        Path(owner["source"]).resolve() != root / PREDECESSOR_SOURCE / "alfworld_trained_actor.py"
        or not predecessor_complete(json.loads((prior / "SUMMARY.json").read_text()))
    ):
        raise ValueError("042b authenticated complete48 required; no silent advancement")
    if time.time() >= deadline:
        raise RuntimeError("insufficient allocation for bounded one-hour pilot")
    cmd = command(root)
    print(json.dumps({"command": cmd, "started": time.time()}), flush=True)
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
    released_ok = released(output)
    summary = (
        json.loads((output / "SUMMARY.json").read_text())
        if (output / "SUMMARY.json").exists()
        else {}
    )
    success = result.returncode == 0 and released_ok and complete(summary)
    save(
        output / "QUEUE-RESULT.json",
        {
            "returncode": result.returncode,
            "released": released_ok,
            "complete32": complete(summary),
            "success": success,
            "ended": time.time(),
        },
    )
    return int(not success)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    raise SystemExit(main(parser.parse_args()))
