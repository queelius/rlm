"""Queue the two prepared TRAIN collectors after the bounded restart comparison."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
HERE = Path(__file__).resolve().parent
PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)
SUPERVISOR_PYTHON = "/project/alex_phd/envs/prime-rl-5990b1b/bin/python"


def main():
    output = ROOT / "rl-collect-queue-20260928-001"
    if output.exists():
        raise FileExistsError(output)
    jobs = []
    for mode in ("raw", "binder"):
        target = ROOT / "textcraft-rl-assist-20260928-001" / mode / "collect-0001"
        path = target / "PLAN.json"
        plan = json.loads(path.read_text())
        if list(target.glob("OWNER-*.json")) or (target / "calls").exists():
            raise ValueError("collection already attempted: " + str(target))
        pins = dict(plan["source_sha256"])
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        jobs.append(
            dict(
                name="rl-assist-20260928-" + mode + "-collect-0001",
                argv=[
                    PYTHON,
                    str(HERE / "rl_resume_20260928/collect.py"),
                    "--mode",
                    mode,
                    "--output",
                    str(target),
                ],
                output=str(target),
                cap_seconds=plan["budget_seconds"] + 300,
                pins=pins,
            )
        )
    receipt = dict(
        status="accepted",
        question="Does observed-recipe assistance improve reward diversity and useful RL credit?",
        authorized_by="User's September 28 autonomous research instruction",
        predecessor=str(ROOT / "textcraft-breadth-p06-w50-soriginal-binder-001"),
        maximum_seconds=21600,
        jobs=jobs,
        expected_collection_hours="1 to 1.5, plus wait for earlier accepted jobs",
        queue_bound_hours=6,
        current_allocation_end=os.environ["SLURM_JOB_END_TIME"],
        distinction="TRAIN collections, no optimizer updates accepted by this receipt",
    )
    output.mkdir()
    with (output / "ACCEPTED.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    (output / "launch_source.py").write_bytes(Path(__file__).read_bytes())
    argv = [
        SUPERVISOR_PYTHON,
        str(ROOT / "source-queue-handoff-generic-004/run_independent_queue.py"),
        "--receipt",
        str(output / "ACCEPTED.json"),
        "--output",
        str(output / "queue"),
    ]
    with (output / "supervisor.log").open("x") as stream:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with (output / "LAUNCH.json").open("x") as stream:
        json.dump(dict(pid=child.pid, started=time.time(), argv=argv), stream, indent=2)
    print(json.dumps(dict(pid=child.pid, receipt=str(output / "ACCEPTED.json"))))


if __name__ == "__main__":
    main()
