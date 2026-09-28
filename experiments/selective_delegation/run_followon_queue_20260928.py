"""Run independent work after a whole queue settles, including explicit failures.

The existing executor, caps and scientific ownership checks remain unchanged.
Unlike waiting for one final checkpoint owner, this handoff also works when the
last job skips before acquiring the GPU. No running owner is stopped or bypassed.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import psutil

GENERIC = Path(
    "/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/"
    "source-queue-handoff-generic-004/run_independent_queue.py"
)
spec = importlib.util.spec_from_file_location("accepted_generic_executor", GENERIC)
generic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generic)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def active(invocation):
    try:
        process = psutil.Process(invocation["pid"])
        # Invocation is saved just after startup. A reused PID is not this owner.
        return (
            abs(process.create_time() - invocation["started"]) < 30
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def predecessor_state(root, process_active=active):
    receipt = read(root / "ACCEPTED.json")
    invocation = read(root / "queue/INVOCATION.json")
    if not receipt or not invocation:
        return dict(settled=False, reason="predecessor has not started")
    if receipt.get("status") != "accepted" or not receipt.get("jobs"):
        raise ValueError("invalid predecessor acceptance")
    if process_active(invocation):
        return dict(settled=False, reason="predecessor queue still active")
    for job in receipt["jobs"]:
        output = job.get("output")
        if (
            output
            and list(Path(output).glob("OWNER-*.json"))
            and not generic.resource_released(output)
        ):
            return dict(settled=False, reason="predecessor scientific owner unresolved")
    records = [read(root / "queue" / (job["name"] + ".json")) for job in receipt["jobs"]]
    return dict(
        settled=True,
        jobs=len(records),
        exited_jobs=sum(row is not None for row in records),
        successful_jobs=sum(row is not None and row.get("returncode") == 0 for row in records),
        reason="queue process exited and every acquired scientific owner released",
    )


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def main(args):
    receipt = read(args.receipt)
    if receipt.get("status") != "accepted" or not receipt.get("jobs"):
        raise ValueError("finite accepted queue required")
    if hashlib.sha256(GENERIC.read_bytes()).hexdigest() != receipt["executor_sha256"]:
        raise ValueError("accepted executor changed")
    args.output.mkdir(parents=True, exist_ok=True)
    save(
        args.output / "INVOCATION.json",
        dict(
            pid=os.getpid(),
            started=time.time(),
            receipt=str(args.receipt.resolve()),
            receipt_sha256=hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
        ),
    )
    deadline = min(
        time.time() + receipt["maximum_seconds"], int(os.environ["SLURM_JOB_END_TIME"]) - 600
    )
    while not (state := predecessor_state(Path(receipt["predecessor_queue"])))["settled"]:
        if time.time() >= deadline:
            raise TimeoutError("predecessor queue did not settle within cap")
        time.sleep(5)
    save(args.output / "PREDECESSOR-RELEASE.json", dict(observed=time.time(), **state))
    try:
        records = generic.execute_jobs(receipt["jobs"], args.output, deadline)
    except Exception as exc:
        save(args.output / "TERMINAL.json", dict(ended=time.time(), failure=repr(exc)))
        raise
    save(args.output / "TERMINAL.json", dict(ended=time.time(), jobs=len(records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
