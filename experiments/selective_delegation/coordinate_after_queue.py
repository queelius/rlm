"""Bounded handoff after a pinned supervisor exits and every extant owner releases."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import run_independent_queue as queue
from coordinate_independent_queue import ready


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(args):
    pins = {
        str(args.previous_invocation): args.previous_invocation_sha256,
        str(args.previous_receipt): args.previous_receipt_sha256,
    }
    pins.update(
        {str(Path(fn)): sha(Path(fn)) for fn in (queue.__file__, ready.__code__.co_filename)}
    )
    queue.validate_pins(pins)
    previous = json.loads(args.previous_receipt.read_text())
    invocation = json.loads(args.previous_invocation.read_text())
    if (
        invocation["pid"] != args.previous_pid
        or invocation["receipt_sha256"] != args.previous_receipt_sha256
        or Path(invocation["receipt"]).resolve() != args.previous_receipt.resolve()
        or not 0 <= invocation["started"] - args.previous_create_time <= 60
        or previous.get("status") != "accepted"
        or not previous.get("jobs")
    ):
        raise ValueError("pinned previous accepted queue/invocation identity differs")
    # CPU analysis jobs need not own scientific output/GPU resources. ready() still
    # waits for the authenticated whole supervisor, including those CPU jobs.
    outputs = [Path(job["output"]) for job in previous["jobs"] if job.get("output")]
    if not outputs:
        raise ValueError("predecessor must declare at least one scientific output")
    if args.receipt:
        receipt = json.loads(args.receipt.read_text())
        jobs = receipt.get("jobs")
        if (
            receipt.get("status") != "accepted"
            or not isinstance(jobs, list)
            or not jobs
            or type(receipt.get("maximum_seconds")) is not int
            or receipt["maximum_seconds"] <= 0
            or type(receipt.get("wait_seconds", 14400)) is not int
            or not 0 < receipt.get("wait_seconds", 14400) <= 86400
            or any(
                type(j.get("cap_seconds")) is not int
                or j["cap_seconds"] <= 0
                or not j.get("argv")
                or not isinstance(j.get("pins"), dict)
                for j in jobs
            )
        ):
            raise ValueError("accepted finite capped jobs required")
        pins[str(args.receipt)] = sha(args.receipt)
    elif not args.prepare_only:
        raise ValueError("accepted next receipt required before waiting or executing")
    return outputs, pins


def main(args):
    outputs, pins = validate(args)
    if args.prepare_only:
        print(
            json.dumps(
                dict(
                    validated=True,
                    next_receipt_pending=args.receipt is None,
                    ready=ready(args.previous_pid, args.previous_create_time, outputs),
                    pins=pins,
                    GPU_started=False,
                )
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(
        pid=os.getpid(),
        create_time=psutil.Process().create_time(),
        started=time.time(),
        receipt=str(args.receipt.resolve()),
        receipt_sha256=sha(args.receipt),
        previous_pid=args.previous_pid,
        previous_create_time=args.previous_create_time,
        pins=pins,
        source_sha256=sha(Path(__file__)),
        records=[],
        failure=None,
    )
    lease = int(os.environ["SLURM_JOB_END_TIME"]) - 600
    wait_seconds = json.loads(args.receipt.read_text()).get("wait_seconds", 14400)
    deadline = min(time.time() + wait_seconds, lease)
    with (args.output / "GATE-INVOCATION.json").open("x") as stream:
        json.dump({**record, "wait_deadline": deadline}, stream, indent=2)
    try:
        while not ready(args.previous_pid, args.previous_create_time, outputs):
            if time.time() >= deadline:
                raise TimeoutError("previous queue release wait exhausted; no jobs launched")
            time.sleep(10)
        if time.time() >= deadline:
            raise TimeoutError("wait/lease deadline reached; no jobs launched")
        queue.validate_pins(pins)
        receipt = json.loads(args.receipt.read_text())
        execution_deadline = min(time.time() + receipt["maximum_seconds"], lease)
        record["records"] = queue.execute_jobs(receipt["jobs"], args.output, execution_deadline)
    except BaseException as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (args.output / "GATE-RESULT.json").open("x") as stream:
            json.dump({**record, "ended": time.time()}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-invocation", "previous-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("previous-invocation-sha256", "previous-receipt-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--previous-pid", type=int, required=True)
    parser.add_argument("--previous-create-time", type=float, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    main(parser.parse_args())
