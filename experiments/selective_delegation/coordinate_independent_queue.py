"""Serialize queue004 after authenticated queue003 exit, not scientific success."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import run_independent_queue as queue

PID, CREATED = 62417, 1790064194.82
DRIVER_SHA = "35d1c12376dd29513960303df97fee109aa353c7f757175d948f680b00fcd6c8"
PINS = {
    "independent-training-queue-003/INVOCATION.json": (
        "051562960521b1b729310f839c1baf94e2bd38e051deea9ebf0191d5c4649b93"
    ),
    "INDEPENDENT-TRAINING-QUEUE-003A.json": (
        "61ec8b7f1173f3af93737183ed80b647f0db553b817a38fc0b0ce3de5962c5ec"
    ),
}


def ready(pid, created, outputs):
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - created) < 0.01 and process.status() != psutil.STATUS_ZOMBIE:
            return False
    except psutil.NoSuchProcess:
        pass
    return all(
        not list(Path(output).glob("OWNER-*.json")) or queue.resource_released(output)
        for output in outputs
    )


def main(args):
    pins = {str(args.root / name): value for name, value in PINS.items()}
    pins[str(Path(queue.__file__))] = DRIVER_SHA
    pins[str(args.receipt)] = hashlib.sha256(args.receipt.read_bytes()).hexdigest()
    queue.validate_pins(pins)
    previous = json.loads((args.root / "INDEPENDENT-TRAINING-QUEUE-003A.json").read_text())
    invocation = json.loads(
        (args.root / "independent-training-queue-003/INVOCATION.json").read_text()
    )
    outputs = [Path(job["output"]) for job in previous["jobs"]]
    receipt = json.loads(args.receipt.read_text())
    if (
        invocation["pid"] != PID
        or receipt.get("status") != "accepted"
        or not receipt.get("jobs")
        or Path(receipt["predecessor"]) != outputs[0]
        or not list(outputs[0].glob("OWNER-*.json"))
    ):
        raise ValueError("queue identity, accepted jobs or existing058 predecessor differs")
    if args.validate_only:
        print(json.dumps({"validated": True, "ready": ready(PID, CREATED, outputs), "pins": pins}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(
        pid=os.getpid(),
        create_time=psutil.Process().create_time(),
        started=time.time(),
        previous_pid=PID,
        previous_create_time=CREATED,
        pins=pins,
        outputs=list(map(str, outputs)),
        failure=None,
        handed_off=False,
    )
    deadline = min(time.time() + 10800, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    with (args.output / "GATE-INVOCATION.json").open("x") as stream:
        json.dump({**record, "wait_deadline": deadline}, stream, indent=2)
    try:
        with (args.output / "GATE.log").open("x") as log:
            log.write("Waiting for authenticated queue003 exit and extant owner release.\n")
            log.flush()
            while not ready(PID, CREATED, outputs):
                if time.time() >= deadline:
                    raise TimeoutError("queue003 handoff wait exhausted; no jobs launched")
                time.sleep(10)
            if time.time() >= deadline:
                raise TimeoutError("allocation/wait deadline reached; no jobs launched")
            queue.validate_pins(pins)
            record["handed_off"] = True
            log.write("Queue003 exited; all extant owners released. Calling unchanged054.main.\n")
            log.flush()
            queue.main(args)
    except BaseException as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (args.output / "GATE-RESULT.json").open("x") as stream:
            json.dump({**record, "ended": time.time()}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    main(parser.parse_args())
