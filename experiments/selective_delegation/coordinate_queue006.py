"""CPU gate: queue005 release before the accepted one-hour TRAIN063 readout."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import run_independent_queue as queue
from coordinate_independent_queue import ready

PID, CREATED = 101655, 1790069650.22
DRIVER_SHA = "35d1c12376dd29513960303df97fee109aa353c7f757175d948f680b00fcd6c8"
READY_SHA = "46f412a2eab6fad59d964d5ed070ce49bd62f62fbf059b3aec9dae602e9d823c"
PINS = {
    "independent-training-queue-005/INVOCATION.json": (
        "0a7c5e9f37cec6afdee5a7ff36fbba7dc2d14c7c1354db9a8d1ad99add27a20f"
    ),
    "INDEPENDENT-TRAINING-QUEUE-005.json": (
        "930ccb4f6c394c40a0b43dbdc728f4ad611e8a786800db0a96864c12390f6a2e"
    ),
}


def main(args):
    pins = {str(args.root / name): digest for name, digest in PINS.items()}
    pins[str(Path(queue.__file__))] = DRIVER_SHA
    pins[str(Path(__file__).with_name("coordinate_independent_queue.py"))] = READY_SHA
    pins[str(args.receipt)] = hashlib.sha256(args.receipt.read_bytes()).hexdigest()
    queue.validate_pins(pins)
    previous = json.loads((args.root / "INDEPENDENT-TRAINING-QUEUE-005.json").read_text())
    invocation = json.loads(
        (args.root / "independent-training-queue-005/INVOCATION.json").read_text()
    )
    outputs = [Path(job["output"]) for job in previous["jobs"]]
    receipt = json.loads(args.receipt.read_text())
    if (
        invocation["pid"] != PID
        or receipt.get("status") != "accepted"
        or len(receipt.get("jobs", [])) != 1
        or Path(receipt["predecessor"]) != args.root / "textcraft-procedure-control-001"
        or Path(receipt["predecessor"]) != outputs[0]
        or not list(outputs[0].glob("OWNER-*.json"))
        or receipt["jobs"][0]["cap_seconds"] != 3600
        or Path(receipt["jobs"][0]["output"]) != args.root / "textcraft-train-readiness-001"
    ):
        raise ValueError("authenticated005, existing061 predecessor and one-hour063 job required")
    if args.validate_only:
        print(json.dumps(dict(validated=True, ready=ready(PID, CREATED, outputs), pins=pins)))
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
            log.write("Waiting for authenticated queue005 exit and extant061/062 owner release.\n")
            log.flush()
            while not ready(PID, CREATED, outputs):
                if time.time() >= deadline:
                    raise TimeoutError("queue005 handoff wait exhausted; no jobs launched")
                time.sleep(10)
            if time.time() >= deadline:
                raise TimeoutError("allocation/wait deadline reached; no jobs launched")
            queue.validate_pins(pins)
            record["handed_off"] = True
            log.write("Queue005 exited; extant owners released. Calling unchanged054.main.\n")
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
