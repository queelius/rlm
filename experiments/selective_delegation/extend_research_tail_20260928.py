"""Insert evidence-driven replications while preserving every accepted scientific job."""

import argparse
import json
import sys
import time
from contextlib import suppress
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "information-first-tail-20260928-002"
PREDECESSOR = "interface-dose-queue-20260928-001"
# Suspend downstream first: releasing the predecessor must not wake its old waiter.
OLD = {
    "payload-mask-queue-20260928-001": (
        "bc784f9b051b56b1bd1a6e690d18f062f207a035d81d522c062b5cef2cb0b846"
    ),
    "information-first-tail-20260928-001": (
        "fac47014954360cc025e977b77d1c6d6f3fd6504ffd9f98eb410558d3fb22344"
    ),
}
WARM = ("fresh-raw-warm-readout-0001", "fresh-binder-warm-readout-0001")


def extend_tail(inherited, mask, teaching, transfer):
    names = [job["name"] for job in inherited]
    if any(names.count(name) != 1 for name in WARM):
        raise ValueError("exactly two inherited warm controls required")
    boundary = names.index("fresh-raw-rl-collect-0001")
    warm = [next(job for job in inherited if job["name"] == name) for name in WARM]
    if any(name in names[:boundary] for name in WARM):
        raise ValueError("warm controls already promoted; inspect new queue state")
    remainder = [job for job in inherited[boundary:] if job["name"] not in WARM]
    jobs = inherited[:boundary] + teaching + warm + transfer + remainder + mask
    new_names = [job["name"] for job in jobs]
    outputs = [job["output"] for job in jobs if job.get("output")]
    if len(new_names) != len(set(new_names)):
        raise ValueError("duplicate job name")
    if len(outputs) != len(set(outputs)):
        raise ValueError("duplicate scientific output")
    return jobs


def augment_pins(jobs, pins):
    for job in jobs:
        for path in job["pins"].keys() & pins.keys():
            if job["pins"][path] != pins[path]:
                raise ValueError("changed inherited pin: " + path)
    return [dict(job, pins={**job["pins"], **pins}) for job in jobs]


def main(args):
    import psutil

    sys.path.insert(0, str(HERE))
    import queue_transfer_20260928 as q
    import reprioritize_tail_20260928 as prior
    from run_followon_queue_20260928 import predecessor_state, save

    pins = {str(p): q.digest(p) for p in (Path(__file__), Path(q.__file__), Path(prior.__file__))}
    receipts, processes, provenance = {}, {}, {}
    for name, expected in OLD.items():
        root = q.ROOT / name
        path = root / "ACCEPTED.json"
        if q.digest(path) != expected or (root / "SUPERSEDED.json").exists():
            raise ValueError("old receipt changed or already superseded: " + name)
        receipt = json.loads(path.read_text())
        invocation = json.loads((root / "LAUNCH.json").read_text())
        process = psutil.Process(invocation["pid"])
        if process.cmdline() != invocation["argv"] or (
            abs(process.create_time() - invocation["started"]) > 5
        ):
            raise ValueError("waiting process identity differs: " + name)
        prior.assert_idle(root, receipt, process)
        receipts[name], processes[name] = receipt, process
        pins[str(path)] = expected
        provenance[name] = dict(
            launch=invocation,
            process_create_time=process.create_time(),
            receipt_sha256=expected,
            jobs=receipt["jobs"],
        )
    if predecessor_state(q.ROOT / PREDECESSOR)["settled"]:
        raise ValueError("predecessor released; do not interrupt newly active work")
    prepared = {}
    for kind, location, expected in (
        ("teaching", "textcraft-teaching-replication-20260928-001", args.teaching_sha),
        ("transfer", "textcraft-rl-transfer-20260928-001", args.transfer_sha),
    ):
        path = q.ROOT / location / "PREPARED-JOBS.json"
        if q.digest(path) != expected:
            raise ValueError("reviewed follow-up preparation changed: " + kind)
        prepared[kind] = json.loads(path.read_text())
        pins[str(path)] = expected
    if prepared["transfer"].get("warm_controls_independent") is not True:
        raise ValueError("warm controls need explicit independent-dependency review")
    jobs = extend_tail(
        inherited=receipts["information-first-tail-20260928-001"]["jobs"],
        mask=receipts["payload-mask-queue-20260928-001"]["jobs"],
        teaching=prepared["teaching"]["jobs"],
        transfer=prepared["transfer"]["jobs"],
    )
    # Extra provenance pins only; every inherited command/output/cap/original pin is retained.
    augmented = augment_pins(jobs, pins)
    question = (
        "Replicate the positive teaching-order result across another fit seed and goal panel; "
        "test transfer of the actual first RL endpoints before new-goal optimization. "
        "Keep independent helper/quantity/Phi/ALF probes ahead of these followups and then "
        "retain all original fresh-RL/reward/compact/payload-mask jobs."
    )
    limits = (
        "Supersedes two authenticated idle waiters only; zero scientific owners interrupted. "
        "Move the two unchanged independent warm-B descriptors, never duplicate their outputs. "
        "New teaching replication and four fixed-endpoint RL transfer readouts are exploratory. "
        "Same task roots across worlds and seeds remain dependent; diagnostic B is not an "
        "untouched official test. Actual checkpoints and native outcomes required. "
        "Original caps, source pins and all scientific limitations remain in force."
    )
    q.accept(NAME, PREDECESSOR, augmented, question, limits, launch=False)
    if not args.launch:
        return
    intent = q.ROOT / (NAME + "-INTENT.json")
    save(
        intent,
        dict(
            observed=time.time(),
            replacement=str(q.ROOT / NAME),
            old_waiters=provenance,
            new_preparation_sha256=dict(teaching=args.teaching_sha, transfer=args.transfer_sha),
            jobs_before_new_orchestration_pins=jobs,
            scientific_owners_to_stop=0,
            reason=question,
        ),
    )
    suspended = []
    try:
        for name, process in processes.items():
            process.suspend()
            suspended.append(process)
            prior.assert_idle(q.ROOT / name, receipts[name], process)
        if predecessor_state(q.ROOT / PREDECESSOR)["settled"]:
            raise ValueError("predecessor released during inspection; keep existing order")
        for process in processes.values():
            process.terminate()
    finally:
        for process in suspended:
            with suppress(psutil.NoSuchProcess):
                process.resume()
    for name, process in processes.items():
        code = process.wait(timeout=5)
        save(
            q.ROOT / name / "SUPERSEDED.json",
            dict(
                replacement=str(q.ROOT / NAME),
                intent=str(intent),
                original_receipt_sha256=OLD[name],
                original_receipt_unchanged=True,
                stopped_only_waiting_pid=process.pid,
                returncode=code,
                ended=time.time(),
                scientific_owners_stopped=0,
            ),
        )
    q.accept(NAME, PREDECESSOR, augmented, question, limits, launch=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teaching-sha", required=True)
    parser.add_argument("--transfer-sha", required=True)
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args())
