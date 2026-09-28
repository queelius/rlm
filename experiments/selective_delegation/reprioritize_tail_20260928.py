"""Promote independent probes without changing or interrupting scientific jobs."""

import argparse
import json
import sys
import time
from contextlib import suppress
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "information-first-tail-20260928-001"
PREDECESSOR = "interface-dose-queue-20260928-001"
# Downstream-to-upstream order is necessary: predecessor exit releases a waiter.
OLD = {
    "transfer-queue-20260928-001": "e4d68a61a95076d3b35b68557200eb42812cc77c7331f8cdeb5efe8239a01253",
    "compact-delegation-queue-20260928-001": "9452210ee2b4700ebc0b0c5f6d8a5b6c5b3668a657dea4f407c9cb809b5bda35",
    "reward-diagnostics-queue-20260928-001": "2a5c29c0edeeccaf8249e12ff6c7657bd96f4d2e40a1d4d5d8ad00d221f28215",
    "fresh-rl-queue-20260928-002": "50cf6866182a5052e5e5964a28d8bea5793b8048147c2a6eee54a3eb660921ba",
}
ALF_SHA = "56497c642c69fc3a2e4cb822e358ddf77a8ce04bb9cffb74e9d462f2dee42b71"


def prioritize(compact, transfer, alf, fresh, reward):
    screens = [j for j in compact if j["name"].startswith("flexible-decomposition-")]
    if len(screens) != 6 or sum(bool(j.get("output")) for j in screens) != 3:
        raise ValueError("expected six screen descriptors, including three native audits")
    remainder = [j for j in compact if j not in screens]
    jobs = screens + transfer + alf + fresh + reward + remainder
    names = [j["name"] for j in jobs]
    outputs = [j["output"] for j in jobs if j.get("output")]
    if len(names) != len(set(names)):
        raise ValueError("duplicate job name")
    if len(outputs) != len(set(outputs)):
        raise ValueError("duplicate scientific output")
    return jobs


def assert_idle(root, receipt, process):
    if process.children(recursive=True):
        raise ValueError("waiter has children; leave its work alone")
    if {path.name for path in (root / "queue").iterdir()} != {"INVOCATION.json"}:
        raise ValueError("waiter has begun dispatch; leave its work alone")
    if any(
        list(Path(job["output"]).glob("OWNER-*.json"))
        for job in receipt["jobs"]
        if job.get("output")
    ):
        raise ValueError("scientific owner exists; no interruption authorized here")


def main(launch):
    import psutil

    sys.path.insert(0, str(HERE))
    import queue_transfer_20260928 as q
    from run_followon_queue_20260928 import predecessor_state, save

    receipts, processes, provenance = {}, {}, {}
    pins = {str(Path(__file__).resolve()): q.digest(Path(__file__))}
    pins[str(Path(q.__file__))] = q.digest(Path(q.__file__))
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
        assert_idle(root, receipt, process)
        receipts[name], processes[name] = receipt, process
        pins[str(path)] = expected
        provenance[name] = dict(
            launch=invocation,
            process_create_time=process.create_time(),
            receipt_sha256=expected,
            jobs=receipt["jobs"],
        )
    if predecessor_state(q.ROOT / PREDECESSOR)["settled"]:
        raise ValueError("predecessor already released; leave ready work alone")
    alf_path = q.ROOT / "alfworld-representation-20260928-001/PREPARED-JOBS.json"
    if q.digest(alf_path) != ALF_SHA:
        raise ValueError("reviewed ALFWorld preparation changed")
    pins[str(alf_path)] = ALF_SHA
    alf = json.loads(alf_path.read_text())
    jobs = prioritize(
        compact=receipts["compact-delegation-queue-20260928-001"]["jobs"],
        transfer=receipts["transfer-queue-20260928-001"]["jobs"],
        alf=alf["jobs"],
        fresh=receipts["fresh-rl-queue-20260928-002"]["jobs"],
        reward=receipts["reward-diagnostics-queue-20260928-001"]["jobs"],
    )
    # Original job content is retained, including every original pin. The only
    # added keys authenticate this waiting-order decision and its input receipts.
    augmented = [dict(job, pins={**job["pins"], **pins}) for job in jobs]
    question = (
        "Prioritize distinct explanations before repeated RL cycles: flexible helper uptake, "
        "public quantity calculation, second-model teaching transfer, ALFWorld learning "
        "representation, then unchanged fresh RL/reward/compact learning comparisons."
    )
    limits = (
        "Supersedes four idle waiters only; all scientific commands, output paths, caps and "
        "original pins unchanged. ALFWorld adds one SFT and four readouts on12 new games. "
        "Both representations retain the same available-action list/history; target tokens "
        "and per-example loss weighting differ. All inherited limitations still apply. "
        "Conditional unavailable endpoints remain skips/unknown, never fallback weights."
    )
    q.accept(NAME, PREDECESSOR, augmented, question, limits, launch=False)
    if not launch:
        return
    intent = q.ROOT / (NAME + "-INTENT.json")
    save(
        intent,
        dict(
            observed=time.time(),
            replacement=str(q.ROOT / NAME),
            old_waiters=provenance,
            new_ALF_preparation_sha256=ALF_SHA,
            jobs_before_new_orchestration_pins=jobs,
            exact_original_job_preservation=True,
            scientific_owners_to_stop=0,
        ),
    )
    suspended = []
    try:
        for name, process in processes.items():
            process.suspend()
            suspended.append(process)
            assert_idle(q.ROOT / name, receipts[name], process)
        if predecessor_state(q.ROOT / PREDECESSOR)["settled"]:
            raise ValueError("predecessor released during inspection; retain old waiting order")
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
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args().launch)
