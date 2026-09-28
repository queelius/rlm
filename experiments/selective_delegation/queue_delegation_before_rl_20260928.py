"""Insert a short admission screen by superseding only an authenticated idle waiter.

The accepted scientific RL jobs and all downstream scientific paths stay intact.
No collector, model owner, accepted receipt, or pinned source is edited/stopped.
"""

import argparse
import json
import os
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import psutil
from queue_rl_updates_20260928 import PYTHON, ROOT, SUPERVISOR_PYTHON, digest
from run_followon_queue_20260928 import GENERIC, generic

HERE = Path(__file__).resolve().parent
OLD = ROOT / "rl-collect-queue-20260928-001"
OUTPUT = ROOT / "rl-collect-with-delegation-20260928-001"


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def main(launch):
    old = json.loads((OLD / "ACCEPTED.json").read_text())
    old_launch = json.loads((OLD / "LAUNCH.json").read_text())
    if len(old["jobs"]) != 2:
        raise ValueError("expected only two unstarted paired RL collection jobs")
    jobs, pins = (
        [],
        {
            str(p): digest(p)
            for p in (
                Path(__file__),
                HERE / "run_followon_queue_20260928.py",
                GENERIC,
            )
        },
    )
    for mode in ("flat", "fixed", "adaptive"):
        target = ROOT / f"textcraft-decomp-{mode}-20260928-001"
        path = target / "PLAN.json"
        plan = json.loads(path.read_text())
        if plan["admission_response_cap"] != 32 or plan["budget_seconds"] != 900:
            raise ValueError("bounded admission plan changed")
        pins[str(path)] = digest(path)
        pins.update(plan["source_sha256"])
        argv = [
            PYTHON,
            str(HERE / "decomposition_20260928/run.py"),
            "--mode",
            mode,
            "--output",
            str(target),
        ]
        jobs.append(
            dict(
                name=f"decomposition-{mode}",
                argv=argv,
                output=str(target),
                cap_seconds=1200,
                pins=pins,
            )
        )
        jobs.append(
            dict(
                name=f"decomposition-{mode}-audit",
                argv=["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *argv, "--audit"],
                cap_seconds=300,
                pins=pins,
            )
        )
    jobs.extend(old["jobs"])
    unique = dict(pins)
    for job in jobs:
        unique.update(job["pins"])
        if job.get("output") and list(Path(job["output"]).glob("OWNER-*.json")):
            raise ValueError("scientific job already attempted")
    generic.validate_pins(unique)
    receipt = dict(
        status="accepted",
        jobs=jobs,
        maximum_seconds=8 * 3600,
        predecessor_queue=str(ROOT / "resume-20260928-001"),
        executor_sha256=digest(GENERIC),
        allocation_end=os.environ["SLURM_JOB_END_TIME"],
        authorized_by="User requests autonomous informative research and adaptive priorities",
        reason="A6–15minute real delegation admission screen is ready. Run before longer RL "
        "collections without interrupting the current breadth collector. Preserve both "
        "original RL collection commands, inputs, caps and downstream scientific paths.",
        supersedes_waiter=str(OLD),
        original_receipt_sha256=digest(OLD / "ACCEPTED.json"),
        expected_useful_hours="Admission6–15min plus original paired RL collection estimates",
        upper_job_caps_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
        scientific_limits="One fresh task/seed, max96 real responses across3arms. This only "
        "tests helper uptake; planned admission stops retain unknown root success.",
    )
    if not launch:
        print(json.dumps(dict(jobs=len(jobs), unique_pins=len(unique), old_pid=old_launch["pid"])))
        return
    process = psutil.Process(old_launch["pid"])
    if (
        process.cmdline() != old_launch["argv"]
        or abs(process.create_time() - old_launch["started"]) > 5
    ):
        raise ValueError("old waiter identity differs")
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT / "ACCEPTED.json", receipt)
    save(
        OUTPUT / "REPRIORITIZATION-INTENT.json",
        dict(
            old_launch=old_launch,
            old_process_create_time=process.create_time(),
            observed=time.time(),
            current_GPU_owner_untouched=True,
        ),
    )
    # Briefly freeze ONLY the waiting queue to eliminate a spawn/check race.
    process.suspend()
    try:
        if process.children(recursive=True) or any(
            list(Path(j["output"]).glob("OWNER-*.json")) for j in old["jobs"]
        ):
            raise ValueError("waiter began scientific work; do not interrupt it")
        if generic.resource_released(old["predecessor"]):
            raise ValueError("predecessor already released; leave the ready collector alone")
        process.terminate()
    finally:
        with suppress(psutil.NoSuchProcess):
            process.resume()
    returncode = process.wait(timeout=5)
    save(
        OLD / "SUPERSEDED.json",
        dict(
            reason=receipt["reason"],
            replacement=str(OUTPUT),
            original_receipt_unchanged=True,
            stopped_only_waiting_pid=process.pid,
            returncode=returncode,
            ended=time.time(),
            scientific_jobs_started=0,
            scientific_owners_stopped=0,
        ),
    )
    argv = [
        SUPERVISOR_PYTHON,
        str(HERE / "run_followon_queue_20260928.py"),
        "--receipt",
        str(OUTPUT / "ACCEPTED.json"),
        "--output",
        str(OUTPUT / "queue"),
    ]
    with (OUTPUT / "supervisor.log").open("x") as stream:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(OUTPUT / "LAUNCH.json", dict(pid=child.pid, started=time.time(), argv=argv))
    print(
        json.dumps(
            dict(
                pid=child.pid,
                receipt=str(OUTPUT / "ACCEPTED.json"),
                superseded_waiter_pid=process.pid,
                scientific_owners_stopped=0,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args().launch)
