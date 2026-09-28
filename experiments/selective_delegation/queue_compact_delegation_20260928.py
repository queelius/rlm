"""Append the compact RL cycle and repaired helper-uptake screen, without races."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from queue_rl_updates_20260928 import PYTHON, ROOT, SUPERVISOR_PYTHON, digest
from run_followon_queue_20260928 import GENERIC, generic

HERE = Path(__file__).resolve().parent
PREPARED = ROOT / "textcraft-compact-rl-20260928-001/PREPARED-JOBS.json"
PREPARED_SHA = "4603f4e1ab13318bf30933555377b0ef5ecdb6a998ceae93d531ab0b9cbed0d0"


def main(launch):
    if digest(PREPARED) != PREPARED_SHA:
        raise ValueError("reviewed compact RL preparation changed")
    jobs = []
    pins = {
        str(p): digest(p)
        for p in (Path(__file__), HERE / "run_followon_queue_20260928.py", GENERIC, PREPARED)
    }
    fixture = ROOT / "decomposition-flexible-cpu-fixture-20260928-002/VERIFICATION.json"
    if digest(fixture) != "08ed97452bc79b0ea4bc4b9be6f645bf0c28f15859d215959384c3a32c0b6167":
        raise ValueError("reviewed flexible saved-request fixture changed")
    pins[str(fixture)] = digest(fixture)
    for mode in ("flat", "fixed", "adaptive"):
        output = ROOT / f"textcraft-decomp-flexible-{mode}-20260928-001"
        path = output / "PLAN.json"
        plan = json.loads(path.read_text())
        if (
            plan["schema"] != "textcraft-decomposition-flexible-admission-20260928-v2"
            or plan["admission_response_cap"] != 32
            or plan["budget_seconds"] != 900
        ):
            raise ValueError("reviewed flexible admission plan changed")
        job_pins = {**pins, **plan["source_sha256"], str(path): digest(path)}
        argv = [
            PYTHON,
            str(HERE / "decomposition_flexible_20260928/run_flexible.py"),
            "--mode",
            mode,
            "--output",
            str(output),
        ]
        jobs.append(
            dict(
                name=f"flexible-decomposition-{mode}",
                argv=argv,
                output=str(output),
                cap_seconds=1200,
                pins=job_pins,
            )
        )
        jobs.append(
            dict(
                name=f"flexible-decomposition-{mode}-audit",
                argv=["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *argv, "--audit"],
                cap_seconds=300,
                pins=job_pins,
            )
        )
    for job in json.loads(PREPARED.read_text())["jobs"]:
        job["pins"].update(pins)
        jobs.append(job)
    unique = {}
    for job in jobs:
        unique.update(job["pins"])
        if job.get("output") and list(Path(job["output"]).glob("OWNER-*.json")):
            raise ValueError("existing scientific attempt: " + job["output"])
    generic.validate_pins(unique)
    receipt = dict(
        status="accepted",
        authorized_by="September28 autonomous broad informative research request",
        predecessor_queue=str(ROOT / "reward-diagnostics-queue-20260928-001"),
        executor_sha256=digest(GENERIC),
        maximum_seconds=72 * 3600,
        allocation_end=os.environ["SLURM_JOB_END_TIME"],
        jobs=jobs,
        expected_useful_hours="Compact RL2–4h, bounded delegation screen6–15min",
        upper_job_caps_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
        question="Does the compact interface learn from rewards beyond its own unchanged "
        "weights? Separately, can the existing actor actually invoke a required helper "
        "when useful recipe queries are no longer rejected for ordering?",
        scientific_limits="Prepared future compact endpoint is bound only when real. "
        "Matched SFT steps differ in token exposure and unobserved-recipe semantics. "
        "Delegation screen reuses one exploratory task and forces the public rule's "
        "subgoal; this is not learned routing or a recursion efficacy claim.",
    )
    if not launch:
        print(
            json.dumps(
                dict(
                    jobs=len(jobs),
                    unique_pins=len(unique),
                    upper_job_caps_hours=receipt["upper_job_caps_hours"],
                )
            )
        )
        return
    output = ROOT / "compact-delegation-queue-20260928-001"
    output.mkdir(exist_ok=False)
    with (output / "ACCEPTED.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    argv = [
        SUPERVISOR_PYTHON,
        str(HERE / "run_followon_queue_20260928.py"),
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
    print(
        json.dumps(
            dict(
                pid=child.pid,
                receipt=str(output / "ACCEPTED.json"),
                receipt_sha256=digest(output / "ACCEPTED.json"),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args().launch)
