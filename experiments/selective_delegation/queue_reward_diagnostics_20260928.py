"""Accept distinct reward-cost and fixed-fit diagnostics after the fresh RL study."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from queue_rl_updates_20260928 import PYTHON, ROOT, SUPERVISOR_PYTHON, digest
from run_followon_queue_20260928 import GENERIC, generic

HERE = Path(__file__).resolve().parent
PREPARED = ROOT / "textcraft-error-cost-20260928-001/PREPARED-JOBS.json"
PREPARED_SHA = "d0d3b76a258ec2a5f55bc13bdac1a557804b65d424b1ba906d420e686448a0e8"


def main(launch):
    if digest(PREPARED) != PREPARED_SHA:
        raise ValueError("reviewed cost-reward preparation changed")
    prepared = json.loads(PREPARED.read_text())
    jobs = prepared["jobs"]
    extra = {
        str(p): digest(p)
        for p in (Path(__file__), HERE / "run_followon_queue_20260928.py", GENERIC, PREPARED)
    }
    for job in jobs:
        job["pins"].update(extra)
    nll = HERE / "teaching_dose_20260928/nll.py"
    nll_pins = {
        **extra,
        **{
            str(nll.with_name(name)): digest(nll.with_name(name))
            for name in ("nll.py", "dose_common.py", "readout.py")
        },
    }
    for step in (23, 46, 69):
        for teacher in ("discovery", "known_recipe"):
            output = (
                ROOT
                / "textcraft-teaching-dose-20260928-001"
                / (f"nll-{teacher}-cp{step}-on-{teacher}")
            )
            jobs.append(
                dict(
                    name=f"fit-{teacher}-cp{step}",
                    argv=[PYTHON, str(nll), "--teacher", teacher, "--step", str(step)],
                    output=str(output),
                    cap_seconds=1200,
                    pins=nll_pins,
                )
            )
    # Check each unique small source/input once, before any GPU lock.
    unique = {}
    for job in jobs:
        unique.update(job["pins"])
    generic.validate_pins(unique)
    receipt = dict(
        status="accepted",
        authorized_by="September28 autonomous informative experiment portfolio",
        predecessor_queue=str(ROOT / "fresh-rl-queue-20260928-002"),
        executor_sha256=digest(GENERIC),
        allocation_end=os.environ["SLURM_JOB_END_TIME"],
        maximum_seconds=70 * 3600,
        jobs=jobs,
        expected_useful_hours="Cost learning1.3–3h, six fixed training-fit measurements6–18min",
        upper_job_caps_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
        question="Does an error penalty improve native task success, or only encourage "
        "cheap failure? Separately, do the teacher models fit their own examples after "
        "one/two/three epochs even when rollout success remains low?",
        scientific_limits="Changed multiobjective reward, not equivalent potential shaping. "
        "Reuse exactly first fresh batch/warm actor once per objective. Success remains "
        "primary. Fit uses full target-token NLL on TRAIN, not rollout accuracy or hidden "
        "name-only likelihood. No favorable checkpoint selection.",
        reviewed_preparation_sha256=PREPARED_SHA,
    )
    if not launch:
        print(
            json.dumps(
                dict(
                    jobs=len(jobs),
                    pins=len(unique),
                    upper_job_caps_hours=receipt["upper_job_caps_hours"],
                )
            )
        )
        return
    for job in jobs:
        if job.get("output") and list(Path(job["output"]).glob("OWNER-*.json")):
            raise ValueError("already attempted: " + job["output"])
    output = ROOT / "reward-diagnostics-queue-20260928-001"
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
    print(json.dumps(dict(pid=child.pid, receipt=str(output / "ACCEPTED.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args().launch)
