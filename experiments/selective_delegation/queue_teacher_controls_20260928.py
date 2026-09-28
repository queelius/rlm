"""Accept two matched teacher-order trainings and paired familiar/changed-world readouts."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from queue_rl_updates_20260928 import PYTHON, ROOT, SUPERVISOR_PYTHON, digest

HERE = Path(__file__).resolve().parent / "teaching_order_20260928"
STUDY = ROOT / "textcraft-teaching-order-20260928-001"


def main(launch):
    pins = {
        str(p): digest(p)
        for p in (
            HERE / "prepare.py",
            HERE / "train.py",
            HERE / "readout.py",
            Path(__file__).resolve(),
        )
    }
    jobs = []
    for mode in ("stable_visible", "random_visible"):
        prepared = STUDY / mode
        manifest = json.loads((prepared / "MANIFEST.json").read_text())
        if digest(prepared / "rows.jsonl") != manifest["rows_sha256"]:
            raise ValueError("teacher rows changed")
        for name in ("rows.jsonl", "MANIFEST.json", "tasks.jsonl"):
            path = prepared / name
            pins[str(path)] = digest(path)
        training = STUDY / f"train-{mode}-seed2026092208"
        plan_path = training / "PLAN.json"
        plan = json.loads(plan_path.read_text())
        if plan["planned_updates"] != 23 or plan["seed"] != 2026092208:
            raise ValueError("matched fixed-dose control differs")
        pins[str(plan_path)] = digest(plan_path)
        jobs.append(
            dict(
                name="teacher-" + mode + "-train",
                output=str(training),
                cap_seconds=2100,
                argv=[
                    PYTHON,
                    str(HERE / "train.py"),
                    "--mode",
                    mode,
                    "--seed",
                    "2026092208",
                    "--resume",
                ],
                pins=pins,
            )
        )
    for world in (42, 50):
        for mode in ("stable_visible", "random_visible"):
            output = STUDY / f"eval-{mode}-s2026092208-p00-w{world}-raw"
            argv = [
                PYTHON,
                str(HERE / "readout.py"),
                "--mode",
                mode,
                "--seed",
                "2026092208",
                "--panel",
                "0",
                "--world",
                str(world),
                "--execution",
                "raw",
            ]
            jobs.append(
                dict(
                    name=f"teacher-{mode}-w{world}",
                    output=str(output),
                    cap_seconds=3000,
                    argv=argv,
                    pins=pins,
                )
            )
            jobs.append(
                dict(
                    name=f"teacher-{mode}-w{world}-audit",
                    cap_seconds=600,
                    argv=["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *argv, "--audit"],
                    pins=pins,
                )
            )
    for job in jobs:
        if job.get("output") and list(Path(job["output"]).glob("OWNER-*.json")):
            raise ValueError("already attempted: " + job["output"])
    receipt = dict(
        status="accepted",
        question="Does revealing prerequisite names before demonstrating queries improve "
        "learnability when exact actions and per-minibatch supervised targets are held fixed?",
        authorized_by="September28 user instruction for broad autonomous research",
        predecessor=str(ROOT / "textcraft-rl-assist-20260928-001/binder/readout-warm"),
        maximum_seconds=57600,
        current_allocation_end=os.environ["SLURM_JOB_END_TIME"],
        jobs=jobs,
        expected_useful_hours="about1.5 to2 following earlier accepted RL study",
        upper_job_caps_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
        queue_bound_hours=16,
        pairing="Original known-teacher and discovery reference outcomes already exist for "
        "panel00/worlds42,50/seed2026092208. New teachers preserve known-teacher target row order, "
        "initial weights/optimizer/shuffle/23updates and8820labeltokens. Input histories differ.",
        limitation="Exposed exploratory panel; offline teacher schedules, not learned "
        "decomposition. Native success/cost and visible-query behavior are primary diagnostics.",
        failure_policy="Preserve failures, attempt independent next jobs, do not call absent "
        "fixed endpoints trained. Parent checks queue transitions and native responses.",
    )
    if not launch:
        print(
            json.dumps(
                dict(
                    jobs=[j["name"] for j in jobs],
                    upper_job_caps_hours=receipt["upper_job_caps_hours"],
                ),
                indent=2,
            )
        )
        return
    output = ROOT / "teacher-controls-queue-20260928-001"
    output.mkdir(exist_ok=False)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args().launch)
