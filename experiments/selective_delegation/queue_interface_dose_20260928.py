"""Accept short format and longer-teaching comparisons after the current controls."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from queue_rl_updates_20260928 import PYTHON, ROOT, SUPERVISOR_PYTHON, digest
from run_followon_queue_20260928 import GENERIC

HERE = Path(__file__).resolve().parent
COMPACT = HERE / "compact_actions_20260928"
DOSE = HERE / "teaching_dose_20260928"
STUDY = ROOT / "textcraft-teaching-dose-20260928-001"


def make_jobs():
    paths = [
        Path(__file__).resolve(),
        HERE / "run_followon_queue_20260928.py",
        HERE / "queue_rl_updates_20260928.py",
        GENERIC,
        *(
            COMPACT / name
            for name in ("compact_bridge.py", "prepare.py", "train.py", "evaluate.py")
        ),
        *(DOSE / name for name in ("dose_common.py", "train.py", "readout.py")),
    ]
    pins = {str(path): digest(path) for path in paths}
    training = ROOT / "textcraft-compact-sft-20260928-001"
    for output in [training, *(STUDY / f"train-{t}" for t in ("discovery", "known_recipe"))]:
        path = output / "PLAN.json"
        plan = json.loads(path.read_text())
        pins[str(path)] = digest(path)
        if output == training:
            contract_path = output / "COMPACT-CONTRACT.json"
            contract = json.loads(contract_path.read_text())
            pins[str(contract_path)] = digest(contract_path)
            pins.update(contract["source_sha256"])
        else:
            pins.update(plan["source_sha256"])
    jobs = []

    def add(name, argv, output=None, cap=3000):
        jobs.append(
            dict(
                name=name,
                argv=argv,
                output=str(output) if output else None,
                cap_seconds=cap,
                pins=pins,
            )
        )

    add(
        "compact-train",
        [PYTHON, str(COMPACT / "train.py"), "--output", str(training), "--hours", ".5", "--resume"],
        training,
        2100,
    )
    for world in (42, 50):
        output = ROOT / f"textcraft-compact-p00-w{world}-20260928-001"
        argv = [
            PYTHON,
            str(COMPACT / "evaluate.py"),
            "--world",
            str(world),
            "--output",
            str(output),
        ]
        add(f"compact-w{world}", argv, output)
        add(
            f"compact-w{world}-audit",
            ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *argv, "--audit"],
            cap=600,
        )
    for teacher in ("discovery", "known_recipe"):
        add(
            f"dose-{teacher}-train",
            [PYTHON, str(DOSE / "train.py"), "--teacher", teacher, "--resume"],
            STUDY / f"train-{teacher}",
            2100,
        )
    for step in (46, 69):
        for world in (42, 50):
            for teacher in ("discovery", "known_recipe"):
                output = STUDY / f"eval-{teacher}-cp{step}-p00-w{world}-raw"
                argv = [
                    PYTHON,
                    str(DOSE / "readout.py"),
                    "--teacher",
                    teacher,
                    "--step",
                    str(step),
                    "--world",
                    str(world),
                    "--execution",
                    "raw",
                ]
                name = f"dose-{teacher}-cp{step}-w{world}"
                add(name, argv, output)
                add(
                    name + "-audit",
                    ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *argv, "--audit"],
                    cap=600,
                )
    return jobs


def main(launch):
    jobs = make_jobs()
    receipt = dict(
        status="accepted",
        questions=[
            "Does predicting only item and quantity help when code binds observed ingredients?",
            "Does the weaker known-recipe teacher simply need longer optimization?",
        ],
        authorized_by="September28 broad autonomous research instruction",
        predecessor_queue=str(ROOT / "teacher-controls-queue-20260928-001"),
        executor_sha256=digest(GENERIC),
        maximum_seconds=40 * 3600,
        allocation_end=os.environ["SLURM_JOB_END_TIME"],
        jobs=jobs,
        expected_useful_hours="About4–6 total; training short, "
        "ten paired16-episode readouts dominate",
        upper_job_caps_hours=sum(job["cap_seconds"] for job in jobs) / 3600,
        endpoints="Compact fixedcp23. Both original teachers continue real AdamW/RNG for46steps; "
        "all fixedcp46/cp69 readouts on worlds42/50 are reported, never selected by score.",
        limitations="Reused small exploratory panels. Compact changes output schema, instruction, "
        "token dose and rejects unseen recipes; fullformat binder sometimes falls through. "
        "Dose holds original teacher histories fixed, not the conditioning gap.",
    )
    if not launch:
        print(
            json.dumps(
                dict(
                    jobs=len(jobs),
                    gpu_jobs=sum(bool(j["output"]) for j in jobs),
                    upper_job_caps_hours=receipt["upper_job_caps_hours"],
                ),
                indent=2,
            )
        )
        return
    for job in jobs:
        if job["output"] and list(Path(job["output"]).glob("OWNER-*.json")):
            raise ValueError("already attempted: " + job["output"])
    output = ROOT / "interface-dose-queue-20260928-001"
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
