"""Continue accepted TRAIN collections with small updates and crossed readouts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
HERE = Path(__file__).resolve().parent
PROBE = HERE / "rl_resume_20260928"
STUDY = ROOT / "textcraft-rl-assist-20260928-001"
PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)
SUPERVISOR_PYTHON = "/project/alex_phd/envs/prime-rl-5990b1b/bin/python"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readout(args):
    summary_path = args.readout_from / "SUMMARY.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    if summary is None or not summary.get("endpoint_usable"):
        args.output.mkdir(parents=True, exist_ok=True)
        record = dict(
            reason="No usable committed training endpoint; no scientific readout attempted",
            training_summary=summary,
            training_summary_path=str(summary_path),
            observed_episodes=0,
            native_successes=None,
            GPU_loaded=False,
        )
        with (args.output / "CONDITIONAL-SKIP.json").open("x") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        return
    argv = [
        sys.executable,
        str(PROBE / "collect.py"),
        "--mode",
        args.mode,
        "--phase",
        "readout",
        "--hours",
        ".75",
        "--checkpoint",
        summary["endpoint"],
        "--output",
        str(args.output),
    ]
    os.execv(sys.executable, argv)


def make_jobs():
    jobs = []
    pins = {str(Path(__file__).resolve()): digest(Path(__file__))}
    for mode in ("raw", "binder"):
        for stage in ("collect-0001", "train-0001", "readout-warm"):
            path = STUDY / mode / stage / "PLAN.json"
            plan = json.loads(path.read_text())
            pins.update(plan["source_sha256"])
            pins[str(path)] = digest(path)
    pins[str(PROBE / "compare.py")] = digest(PROBE / "compare.py")

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

    for mode in ("raw", "binder"):
        target = STUDY / mode / "train-0001"
        add(
            "rl-" + mode + "-train-0001",
            [
                PYTHON,
                str(PROBE / "train.py"),
                "--collection",
                str(STUDY / mode / "collect-0001"),
                "--output",
                str(target),
            ],
            target,
            3900,
        )
    target = STUDY / "raw/readout-warm"
    add(
        "rl-raw-readout-warm",
        [
            PYTHON,
            str(PROBE / "collect.py"),
            "--mode",
            "raw",
            "--phase",
            "readout",
            "--hours",
            ".75",
            "--output",
            str(target),
        ],
        target,
    )
    for trained, executed in (
        ("raw", "raw"),
        ("binder", "binder"),
        ("raw", "binder"),
        ("binder", "raw"),
    ):
        name = "readout-0001" if trained == executed else "readout-0001-as-" + executed
        target = STUDY / trained / name
        add(
            "rl-" + trained + "-" + name,
            [
                PYTHON,
                str(Path(__file__).resolve()),
                "--readout-from",
                str(STUDY / trained / "train-0001"),
                "--mode",
                executed,
                "--output",
                str(target),
            ],
            target,
        )
    # Finish with an independent useful job, even when either optimizer is skipped.
    target = STUDY / "binder/readout-warm"
    add(
        "rl-binder-readout-warm",
        [
            PYTHON,
            str(PROBE / "collect.py"),
            "--mode",
            "binder",
            "--phase",
            "readout",
            "--hours",
            ".75",
            "--output",
            str(target),
        ],
        target,
    )
    add(
        "rl-own-interface-analysis",
        [
            PYTHON,
            str(PROBE / "compare.py"),
            "--root",
            str(STUDY),
            "--output",
            str(STUDY / "COMPARISON-0001.json"),
        ],
        cap=600,
    )
    return jobs


def launch(do_launch):
    jobs = make_jobs()
    for job in jobs:
        if job["output"]:
            path = Path(job["output"])
            if list(path.glob("OWNER-*.json")) or (path / "CONDITIONAL-SKIP.json").exists():
                raise ValueError("already attempted: " + str(path))
    receipt = dict(
        status="accepted",
        question="Does deterministic execution assistance change what terminal reward "
        "learning improves, and does any learned change transfer between interfaces?",
        authorized_by="September 28 user direction: autonomous broad informative experiments",
        predecessor=str(STUDY / "binder/collect-0001"),
        maximum_seconds=43200,
        jobs=jobs,
        expected_useful_hours="2 to 4 after predecessor collection, conditional on updates",
        upper_job_caps_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
        queue_bound_hours=12,
        current_allocation_end=os.environ["SLURM_JOB_END_TIME"],
        limits="Exploratory TRAIN8 mechanism study. Sixteen paired readout attempts per cell; "
        "eight task clusters. No held-out efficacy claim. Absent training endpoints are skipped, "
        "not evaluated as if trained. Fresh-goal and extra-SFT controls prepared separately.",
    )
    if not do_launch:
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
    output = ROOT / "rl-update-queue-20260928-001"
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
    parser.add_argument("--readout-from", type=Path)
    parser.add_argument("--mode", choices=("raw", "binder"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.readout_from:
        if not args.mode or not args.output:
            parser.error("readout requires --mode and --output")
        readout(args)
    else:
        launch(args.launch)
