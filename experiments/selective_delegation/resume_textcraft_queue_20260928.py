"""Finish a saved comparison and one prospective replication under a short cap."""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")


def main(launch=False):
    original = ROOT / "TEXTCRAFT-BREADTH-CAMPAIGN-002.json"
    receipt = json.loads(original.read_text())
    jobs = {job["name"]: job for job in receipt["jobs"]}
    first = "textcraft-breadth-p06-w42-soriginal"
    second = "textcraft-breadth-p06-w50-soriginal"
    names = [
        first + "-binder-001",
        first + "-raw-001-audit",
        first + "-binder-001-audit",
        "analysis-" + first + "-001",
        second + "-raw-001",
        second + "-raw-001-audit",
        second + "-binder-001",
        second + "-binder-001-audit",
        "analysis-" + second + "-001",
    ]
    selected = [jobs[name] for name in names]
    for job in selected:
        if job.get("output") and Path(job["output"]).exists():
            raise FileExistsError("Refuse duplicate collection: " + job["output"])
    now = time.time()
    deadline = min(now + 12600, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    receipt.update(
        campaign="September 28 bounded restart: complete pair and replicate on changed recipes",
        question="Does recipe argument binding help the next preselected unseen goal panel?",
        predecessor=str(ROOT / (first + "-raw-001")),
        jobs=selected,
        maximum_seconds=12600,
        deadline_utc=dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat(),
        ordering="Finish the already-collected baseline pair, then raw/assisted world50 pair.",
        rationale="Prespecified panel6 was unexamined when six completed panels were summarized. "
        "Three GPU jobs cover preparation of the next RL experiment; no completed run is repeated.",
        parent_receipt=str(original),
        parent_receipt_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        resume_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reservation={
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_START_TIME",
                "SLURM_JOB_END_TIME",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    )
    if not launch:
        print(json.dumps({"jobs": names, "deadline": receipt["deadline_utc"]}, indent=2))
        return
    output = ROOT / "resume-20260928-001"
    output.mkdir(exist_ok=False)
    target = output / "ACCEPTED.json"
    with target.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    python = "/project/alex_phd/envs/prime-rl-5990b1b/bin/python"
    argv = [
        python,
        str(ROOT / "source-fixed-deadline-campaign-001/run_fixed_deadline_campaign.py"),
        "--receipt",
        str(target),
        "--output",
        str(output / "campaign"),
        "--queue-output",
        str(output / "queue"),
        "--generic",
        str(ROOT / "source-queue-handoff-generic-004/run_independent_queue.py"),
        "--python",
        python,
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
        json.dump({"pid": child.pid, "started": now, "argv": argv}, stream, indent=2)
    print(json.dumps({"pid": child.pid, "output": str(output), "jobs": names}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args().launch)
