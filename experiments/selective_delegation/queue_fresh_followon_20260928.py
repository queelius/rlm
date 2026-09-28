"""Dispatch the prepared fresh-goal RL campaign after the interface/dose queue."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from queue_rl_updates_20260928 import ROOT, SUPERVISOR_PYTHON, digest
from run_followon_queue_20260928 import GENERIC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rl_fresh_20260928"))
import launch_campaign as campaign  # noqa: E402


def main(launch):
    # Reuse the reviewed experiment schedule unchanged; only the queue handoff differs.
    jobs = campaign.make_jobs(campaign.STUDY, 4)
    for job in jobs:
        job["pins"].update(
            {
                str(p): digest(p)
                for p in (Path(__file__), HERE / "run_followon_queue_20260928.py", GENERIC)
            }
        )
    receipt = json.loads((campaign.STUDY / "PREPARED-CAMPAIGN-0004.json").read_text())
    receipt.pop("predecessor")
    receipt.update(
        status="accepted",
        predecessor_queue=str(ROOT / "interface-dose-queue-20260928-001"),
        executor_sha256=digest(GENERIC),
        allocation_end=os.environ["SLURM_JOB_END_TIME"],
        jobs=jobs,
        handoff="Wait for the entire independent predecessor queue to settle, including "
        "preflight failures; require every acquired scientific owner released.",
    )
    if not launch:
        print(json.dumps(dict(jobs=len(jobs), useful_hours=receipt["expected_useful_hours"])))
        return
    for job in jobs:
        if job.get("output"):
            path = Path(job["output"])
            if list(path.glob("OWNER-*.json")) or (path / "CONDITIONAL-SKIP.json").exists():
                raise ValueError("already attempted: " + str(path))
    output = ROOT / "fresh-rl-queue-20260928-002"
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
