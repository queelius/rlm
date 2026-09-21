"""Two accepted bounded follow-ups; no model calls or open-ended scheduling here."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import probe


def run(root):
    source = Path(__file__).resolve().parent
    pilot = root / "pilot-001"
    plan = {
        "question": "Separate final-answer wording from action differences, then test new parents.",
        "source_hashes": {
            p.name: probe.campaign.sha(p)
            for p in (source / "followups.py", source / "probe.py", source / "answer_format.py")
        },
        "jobs": [
            [
                sys.executable,
                str(source / "answer_format.py"),
                "--source-output",
                str(pilot),
                "--cases",
                str(root / "inputs-001/cases.jsonl"),
                "--output",
                str(root / "answer-format-001"),
                "--hours",
                "2",
            ],
            [
                sys.executable,
                str(source / "probe.py"),
                "run",
                "--cases",
                str(root / "inputs-001/cases.jsonl"),
                "--output",
                str(root / "validation-span-001"),
                "--split",
                "validation",
                "--limit",
                "32",
                "--answer-style",
                "span",
                "--hours",
                "2",
            ],
        ],
        "validation_decision": "Freeze short-answer contract before seeing format outcomes. "
        "This is an exploratory validation block, not an untouched final test.",
        "started": time.time(),
        "pid": os.getpid(),
    }
    destination = root / "followups-001"
    destination.mkdir(exist_ok=False)
    probe.runtime.save(destination / "PLAN.json", plan)
    deadline = min(time.time() + 6 * 3600, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    stopping = False
    child = None

    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    while not list(pilot.glob("TERMINAL-*.json")) and time.time() < deadline and not stopping:
        time.sleep(5)
    terminals = [json.loads(p.read_text()) for p in pilot.glob("TERMINAL-*.json")]
    if not terminals or any(t.get("failure") for t in terminals):
        raise RuntimeError("pilot did not finish successfully; do not launch dependent jobs")
    # TERMINAL is written immediately before unlock; allow the owner to release normally.
    time.sleep(2)
    completed = []
    for index, command in enumerate(plan["jobs"]):
        if stopping or time.time() > deadline - 300:
            break
        with (destination / f"job-{index}.log").open("x") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            probe.runtime.save(
                destination / f"START-{index}.json",
                {"pid": child.pid, "command": command, "started": time.time()},
            )
            while child.poll() is None:
                if time.time() >= deadline:
                    stop()
                time.sleep(5)
            completed.append({"job": index, "returncode": child.returncode, "ended": time.time()})
            probe.campaign.snapshot(destination / "STATUS.json", {"completed": completed})
            if child.returncode != 0:
                break
    probe.runtime.save(
        destination / "TERMINAL.json",
        {"completed": completed, "stopping": stopping, "ended": time.time()},
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
