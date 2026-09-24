"""Bound an accepted independent queue by an absolute UTC deadline.

The underlying runner retains ownership, pin validation, release and failure-continuation
semantics. This supervisor enforces the earlier requested/allocation deadline and
checks initial response counts. Sustained all-failed calls trigger a graceful stop
of the matching recorded owner. The scientific sources are never edited.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil


def deadline_epoch(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC deadline requires timezone")
    return int(parsed.timestamp())


def save_exclusive(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def stop_authenticated_owner(path):
    try:
        owner = json.loads(path.read_text())
        process = psutil.Process(owner["pid"])
    except (FileNotFoundError, json.JSONDecodeError, psutil.NoSuchProcess):
        return False
    if abs(process.create_time() - owner["create_time"]) >= 0.01:
        raise ValueError("OWNER pid identity changed")
    os.kill(owner["pid"], signal.SIGTERM)
    return True


def main(args):
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("status") != "accepted" or not receipt.get("jobs"):
        raise ValueError("explicit accepted queue required")
    deadline = deadline_epoch(receipt["deadline_utc"])
    inherited_end = int(os.environ.get("SLURM_JOB_END_TIME", deadline + 600)) - 600
    deadline = min(deadline, inherited_end)
    if deadline <= time.time() + 600:
        raise TimeoutError("campaign deadline already too near")
    args.output.mkdir(parents=True, exist_ok=True)
    save_exclusive(
        args.output / "CAMPAIGN-INVOCATION.json",
        {
            "pid": os.getpid(),
            "started": time.time(),
            "receipt": str(args.receipt.resolve()),
            "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
            "deadline_utc": receipt["deadline_utc"],
            "deadline_epoch": deadline,
        },
    )
    env = dict(os.environ, SLURM_JOB_END_TIME=str(deadline + 600))
    child = subprocess.Popen(
        [
            str(args.python), str(args.generic), "--receipt", str(args.receipt),
            "--output", str(args.queue_output),
        ],
        env=env,
        start_new_session=True,
    )
    checked = set()
    while child.poll() is None:
        now = time.time()
        if now >= deadline:
            os.killpg(child.pid, signal.SIGTERM)
            raise TimeoutError("fixed UTC deadline reached")
        for job in receipt["jobs"]:
            target = job.get("output")
            if not target or job["name"] in checked:
                continue
            owner = list(Path(target).glob("OWNER-*.json"))
            if owner:
                status_path = Path(target) / "STATUS.json"
                calls = list((Path(target) / "calls").glob("*.json"))
                if not status_path.exists() or not calls:
                    continue
                try:
                    first_started = min(json.loads(path.read_text())["started"] for path in calls)
                    status = json.loads(status_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    continue
                if now - first_started < 90:
                    continue
                stopped = False
                if status.get("returned", 0) == 0 and status.get("failed", 0) > 0:
                    stopped = stop_authenticated_owner(owner[0])
                save_exclusive(
                    args.output / (job["name"] + "-FIRST-RESPONSE.json"),
                    {
                        "checked": now,
                        "owner": str(owner[0]),
                        "returned": status.get("returned"),
                        "failed": status.get("failed"),
                        "first_request_started": first_started,
                        "authenticated_stop_sent": stopped,
                        "warning": (
                            "observation only; inspect zero receipts before allowing an error loop"
                        ),
                    },
                )
                checked.add(job["name"])
        time.sleep(15)
    save_exclusive(
        args.output / "CAMPAIGN-TERMINAL.json",
        {"ended": time.time(), "returncode": child.returncode},
    )
    if child.returncode:
        raise SystemExit(child.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    main(parser.parse_args())
