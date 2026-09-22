"""Run explicitly accepted, individually capped jobs after resource release.

Each scientific reader enforces its own data/checkpoint prerequisites. A failed
independent job does not prevent the next job from attempting its own preflight.
This launcher is for the direct, process-local GPU collectors in this campaign.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import psutil


def resource_released(output):
    owners = list(Path(output).glob("OWNER-*.json"))
    if not owners:
        return False
    for path in owners:
        owner = json.loads(path.read_text())
        if not path.with_name(path.name.replace("OWNER-", "TERMINAL-")).exists():
            return False
        try:
            process = psutil.Process(owner["pid"])
            if (
                abs(process.create_time() - owner["create_time"]) < 0.01
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                return False
        except psutil.NoSuchProcess:
            pass
    return True


def validate_pins(pins):
    for filename, expected in pins.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("accepted source/input changed: " + filename)


def execute_jobs(jobs, output, deadline):
    records = []
    for job in jobs:
        started = time.time()
        if started + job["cap_seconds"] + 120 > deadline:
            raise TimeoutError("insufficient remaining allocation/queue allowance")
        record = {"name": job["name"], "started": started, "argv": job["argv"]}
        try:
            validate_pins(job["pins"])
            with (output / (job["name"] + ".log")).open("x") as stream:
                result = subprocess.run(
                    job["argv"],
                    check=False,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=job["cap_seconds"] + 120,
                    env={
                        **os.environ,
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                    },
                )
            record["returncode"] = result.returncode
        except (ValueError, subprocess.TimeoutExpired) as exc:
            record.update(returncode=None, failure=f"{type(exc).__name__}: {exc}")
        record["ended"] = time.time()
        with (output / (job["name"] + ".json")).open("x") as stream:
            json.dump(record, stream, indent=2)
        records.append(record)
        print(json.dumps(record), flush=True)
        scientific_output = job.get("output")
        if (
            scientific_output
            and list(Path(scientific_output).glob("OWNER-*.json"))
            and not resource_released(scientific_output)
        ):
            raise RuntimeError("job owner unresolved; inspect before sharing GPU")
    return records


def main(args):
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("status") != "accepted" or not receipt.get("jobs"):
        raise ValueError("explicit accepted finite queue required")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "INVOCATION.json").open("x") as stream:
        json.dump(
            {
                "receipt": str(args.receipt.resolve()),
                "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
                "pid": os.getpid(),
                "started": time.time(),
            },
            stream,
            indent=2,
        )
    deadline = min(
        time.time() + receipt["maximum_seconds"], int(os.environ["SLURM_JOB_END_TIME"]) - 600
    )
    while not resource_released(receipt["predecessor"]):
        if time.time() >= deadline:
            raise TimeoutError("predecessor release wait exhausted")
        time.sleep(5)
    execute_jobs(receipt["jobs"], args.output, deadline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
