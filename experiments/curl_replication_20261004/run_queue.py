"""Serial, foreground CURL queue; config/outputs live outside Git."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # The child can be between writing a record and its newline.
    return result


def completed(output: Path, job: dict[str, Any]) -> bool:
    config_path = output / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        mapping = {
            "arm": "arm",
            "seed": "seed",
            "steps": "num_train_steps",
            "eval_every": "eval_freq",
            "eval_episodes": "num_eval_episodes",
            "cap": "max_seconds",
        }
        if any(config.get(key) != job[name] for name, key in mapping.items()):
            raise ValueError(f"Existing job ID has different immutable inputs: {job['id']}")
    ends = [record for record in records(output / "metrics.jsonl") if record.get("type") == "end"]
    return bool(
        ends
        and ends[-1].get("reason") == "completed"
        and ends[-1].get("step") == job["steps"]
        and (output / "latest.pt").exists()
    )


def run_queue(config: dict[str, Any]) -> None:
    root, snapshot = Path(config["campaign_root"]), Path(config["source_snapshot"])
    root.mkdir(parents=True, exist_ok=True)
    stopped = {"signal": None}
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous_handlers:
        signal.signal(sig, lambda signum, frame: stopped.update(signal=signum))

    def stop_requested() -> bool:
        return bool(stopped["signal"] or (root / "STOP").exists())

    def event(kind: str, **fields: Any) -> None:
        with (root / "queue.jsonl").open("a") as handle:
            handle.write(json.dumps({"type": kind, "time_unix": time.time(), **fields}) + "\n")

    try:
        ids = [job["id"] for job in config["jobs"]]
        if len(ids) != len(set(ids)) or any(
            Path(value).name != value or value in (".", "..") for value in ids
        ):
            raise ValueError("Job IDs must be unique single path components")
        for job in config["jobs"]:
            job_root = root / job["id"]
            output = job_root / "run"
            if stop_requested() or time.time() + job["cap"] + 300 > config["allocation_deadline"]:
                event("queue_stop", id=job["id"], reason="stop_or_deadline")
                break
            exists = job_root.exists()
            if (job_root / "job.json").exists() and json.loads((job_root / "job.json").read_text())[
                "job"
            ] != job:
                raise ValueError(f"Job ID has different immutable inputs: {job['id']}")
            if exists and completed(output, job):
                receipt = job_root / "result.json"
                if not receipt.exists() or json.loads(receipt.read_text())["complete"]:
                    event("skip_complete", id=job["id"])
                    continue
            if not exists:
                job_root.mkdir()
                shutil.copytree(
                    snapshot, job_root / "code", ignore=shutil.ignore_patterns("__pycache__")
                )
                hashes = {
                    str(path.relative_to(job_root / "code")): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in (job_root / "code").rglob("*")
                    if path.is_file()
                }
                (job_root / "job.json").write_text(
                    json.dumps({"job": job, "source_hashes": hashes}, indent=2) + "\n"
                )
                versions = {}
                for package in ("torch", "numpy", "gym", "dm-control", "mujoco", "scikit-image"):
                    try:
                        versions[package] = importlib.metadata.version(package)
                    except importlib.metadata.PackageNotFoundError:
                        versions[package] = None
                selected = (
                    "SLURM_JOB_ID",
                    "SLURM_JOB_END_TIME",
                    "CUDA_VISIBLE_DEVICES",
                    "MUJOCO_GL",
                    "PYOPENGL_PLATFORM",
                    "OMP_NUM_THREADS",
                )
                provenance = {
                    "platform": platform.platform(),
                    "runner_python": sys.version,
                    "runner_executable": sys.executable,
                    "child_python": config["python"],
                    "package_versions_in_runner_environment": versions,
                    "environment": {key: os.environ.get(key) for key in selected},
                }
                provenance["environment"].update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
                (job_root / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
            with (root / "GPU.lock").open("a") as lock:
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if (
                            stop_requested()
                            or time.time() + job["cap"] + 300 > config["allocation_deadline"]
                        ):
                            event(
                                "queue_stop", id=job["id"], reason="stop_or_deadline_waiting_lock"
                            )
                            return
                        time.sleep(1)
                if exists:
                    receipt = job_root / "result.json"
                    valid = completed(output, job) and (
                        not receipt.exists() or json.loads(receipt.read_text())["complete"]
                    )
                    event("skip_complete" if valid else "preserve_attempt", id=job["id"])
                    continue
                if (
                    stop_requested()
                    or time.time() + job["cap"] + 300 > config["allocation_deadline"]
                ):
                    event("queue_stop", id=job["id"], reason="stop_or_deadline_after_lock")
                    break
                options = {
                    "source": config["source"],
                    "output": output,
                    "arm": job["arm"],
                    "seed": job["seed"],
                    "steps": job["steps"],
                    "max-seconds": job["cap"],
                    "eval-every": job["eval_every"],
                    "eval-episodes": job["eval_episodes"],
                }
                command = [config["python"], str(job_root / "code" / "train_reference.py")]
                command += [
                    value
                    for key, argument in options.items()
                    for value in (f"--{key}", str(argument))
                ]
                environment = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
                with (job_root / "console.log").open("w") as console:
                    child = subprocess.Popen(
                        command, stdout=console, stderr=subprocess.STDOUT, env=environment
                    )
                    event("launch", id=job["id"], pid=child.pid, command=command)
                    start, last_progress, count, warned, terminate_at = (
                        time.monotonic(),
                        time.monotonic(),
                        0,
                        False,
                        None,
                    )
                    reason = None
                    while child.poll() is None:
                        scientific = [
                            row
                            for row in records(output / "metrics.jsonl")
                            if row.get("type") in ("eval_episode", "train_episode", "train_metric")
                        ]
                        now = time.monotonic()
                        if len(scientific) > count:
                            if count == 0:
                                event(
                                    "scientific_response",
                                    id=job["id"],
                                    pid=child.pid,
                                    seconds=now - start,
                                )
                            count, last_progress = len(scientific), now
                        if not warned and count == 0 and now - start >= 90:
                            event("startup_no_scientific_record", id=job["id"], pid=child.pid)
                            warned = True
                        reason = reason or (
                            "stop"
                            if stop_requested()
                            else "stalled"
                            if now - last_progress >= 180
                            else "runner_cap"
                            if now - start > job["cap"]
                            else None
                        )
                        if reason and terminate_at is None:
                            child.send_signal(signal.SIGTERM)
                            terminate_at = now
                            event(
                                "terminate_owned_child", id=job["id"], pid=child.pid, reason=reason
                            )
                        if terminate_at is not None and now - terminate_at > 300:
                            child.kill()
                            event("kill_owned_child", id=job["id"], pid=child.pid, reason=reason)
                        time.sleep(1)
                    exit_code = child.wait()
                    count = sum(
                        row.get("type") in ("eval_episode", "train_episode", "train_metric")
                        for row in records(output / "metrics.jsonl")
                    )
                complete = exit_code == 0 and reason is None and completed(output, job)
                result = {
                    "id": job["id"],
                    "pid": child.pid,
                    "exit_code": exit_code,
                    "complete": complete,
                    "reason": reason,
                    "scientific_records": count,
                }
                (job_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                event("child_exit", **result)
                if stop_requested():
                    break
        event("queue_end")
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    run_queue(json.loads(parser.parse_args().config.read_text()))
