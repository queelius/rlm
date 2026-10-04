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
        and ("env_steps" not in job or ends[-1].get("env_steps") == job["env_steps"])
        and (output / "latest.pt").exists()
    )


def checkpoint_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_parent(job: dict[str, Any], snapshot: Path, source: Path) -> dict[str, Any]:
    """Native terminal evidence suffices here; the driver validates checkpoint payload on load."""
    checkpoint = Path(job["resume"]).resolve()
    if not checkpoint.is_file():
        raise ValueError("Continuation parent checkpoint is missing")
    config_path = checkpoint.parent / "config.json"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    config = json.loads(config_path.read_text())
    recovery = job.get("recovery") is True
    expected = {
        key: value
        for key, value in manifest["reference_configuration"].items()
        if key != "num_train_steps"
    }
    expected.update(
        seed=job["seed"],
        arm=job["arm"],
        eval_freq=job["eval_every"],
        num_eval_episodes=job["eval_episodes"],
        num_train_steps=62500 if recovery else 12500,
    )
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches or config.get("upstream_commit") != manifest["upstream"]["commit"]:
        raise ValueError(f"Continuation parent scientific config mismatch: {mismatches}")
    source_hashes = {
        name: stream_sha256(source / name)
        for name in ("curl_sac.py", "utils.py", "encoder.py", "train.py")
    }
    if source_hashes != config.get("upstream_sha256"):
        raise ValueError("Continuation parent upstream source mismatch")
    if (
        job["steps"] <= 12500
        or config.get("action_repeat") != 8
        or config.get("init_steps") != 1000
    ):
        raise ValueError("Continuation parent must be the prescribed cartpole 100k run")
    native = records(checkpoint.parent / "metrics.jsonl")
    ends = [row for row in native if row.get("type") == "end"]
    episodes = [row for row in native if row.get("type") == "train_episode"]
    saves = [row for row in native if row.get("type") == "checkpoint"]
    extra = {}
    if recovery:
        extra, end, episode, saved = recovery_proof(job, checkpoint, native, config)
    elif not ends or not episodes or not saves or native[-1].get("type") != "end":
        raise ValueError("Continuation parent has no terminal proof")
    else:
        end, episode, saved = ends[-1], episodes[-1], saves[-1]
    if not recovery and (
        end.get("reason") != "completed"
        or end.get("step") != 12500
        or end.get("env_steps") != 100000
        or end.get("updates") != 11500
        or episode.get("step") != 12500
        or episode.get("truncated_by_budget") is not False
        or saved.get("reason") != "completed"
        or saved.get("step") != 12500
        or saved.get("env_steps") != 100000
        or Path(saved.get("path", "")).resolve() != checkpoint
    ):
        raise ValueError("Continuation parent is incomplete or not at a natural 100k boundary")
    identity = checkpoint_identity(checkpoint)
    if saved.get("bytes") != identity["size"]:
        raise ValueError("Continuation parent checkpoint size differs from terminal receipt")
    digest = stream_sha256(checkpoint)
    if checkpoint_identity(checkpoint) != identity:
        raise ValueError("Continuation parent changed during checksum verification")
    return {
        **extra,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_identity": identity,
        "config_path": str(config_path),
        "config_sha256": stream_sha256(config_path),
        "config": config,
        "terminal_proof": {"end": end, "train_episode": episode, "checkpoint": saved},
    }


def recovery_proof(
    job: dict[str, Any], checkpoint: Path, native: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Admit only the latest saved natural boundary of a terminated 500k attempt."""
    origin = checkpoint.parent.parent
    result = json.loads((origin / "result.json").read_text())
    origin_path = origin / "job.json"
    origin_digest = stream_sha256(origin_path)
    receipt = json.loads(origin_path.read_text())
    original = receipt.get("job", {})
    ancestry = receipt.get("parent", {})
    resumes = [row for row in native if row.get("type") in ("start", "resume")]
    if (
        result.get("complete") is not False
        or type(result.get("exit_code")) is not int
        or not original.get("id")
        or result.get("id") != original["id"]
        or not receipt.get("source_hashes")
        or any(
            original.get(key) != job[key] for key in ("arm", "seed", "eval_every", "eval_episodes")
        )
        or original.get("steps") != 62500
        or original.get("env_steps", 500000) != 500000
        or job["steps"] != 62500
        or Path(original.get("resume", "")).resolve()
        != Path(ancestry.get("checkpoint", "")).resolve()
        or ancestry.get("config", {}).get("num_train_steps") != 12500
        or any(
            ancestry.get("config", {}).get(key) != config.get(key)
            for key in ("arm", "seed", "upstream_commit", "upstream_sha256")
        )
        or len(resumes) != 1
        or resumes[0].get("type") != "resume"
        or resumes[0].get("step") != 12500
        or Path(resumes[0].get("resume_path", "")).resolve()
        != Path(original.get("resume", "")).resolve()
    ):
        raise ValueError("Recovery parent has no trustworthy terminated origin job")
    if stream_sha256(origin_path) != origin_digest:
        raise ValueError("Recovery parent origin job changed during preparation")
    ends = [row for row in native if row.get("type") == "end"]
    end = ends[-1] if ends else None
    terminal = native[-1] if native else {}
    stopped = terminal.get("type") == "end" and terminal.get("reason") in (
        "signal_15",
        "signal_2",
        "time_cap",
    )
    if terminal.get("type") != "failure" and not stopped:
        raise ValueError("Recovery parent has no native failure or stopped end")
    saves = [row for row in native if row.get("type") == "checkpoint"]
    if not saves:
        raise ValueError("Recovery parent has no successful checkpoint")
    saved = saves[-1]
    step = saved.get("step")
    episodes = [
        row for row in native if row.get("type") == "train_episode" and row.get("step") == step
    ]
    if (
        type(step) is not int
        or not 12500 <= step < 62500
        or saved.get("env_steps") != step * 8
        or Path(saved.get("path", "")).resolve() != checkpoint
        or len(episodes) != 1
        or episodes[0].get("env_steps") != step * 8
        or episodes[0].get("truncated_by_budget") is not False
        or type(terminal.get("step")) is not int
        or terminal["step"] < step
        or (
            stopped
            and (
                terminal["step"] != step
                or terminal.get("env_steps") != step * 8
                or terminal.get("updates") != step - 1000
            )
        )
    ):
        raise ValueError("Recovery parent latest checkpoint is not a natural saved boundary")
    return (
        {
            "kind": "recovery",
            "restore_step": step,
            "restore_env_steps": step * 8,
            "origin_job_path": str(origin_path),
            "origin_job_sha256": origin_digest,
        },
        end,
        episodes[0],
        saved,
    )


def run_queue(config: dict[str, Any]) -> None:
    root, snapshot = Path(config["campaign_root"]), Path(config["source_snapshot"])
    root.mkdir(parents=True, exist_ok=True)
    gpu_lock = Path(config.get("gpu_lock", root / "GPU.lock")).resolve()
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
                parent = (
                    prepare_parent(job, snapshot, Path(config["source"]))
                    if "resume" in job
                    else None
                )
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
                receipt = {"job": job, "source_hashes": hashes}
                if parent is not None:
                    receipt["parent"] = parent
                (job_root / "job.json").write_text(json.dumps(receipt, indent=2) + "\n")
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
            with gpu_lock.open("a") as lock:
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
                if parent is not None:
                    if (
                        checkpoint_identity(Path(parent["checkpoint"]))
                        != parent["checkpoint_identity"]
                    ):
                        raise ValueError("Continuation parent changed after preparation")
                    options["resume"] = parent["checkpoint"]
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
                    event(
                        "launch",
                        id=job["id"],
                        pid=child.pid,
                        command=command,
                        gpu_lock=str(gpu_lock),
                    )
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
