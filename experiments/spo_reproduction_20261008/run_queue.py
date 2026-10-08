"""Run a bounded native job queue; command success is not scientific replication."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_OWNER = (
    Path(__file__).resolve().parents[1] / "r1_replication_20261007/scoped_session_owner.py"
)


def validate(config: dict) -> None:
    if not Path(config["root"]).is_absolute():
        raise ValueError("root must be absolute")
    if not math.isfinite(config["deadline"]) or config["deadline"] <= 0:
        raise ValueError("an absolute positive deadline is required")
    grace = config.get("cleanup_grace", 10)
    if not math.isfinite(grace) or grace < 0:
        raise ValueError("cleanup_grace must be finite and nonnegative")
    ids = set()
    for job in config["jobs"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["id"]) or job["id"] in ids:
            raise ValueError("job ids must be unique path-safe names")
        ids.add(job["id"])
        if not job["argv"] or not all(isinstance(arg, str) for arg in job["argv"]):
            raise ValueError("argv must be a nonempty string array")
        if not Path(job["cwd"]).is_absolute() or not Path(job["cwd"]).is_dir():
            raise ValueError("cwd must be an existing absolute directory")
        if not math.isfinite(job["seconds"]) or job["seconds"] <= 0:
            raise ValueError("job cap must be finite and positive")
        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in job.get("env", {}).items()
        ):
            raise ValueError("env must contain string names and values")
    if not Path(config.get("scoped_owner", DEFAULT_OWNER)).is_file():
        raise ValueError("scoped_session_owner.py dependency is missing")


def compute_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()})


def ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(config: dict, gpu_pids=compute_pids) -> int:
    validate(config)
    root = Path(config["root"])
    lock_path = Path(config.get("lock", root / "GPU.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    helper = Path(config.get("scoped_owner", DEFAULT_OWNER)).resolve()
    grace = config.get("cleanup_grace", 10)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in config["jobs"]:
            if time.time() >= config["deadline"]:
                return 1
            busy = gpu_pids()
            if busy:
                raise RuntimeError(f"GPU compute PIDs present: {busy}")
            # Reserve the helper's TERM grace, five-second KILL cleanup, and final GPU query.
            cap = min(job["seconds"], config["deadline"] - time.time() - grace - 15)
            if cap <= 0:
                return 1
            root.mkdir(parents=True, exist_ok=True)
            output = root / job["id"]
            output.mkdir()  # Never overwrite a prior attempt, even one without a result receipt.
            env = {**os.environ, **job.get("env", {}), "WANDB_MODE": "offline"}
            record = dict(
                id=job["id"],
                argv=job["argv"],
                cwd=job["cwd"],
                started=time.time(),
                seconds=job["seconds"],
                effective_cap_seconds=cap,
                deadline=config["deadline"],
                owner_pid=os.getpid(),
                owner_start_ticks=ticks(os.getpid()),
                scoped_owner=str(helper),
                scoped_owner_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
            )
            command = [
                sys.executable,
                str(helper),
                "--seconds",
                str(cap),
                "--grace",
                str(grace),
                "--",
                *job["argv"],
            ]
            with (output / "command.log").open("w") as log:
                child = subprocess.Popen(command, cwd=job["cwd"], env=env, stdout=log, stderr=log)
                record.update(pid=child.pid, start_ticks=ticks(child.pid))
                save(output / "START.json", record)
                previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
                for sig in previous:
                    signal.signal(
                        sig, lambda signum, _frame, owned=child: owned.send_signal(signum)
                    )
                try:
                    code = child.wait()
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
            with (output / "command.log").open("rb") as log:
                log.seek(max(0, log.seek(0, 2) - 65536))
                try:
                    cleanup = json.loads(
                        log.read().decode(errors="replace").strip().splitlines()[-1]
                    )
                except (ValueError, IndexError):
                    cleanup = {}
            try:
                busy = gpu_pids()
                gpu_error = None
            except Exception as error:
                busy, gpu_error = [], type(error).__name__
            marker = job.get("success_marker")
            marker_exists = not marker or (Path(job["cwd"]) / marker).exists()
            complete = code == 0 and marker_exists and not busy and not gpu_error
            complete = complete and bool(cleanup) and not cleanup.get("remaining_owned_members", [])
            record.update(
                ended=time.time(),
                exit_code=code,
                complete=complete,
                status="command_succeeded" if complete else "command_incomplete_or_failed",
                success_marker=marker,
                marker_exists=marker_exists,
                scoped_cleanup=cleanup,
                gpu_pids_after=busy,
                gpu_check_error=gpu_error,
            )
            save(output / "result.json", record)
            if not complete:
                return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", "--validate", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cfg.setdefault("deadline", 1791621274)
    try:
        validate(cfg)
        if args.dry_run:
            print(json.dumps(dict(valid=True, jobs=[job["id"] for job in cfg["jobs"]])))
            sys.exit(0)
        sys.exit(run(cfg))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Queue refused/stopped: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
