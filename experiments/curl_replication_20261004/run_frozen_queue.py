"""Serial owner for the declared walker fresh-start panels; no checkpoint loading here."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

POLL_SECONDS = 1.0
PREPARATION_SECONDS = 180.0
FIRST_EPISODE_SECONDS = 90.0
STALL_SECONDS = 180.0
GRACE_SECONDS = 120.0
SOURCE_FILES = (
    "evaluate_frozen.py",
    "env_adapter.py",
    "train_reference.py",
    "checkpoint.py",
    "requirements.lock.txt",
)


def records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Polling may see a partly written final line; final validation is strict.
    return rows


def parent_gate(campaign: Path) -> list[dict[str, Any]] | None:
    events = records(campaign / "queue.jsonl")
    if not events or events[-1].get("type") != "queue_end":
        return None
    parents = []
    for seed in (123, 456, 789):
        for arm in ("curl", "no_curl"):
            job_id = f"{arm}-seed{seed}-100k-v1"
            root, expected = (
                campaign / job_id,
                {
                    "arm": arm,
                    "seed": seed,
                    "domain_name": "walker",
                    "task_name": "walk",
                    "action_repeat": 2,
                    "num_train_steps": 50000,
                    "init_steps": 1000,
                },
            )
            config = json.loads((root / "run" / "config.json").read_text())
            receipt = json.loads((root / "result.json").read_text())
            rows = [
                json.loads(line)
                for line in (root / "run" / "metrics.jsonl").read_text().splitlines()
            ]
            end = rows[-1] if rows else {}
            episodes = [row for row in rows if row.get("type") == "train_episode"]
            saves = [row for row in rows if row.get("type") == "checkpoint"]
            checkpoint = root / "run" / "latest.pt"
            if not (
                all(config.get(key) == value for key, value in expected.items())
                and receipt.get("id") == job_id
                and receipt.get("complete") is True
                and type(receipt.get("exit_code")) is int
                and receipt["exit_code"] == 0
                and receipt.get("reason") is None
                and end.get("type") == "end"
                and end.get("reason") == "completed"
                and end.get("step") == 50000
                and end.get("env_steps") == 100000
                and end.get("updates") == 49000
                and episodes
                and saves
                and episodes[-1].get("step") == 50000
                and episodes[-1].get("env_steps") == 100000
                and episodes[-1].get("truncated_by_budget") is False
                and saves[-1].get("reason") == "completed"
                and saves[-1].get("step") == 50000
                and saves[-1].get("env_steps") == 100000
                and Path(saves[-1].get("path", "")).resolve() == checkpoint
                and checkpoint.is_file()
                and saves[-1].get("bytes") == checkpoint.stat().st_size
            ):
                raise ValueError(f"Original completed queue has invalid parent {job_id}")
            parents.append(
                {
                    "id": f"{arm}-seed{seed}-fresh-starts-v1",
                    "arm": arm,
                    "seed": seed,
                    "parent_run": str(root / "run"),
                    "parent_receipt": receipt,
                    "parent_terminal": end,
                }
            )
    return parents


def complete_panel(output: Path, job: dict[str, Any]) -> bool:
    try:
        config = json.loads((output / "config.json").read_text())
        result = json.loads((output / "result.json").read_text())
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        episodes = [row for row in rows if row.get("type") == "eval_episode"]
        expected = {
            "evaluation_only": True,
            "complete": True,
            "reason": "completed",
            "episodes_requested": 50,
            "episodes_completed": 50,
            "training_steps": 0,
            "optimizer_updates": 0,
            "eval_decisions": 25000,
            "eval_env_steps": 50000,
        }
        end = rows[-1] if rows else {}
        parent_config = config["parent"]["config"]
        if not (
            config.get("evaluation_seeds") == list(range(20000, 20050))
            and Path(config["parent"]["run"]).resolve() == Path(job["parent_run"])
            and all(
                parent_config.get(key) == value
                for key, value in {
                    "arm": job["arm"],
                    "seed": job["seed"],
                    "domain_name": "walker",
                    "task_name": "walk",
                    "action_repeat": 2,
                    "num_train_steps": 50000,
                }.items()
            )
            and end.get("type") == "end"
            and all(
                result.get(key) == value and end.get(key) == value
                for key, value in expected.items()
            )
            and len(episodes) == 50
            and [row.get("eval_seed") for row in episodes] == list(range(20000, 20050))
            and not any(row.get("type") == "failure" for row in rows)
            and all(
                row.get("step") == 50000
                and row.get("env_steps") == 100000
                and row.get("eval_decisions") == 500
                and row.get("eval_env_steps") == 1000
                and math.isfinite(float(row["return"]))
                for row in episodes
            )
        ):
            return False
        mean = sum(float(row["return"]) for row in episodes) / 50
        return all(
            math.isclose(float(record["mean_return"]), mean, rel_tol=0, abs_tol=1e-9)
            for record in (result, end)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_queue(config: dict[str, Any]) -> dict[str, Any]:
    declaration_path = Path(config["declaration"]).resolve()
    declaration_bytes = declaration_path.read_bytes()
    declaration = json.loads(declaration_bytes)
    fixed = {
        "task": {"domain": "walker", "task": "walk", "action_repeat": 2},
        "training_seeds": [123, 456, 789],
        "arms": ["curl", "no_curl"],
        "parent_env_steps": 100000,
        "parent_decisions": 50000,
        "parent_updates": 49000,
        "evaluation_seed_start": 20000,
        "evaluation_episodes_per_policy": 50,
        "additional_training_steps": 0,
        "per_policy_cap_seconds": 1200,
        "batch_cap_seconds": 7200,
    }
    if any(declaration.get(key) != value for key, value in fixed.items()):
        raise ValueError("Configuration changes the declared fixed panel")
    root = Path(config.get("root", declaration["output_campaign"])).resolve()
    campaign = Path(config.get("parent_campaign", declaration["parent_campaign"])).resolve()
    gpu_lock = Path(config.get("gpu_lock", declaration["gpu_lock"])).resolve()
    deadline = min(float(declaration["allocation_deadline_epoch"]), 1791387366)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "owner.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another frozen batch owner holds this root") from error
        if (root / "queue.jsonl").exists() or (root / "config.json").exists():
            raise ValueError("Existing frozen batch attempt is immutable; no automatic retries")
        stopped = {"signal": None}
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous_handlers:
            signal.signal(sig, lambda signum, frame: stopped.update(signal=signum))

        def event(kind: str, **fields: Any) -> None:
            with (root / "queue.jsonl").open("a") as handle:
                handle.write(
                    json.dumps({"type": kind, "time_unix": time.time(), **fields}, allow_nan=False)
                    + "\n"
                )

        def stop_reason(batch_start: float | None = None) -> str | None:
            if stopped["signal"]:
                return f"signal_{stopped['signal']}"
            if (root / "STOP").exists():
                return "stop"
            if time.time() + GRACE_SECONDS >= deadline:
                return "allocation_deadline"
            if batch_start is not None and time.monotonic() - batch_start >= 7200 - GRACE_SECONDS:
                return "batch_cap"
            return None

        completed = 0
        reason = None
        child = None

        @contextmanager
        def owning_gpu_lock():
            # An exception must not hand the GPU to another owner while our child is alive.
            with gpu_lock.open("a") as lock:
                try:
                    yield lock
                finally:
                    if child is not None:
                        if child.poll() is None:
                            with suppress(ProcessLookupError):
                                os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=GRACE_SECONDS)
                            except subprocess.TimeoutExpired:
                                with suppress(ProcessLookupError):
                                    os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                        else:
                            child.wait()
                        receipt_path = root / job["id"] / "result.json"
                        if not receipt_path.exists():
                            receipt = {
                                "id": job["id"],
                                "pid": child.pid,
                                "exit_code": child.returncode,
                                "complete": False,
                                "reason": "owner_exception",
                            }
                            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                            event("child_exit", **receipt)

        try:
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "config": config,
                        "declaration": declaration,
                        "declaration_sha256": hashlib.sha256(declaration_bytes).hexdigest(),
                        "owner_pid": os.getpid(),
                        "parent_campaign": str(campaign),
                        "gpu_lock": str(gpu_lock),
                    },
                    indent=2,
                )
                + "\n"
            )
            event("owner_start", pid=os.getpid(), mode="waiting", gpu_lock=str(gpu_lock))
            parents = None
            event("waiting_parents", parent_campaign=str(campaign))
            while parents is None and not (reason := stop_reason()):
                parents = parent_gate(campaign)
                if parents is None:
                    time.sleep(POLL_SECONDS)
            if parents is not None:
                code = root / "code"
                code.mkdir()
                snapshot = Path(config["snapshot"]).resolve()
                for name in SOURCE_FILES:
                    shutil.copyfile(snapshot / name, code / name)
                hashes = {
                    name: hashlib.sha256((code / name).read_bytes()).hexdigest()
                    for name in SOURCE_FILES
                }
                event("source", snapshot=str(snapshot), source_hashes=hashes)
                with owning_gpu_lock() as lock:
                    event("waiting_gpu_lock", gpu_lock=str(gpu_lock))
                    while not (reason := stop_reason()):
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            time.sleep(POLL_SECONDS)
                    if reason is None:
                        batch_start = time.monotonic()  # Parent/GPU waiting consumes no batch cap.
                        event("batch_start", pid=os.getpid())
                        for job in parents:
                            if reason := stop_reason(batch_start):
                                break
                            job_root = root / job["id"]
                            job_root.mkdir()
                            output = job_root / "run"
                            (job_root / "job.json").write_text(
                                json.dumps(
                                    {
                                        "job": job,
                                        "evaluation_seeds": list(range(20000, 20050)),
                                        "source_hashes": hashes,
                                        "declaration_sha256": hashlib.sha256(
                                            declaration_bytes
                                        ).hexdigest(),
                                    },
                                    indent=2,
                                )
                                + "\n"
                            )
                            event(
                                "job",
                                id=job["id"],
                                arm=job["arm"],
                                seed=job["seed"],
                                parent_run=job["parent_run"],
                            )
                            command = [
                                config["python"],
                                str(code / "evaluate_frozen.py"),
                                "--source",
                                str(Path(config["source"]).resolve()),
                                "--parent-run",
                                job["parent_run"],
                                "--output",
                                str(output),
                                "--seed-start",
                                "20000",
                                "--episodes",
                                "50",
                                "--max-seconds",
                                "1200",
                                "--device",
                                config.get("device", "cuda"),
                            ]
                            environment = {
                                **os.environ,
                                "MUJOCO_GL": "egl",
                                "PYOPENGL_PLATFORM": "egl",
                                "OMP_NUM_THREADS": "4",
                                "MKL_NUM_THREADS": "4",
                                "OPENBLAS_NUM_THREADS": "4",
                            }
                            with (job_root / "console.log").open("x") as console:
                                child = subprocess.Popen(
                                    command,
                                    stdout=console,
                                    stderr=subprocess.STDOUT,
                                    env=environment,
                                    start_new_session=True,
                                )
                                launched = time.monotonic()
                                event("launch", id=job["id"], pid=child.pid, command=command)
                                (
                                    seen,
                                    evaluation_start,
                                    last_progress,
                                    terminate_at,
                                    child_reason,
                                ) = 0, None, None, None, None
                                killed = False
                                while child.poll() is None:
                                    rows = records(output / "metrics.jsonl")
                                    now = time.monotonic()
                                    if evaluation_start is None and any(
                                        r.get("type") == "start" for r in rows
                                    ):
                                        evaluation_start = now
                                        event(
                                            "evaluation_start",
                                            id=job["id"],
                                            pid=child.pid,
                                            seconds=now - launched,
                                        )
                                    episodes = [r for r in rows if r.get("type") == "eval_episode"]
                                    for row in episodes[seen:]:
                                        event(
                                            "actual_return",
                                            id=job["id"],
                                            pid=child.pid,
                                            **{
                                                k: v
                                                for k, v in row.items()
                                                if k not in ("type", "time_unix")
                                            },
                                        )
                                    if len(episodes) > seen:
                                        seen, last_progress = len(episodes), now
                                    child_reason = child_reason or stop_reason(batch_start)
                                    child_reason = child_reason or (
                                        "child_failure"
                                        if any(r.get("type") == "failure" for r in rows)
                                        else "policy_cap"
                                        if now - launched >= 1200 - GRACE_SECONDS
                                        else "preparation_stalled"
                                        if evaluation_start is None
                                        and now - launched >= PREPARATION_SECONDS
                                        else "first_episode_stalled"
                                        if evaluation_start is not None
                                        and seen == 0
                                        and now - evaluation_start >= FIRST_EPISODE_SECONDS
                                        else "evaluation_stalled"
                                        if last_progress is not None
                                        and now - last_progress >= STALL_SECONDS
                                        else None
                                    )
                                    if child_reason and terminate_at is None:
                                        terminate_at = now
                                        event(
                                            "terminate_owned_child",
                                            id=job["id"],
                                            pid=child.pid,
                                            reason=child_reason,
                                        )
                                        with suppress(ProcessLookupError):
                                            os.killpg(child.pid, signal.SIGTERM)
                                    if (
                                        terminate_at is not None
                                        and now - terminate_at >= GRACE_SECONDS
                                        and not killed
                                    ):
                                        killed = True
                                        event(
                                            "kill_owned_child",
                                            id=job["id"],
                                            pid=child.pid,
                                            reason=child_reason,
                                        )
                                        with suppress(ProcessLookupError):
                                            os.killpg(child.pid, signal.SIGKILL)
                                    time.sleep(POLL_SECONDS)
                                exit_code = child.wait()
                                final_episodes = [
                                    r
                                    for r in records(output / "metrics.jsonl")
                                    if r.get("type") == "eval_episode"
                                ]
                                for row in final_episodes[seen:]:
                                    event(
                                        "actual_return",
                                        id=job["id"],
                                        pid=child.pid,
                                        **{
                                            k: v
                                            for k, v in row.items()
                                            if k not in ("type", "time_unix")
                                        },
                                    )
                            valid = (
                                exit_code == 0
                                and child_reason is None
                                and complete_panel(output, job)
                            )
                            receipt = {
                                "id": job["id"],
                                "pid": child.pid,
                                "exit_code": exit_code,
                                "complete": valid,
                                "reason": child_reason or (None if valid else "invalid_completion"),
                                "episodes": len(final_episodes),
                                "seconds": time.monotonic() - launched,
                            }
                            (job_root / "result.json").write_text(
                                json.dumps(receipt, indent=2) + "\n"
                            )
                            event("child_exit", **receipt)
                            completed += int(valid)
                            child = None
                            if reason := stop_reason(batch_start):
                                break
            summary = {
                "complete": completed == 6,
                "policies_completed": completed,
                "policies_requested": 6,
                "reason": reason or ("completed" if completed == 6 else "incomplete"),
            }
            event("queue_end", **summary)
            return summary
        except BaseException as error:
            event("failure", error=type(error).__name__, message=str(error))
            event("queue_end", complete=False, policies_completed=completed, reason="failure")
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    summary = run_queue(json.loads(parser.parse_args().config.read_text()))
    raise SystemExit(0 if summary["complete"] else 1)
