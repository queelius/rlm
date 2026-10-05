"""Evaluate a completed walker 100k actor on a separate fixed panel; never train."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc"
SOURCE_NAMES = ("curl_sac.py", "utils.py", "encoder.py", "train.py")
ENDPOINT = {"step": 50000, "env_steps": 100000, "updates": 49000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float, default=1200)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def authenticate_parent(source: Path, parent: Path, seeds: list[int]) -> dict[str, Any]:
    config_bytes = (parent / "config.json").read_bytes()
    config = json.loads(config_bytes)
    expected = {
        "domain_name": "walker",
        "task_name": "walk",
        "action_repeat": 2,
        "num_train_steps": 50000,
        "init_steps": 1000,
        "image_size": 84,
        "encoder_type": "pixel",
        "frame_stack": 3,
        "num_eval_episodes": 10,
        "evaluation_seeds": list(range(10000, 10010)),
    }
    require(
        all(config.get(key) == value for key, value in expected.items()),
        "Parent is not the prescribed walker/walk 100k policy",
    )
    require(
        config.get("arm") in ("curl", "no_curl") and type(config.get("seed")) is int,
        "Parent arm or training seed is invalid",
    )
    require(
        not set(seeds).intersection(config["evaluation_seeds"]),
        "Fresh evaluation seeds overlap the primary panel",
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in SOURCE_NAMES
    }
    require(
        commit == config.get("upstream_commit") == UPSTREAM_COMMIT,
        "Parent/source upstream commit is not pinned",
    )
    require(hashes == config.get("upstream_sha256"), "Parent/source module hashes differ")
    metrics_bytes = (parent / "metrics.jsonl").read_bytes()
    rows = [json.loads(line) for line in metrics_bytes.splitlines()]
    starts = [row for row in rows if row.get("type") in ("start", "resume")]
    require(
        len(starts) == 1
        and starts[0].get("type") == "start"
        and starts[0].get("step") == 0
        and starts[0].get("config") == config,
        "Parent is not a single authenticated fresh training run",
    )
    end = rows[-1] if rows else {}
    require(
        end.get("type") == "end"
        and end.get("reason") == "completed"
        and all(end.get(key) == value for key, value in ENDPOINT.items()),
        "Parent does not have a completed fixed 100k endpoint",
    )
    episodes = [row for row in rows if row.get("type") == "train_episode"]
    require(
        bool(episodes)
        and episodes[-1].get("step") == 50000
        and episodes[-1].get("env_steps") == 100000
        and episodes[-1].get("truncated_by_budget") is False,
        "Parent checkpoint is not at a natural episode boundary",
    )
    evaluations = [
        row for row in rows if row.get("type") == "eval_episode" and row.get("step") == 50000
    ]
    require(
        len(evaluations) == 10
        and sorted(row.get("eval_seed", -1) for row in evaluations) == config["evaluation_seeds"]
        and all(
            row.get("env_steps") == 100000
            and row.get("eval_decisions") == 500
            and row.get("eval_env_steps") == 1000
            and math.isfinite(float(row["return"]))
            for row in evaluations
        ),
        "Parent lacks the complete primary terminal panel",
    )
    checkpoint = parent / "latest.pt"
    saves = [row for row in rows if row.get("type") == "checkpoint"]
    checkpoint_identity = identity(checkpoint)
    require(
        bool(saves)
        and saves[-1].get("reason") == "completed"
        and saves[-1].get("step") == 50000
        and saves[-1].get("env_steps") == 100000
        and Path(saves[-1].get("path", "")).resolve() == checkpoint
        and saves[-1].get("bytes") == checkpoint_identity["size"],
        "Parent latest checkpoint differs from terminal receipt",
    )
    checkpoint_sha256 = stream_sha256(checkpoint)  # Exactly one streaming hash of the large file.
    require(identity(checkpoint) == checkpoint_identity, "Parent checkpoint changed while hashing")
    return {
        "run": str(parent),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity": checkpoint_identity,
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "source": str(source),
        "source_hashes": hashes,
        "terminal_proof": {"end": end, "train_episode": episodes[-1], "checkpoint": saves[-1]},
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Even failed panels are immutable attempts.
    handle = (output / "metrics.jsonl").open("x", buffering=1)
    started = time.monotonic()
    completed_episodes = []
    env = None
    old_handlers = {}
    stopped = {"reason": None}

    def record(value: dict[str, Any]) -> None:
        handle.write(json.dumps({"time_unix": time.time(), **value}, allow_nan=False) + "\n")

    def stop(signum: int, _frame: Any) -> None:
        stopped["reason"] = f"signal_{signum}"

    def result(reason: str) -> dict[str, Any]:
        complete = reason == "completed" and len(completed_episodes) == args.episodes
        value = {
            "evaluation_only": True,
            "complete": complete,
            "reason": reason,
            "episodes_requested": args.episodes,
            "episodes_completed": len(completed_episodes),
            "training_steps": 0,
            "optimizer_updates": 0,
            "eval_decisions": sum(row["eval_decisions"] for row in completed_episodes),
            "eval_env_steps": sum(row["eval_env_steps"] for row in completed_episodes),
            "elapsed_seconds": time.monotonic() - started,
        }
        if complete:
            value["mean_return"] = sum(row["return"] for row in completed_episodes) / args.episodes
        (output / "result.json").write_text(json.dumps(value, indent=2) + "\n")
        return value

    try:
        require(
            args.episodes > 0
            and args.seed_start >= 0
            and math.isfinite(args.max_seconds)
            and args.max_seconds > 0,
            "Panel count, seeds and time cap must be valid positive budgets",
        )
        seeds = list(range(args.seed_start, args.seed_start + args.episodes))
        parent = authenticate_parent(args.source.resolve(), args.parent_run.resolve(), seeds)
        versions = {}
        for package in ("torch", "numpy", "dm-control", "mujoco"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        receipt = {
            "evaluation_only": True,
            "parent": parent,
            "evaluation_source_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "evaluate_frozen.py",
                    "env_adapter.py",
                    "train_reference.py",
                    "checkpoint.py",
                )
            },
            "runtime": {"python_version": sys.version, "package_versions": versions},
            "evaluation_seeds": seeds,
            "device": args.device,
            "max_seconds": args.max_seconds,
            "replay_buffer_constructed": False,
            "optimizer_state_restored": False,
            "checkpoint_load_limitation": (
                "Full trusted payload transiently deserializes replay arrays; "
                "no replay buffer allocation/copy"
            ),
            "policy": "frozen actor, deterministic action, center crop 84x84",
        }
        (output / "config.json").write_text(json.dumps(receipt, indent=2) + "\n")
        record({"type": "start", **receipt})
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, stop)

        import torch
        from train_reference import build_agent, evaluate

        # Campaign-local trusted pickle. Never call training load_checkpoint or allocate replay.
        payload = torch.load(parent["checkpoint"], map_location="cpu", weights_only=False)
        require(payload.get("schema_version") == 1, "Unsupported checkpoint schema")
        require(payload.get("config") == parent["config"], "Checkpoint scientific config differs")
        counters = payload["counters"]
        require(
            all(counters.get(key) == value for key, value in ENDPOINT.items())
            and counters.get("episode_boundary") is True
            and counters.get("boundary_kind") == "natural"
            and counters.get("last_eval_step") == 50000
            and counters.get("episode") == parent["terminal_proof"]["train_episode"].get("episode"),
            "Checkpoint counters do not match the completed natural endpoint",
        )
        actor_state = payload["modules"]["actor"]
        del payload  # Release deserialized replay/optimizer arrays before constructing the agent.
        require(
            identity(Path(parent["checkpoint"])) == parent["checkpoint_identity"],
            "Parent checkpoint changed during deserialization",
        )
        device = torch.device(args.device)
        require(device.type != "cuda" or torch.cuda.is_available(), "Requested CUDA is unavailable")
        sys.path.insert(0, str(args.source.resolve()))
        Agent = importlib.import_module("curl_sac").CurlSacAgent
        from env_adapter import make_env

        env = make_env("walker", "walk", seeds[0], 2)
        agent = build_agent(Agent, parent["config"], env.action_space.shape, device)
        agent.actor.load_state_dict(actor_state)
        del actor_state
        reason = "completed"
        for seed in seeds:
            if stopped["reason"] or time.monotonic() - started >= args.max_seconds:
                reason = stopped["reason"] or "time_cap"
                break
            with torch.no_grad():
                episode = evaluate(env, agent, 50000, 100000, [seed], image_size=84)[0]
            record(episode)  # Flush each real episode, not an entire 50-seed batch.
            completed_episodes.append(episode)
            if stopped["reason"] or time.monotonic() - started >= args.max_seconds:
                reason = stopped["reason"] or "time_cap"
                break
        record({"type": "end", **result(reason)})
    except BaseException as error:
        record(
            {
                "type": "failure",
                "error": type(error).__name__,
                "message": str(error),
                "episodes_completed": len(completed_episodes),
            }
        )
        result("failure")
        raise
    finally:
        if env is not None:
            env.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        handle.close()


if __name__ == "__main__":
    main()
