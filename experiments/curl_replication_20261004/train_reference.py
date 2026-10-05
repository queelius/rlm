"""Thin driver around the pinned authors' CURL agent, not a new learner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint

UPSTREAM_COMMIT = "8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc"


class JsonLogger:
    """Scalar-only upstream logger plus explicit episode/counter records."""

    def __init__(self, path: Path):
        self.handle = path.open("a", buffering=1)

    def record(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps({"time_unix": time.time(), **record}, allow_nan=False) + "\n")

    def log(self, key: str, value: Any, step: int) -> None:
        value = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite training metric {key} at step {step}")
        self.record({"type": "train_metric", "step": step, "metric": key, "value": value})

    def log_histogram(self, *args: Any) -> None:
        pass

    def log_image(self, *args: Any) -> None:
        pass

    def log_param(self, *args: Any) -> None:
        pass

    def close(self) -> None:
        self.handle.close()


def evaluate(
    env: Any,
    agent: Any,
    step: int,
    env_steps: int,
    seeds: list[int],
    image_size: int = 84,
) -> list[dict[str, Any]]:
    rng = capture_rng()
    training = agent.training
    records = []
    try:
        agent.train(False)
        for seed in seeds:
            env.seed(seed)
            obs = env.reset()
            done, episode_return, decisions, actual_steps = False, 0.0, 0, 0
            while not done:
                offset = (obs.shape[-1] - image_size) // 2
                crop = obs[:, offset : offset + image_size, offset : offset + image_size]
                action = agent.select_action(crop)
                obs, reward, done, info = env.step(action)
                episode_return += float(reward)
                decisions += 1
                actual_steps += int(info["env_steps"])
            if not math.isfinite(episode_return):
                raise RuntimeError("Nonfinite evaluation return")
            records.append(
                {
                    "type": "eval_episode",
                    "step": step,
                    "env_steps": env_steps,
                    "eval_seed": seed,
                    "return": episode_return,
                    "eval_decisions": decisions,
                    "eval_env_steps": actual_steps,
                }
            )
    finally:
        agent.train(training)
        restore_rng(rng)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--domain", default="cartpole")
    parser.add_argument("--task", default="swingup")
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument(
        "--steps", type=int, default=12500, help="Training decisions, including warm-up"
    )
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--arm", choices=("curl", "no_curl", "shuffled_curl"), default="curl")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-seconds", type=float, default=900)
    return parser.parse_args()


def build_agent(
    agent_class: Any, config: dict[str, Any], action_shape: tuple[int, ...], device: torch.device
) -> Any:
    kwargs = {
        key: value
        for key, value in config.items()
        if key in inspect.signature(agent_class).parameters
        and key not in {"obs_shape", "action_shape", "device"}
    }
    return agent_class(obs_shape=(9, 84, 84), action_shape=action_shape, device=device, **kwargs)


def configure_arm(agent: Any, replay: Any, config: dict[str, Any]) -> Any:
    """Apply the one declared intervention; reference sampling/math stay untouched."""
    if config["arm"] == "no_curl":
        # Upstream update still samples the positive crop and performs the same SAC updates.
        agent.update_cpc = lambda *unused_args, **unused_kwargs: None
    elif config["arm"] == "shuffled_curl":
        from contrastive_control import ShuffledKeys

        control = ShuffledKeys(agent, replay, config["seed"])
        config["contrastive_control"] = {
            "kind": "deranged_encoded_key_rows",
            "seed": control.seed,
            "generator": "numpy.PCG64",
            "permutation": "uniform_rejection_no_fixed_points",
            "labels": "diagonal_unchanged",
            "encoder_optimizer_steps": "upstream_two",
            "replay_index_tracking": "shadow_global_numpy_first_randint",
        }
        return control
    elif config["arm"] != "curl":
        raise ValueError("Unknown contrastive arm")
    return None


def main() -> None:
    args = parse_args()
    if min(args.steps, args.eval_every, args.eval_episodes, args.action_repeat) <= 0:
        raise ValueError("Budgets and evaluation counts must be positive")
    source = args.source.resolve()
    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Unpinned upstream commit {commit}")
    source_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("curl_sac.py", "utils.py", "encoder.py", "train.py")
    }
    sys.path.insert(0, str(source))
    utils = importlib.import_module("utils")
    Agent = importlib.import_module("curl_sac").CurlSacAgent
    from env_adapter import make_env

    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    config = dict(manifest["reference_configuration"])
    config.update(
        seed=args.seed,
        domain_name=args.domain,
        task_name=args.task,
        action_repeat=args.action_repeat,
        num_train_steps=args.steps,
        eval_freq=args.eval_every,
        num_eval_episodes=args.eval_episodes,
        arm=args.arm,
        upstream_commit=commit,
        upstream_sha256=source_hashes,
        evaluation_seeds=list(range(10000, 10000 + args.eval_episodes)),
        resume_semantics="episode-boundary reset; simulator physics/frame history not serialized",
        training_episode_rng="task and action-space states preserved at boundaries",
        torch_version=torch.__version__,
        python_version=sys.version,
        device=args.device,
        log_interval=100,
        detach_encoder=False,
        max_seconds=args.max_seconds,
        checkpoint_seconds=args.checkpoint_seconds,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "config.json").exists():
        raise ValueError("Output already has a run; choose a fresh directory or --resume")
    utils.set_seed_everywhere(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA training but CUDA is unavailable")
    env = make_env(args.domain, args.task, args.seed, args.action_repeat)
    eval_env = make_env(args.domain, args.task, 10000, args.action_repeat)
    agent = build_agent(Agent, config, env.action_space.shape, device)
    replay = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=config["replay_buffer_capacity"],
        batch_size=config["batch_size"],
        device=device,
        image_size=84,
    )
    control = configure_arm(agent, replay, config)
    counters = {
        "step": 0,
        "env_steps": 0,
        "episode": 0,
        "updates": 0,
        "last_eval_step": -1,
        "episode_boundary": True,
        "eval_env_steps": 0,
        "elapsed_seconds": 0.0,
    }
    if args.resume:
        counters, previous = load_checkpoint(args.resume, agent, replay)
        allowed_changes = {"num_train_steps", "max_seconds", "checkpoint_seconds", "device"}
        changed = {key for key in previous if previous[key] != config.get(key)} - allowed_changes
        if changed:
            raise ValueError(f"Resume config changed scientific inputs: {sorted(changed)}")
        if not counters["episode_boundary"]:
            raise ValueError("Only episode-boundary checkpoints can resume")
        if control is not None:
            control.load_state_dict(counters.get("contrastive_permutation_rng"))
        env.set_rng_state(counters["env_rng"])
    if not (output / "config.json").exists():
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger = JsonLogger(output / "metrics.jsonl")
    logger.record(
        {
            "type": "start" if not args.resume else "resume",
            "step": counters["step"],
            "resume_path": str(args.resume) if args.resume else None,
            "config": config,
        }
    )
    start = time.monotonic()
    previous_elapsed = counters["elapsed_seconds"]
    last_checkpoint = start
    stop = {"reason": None}

    def request_stop(signum: int, _frame: Any) -> None:
        stop["reason"] = f"signal_{signum}"

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def checkpoint(reason: str) -> None:
        nonlocal last_checkpoint
        counters["elapsed_seconds"] = previous_elapsed + time.monotonic() - start
        counters["env_rng"] = env.get_rng_state()
        if control is not None:
            counters["contrastive_permutation_rng"] = control.state_dict()
        before = time.monotonic()
        save_checkpoint(output / "latest.pt", agent, replay, counters, config)
        last_checkpoint = time.monotonic()
        logger.record(
            {
                "type": "checkpoint",
                "step": counters["step"],
                "env_steps": counters["env_steps"],
                "reason": reason,
                "path": str(output / "latest.pt"),
                "seconds": last_checkpoint - before,
                "bytes": (output / "latest.pt").stat().st_size,
            }
        )

    def evaluation() -> None:
        before = time.monotonic()
        records = evaluate(
            eval_env, agent, counters["step"], counters["env_steps"], config["evaluation_seeds"]
        )
        for record in records:
            logger.record(record)
        counters["eval_env_steps"] += sum(record["eval_env_steps"] for record in records)
        counters["last_eval_step"] = counters["step"]
        logger.record(
            {
                "type": "eval_summary",
                "step": counters["step"],
                "env_steps": counters["env_steps"],
                "mean_return": float(np.mean([record["return"] for record in records])),
                "episodes": len(records),
                "seconds": time.monotonic() - before,
            }
        )

    reason = "completed"
    try:
        if counters["last_eval_step"] != counters["step"]:
            evaluation()
        while counters["step"] < args.steps:
            obs = env.reset()
            counters["episode_boundary"] = False
            done, episode_reward, episode_steps, episode_env_steps = False, 0.0, 0, 0
            while not done and counters["step"] < args.steps:
                step = counters["step"]
                if step < config["init_steps"]:
                    action = env.action_space.sample()
                else:
                    with utils.eval_mode(agent):
                        action = agent.sample_action(obs)
                    agent.update(replay, logger, step)
                    if control is not None and step % config["log_interval"] == 0:
                        logger.record(
                            {"type": "contrastive_control", "step": step, **control.diagnostics}
                        )
                    counters["updates"] += 1
                next_obs, reward, done, info = env.step(action)
                if not math.isfinite(float(reward)):
                    raise RuntimeError("Nonfinite training reward")
                # Preserve upstream time-limit bootstrap; env adapter exposes actual truncation.
                timeout = info.get(
                    "TimeLimit.truncated", episode_steps + 1 == env._max_episode_steps
                )
                done_bool = 0.0 if timeout else float(done)
                replay.add(obs, action, reward, next_obs, done_bool)
                obs = next_obs
                counters["step"] += 1
                counters["env_steps"] += int(info["env_steps"])
                episode_steps += 1
                episode_env_steps += int(info["env_steps"])
                episode_reward += float(reward)
                if counters["step"] % args.eval_every == 0:
                    evaluation()
                if time.monotonic() - start >= args.max_seconds:
                    stop["reason"] = stop["reason"] or "time_cap"
            counters["episode"] += 1
            counters["episode_boundary"] = True
            counters["boundary_kind"] = "natural" if done else "budget_truncation"
            logger.record(
                {
                    "type": "train_episode",
                    "step": counters["step"],
                    "env_steps": counters["env_steps"],
                    "episode": counters["episode"],
                    "return": episode_reward,
                    "decisions": episode_steps,
                    "episode_env_steps": episode_env_steps,
                    "truncated_by_budget": not done,
                }
            )
            if stop["reason"]:
                reason = stop["reason"]
                break
            if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                checkpoint("periodic")
        if counters["last_eval_step"] != counters["step"]:
            evaluation()
        checkpoint(reason)
        logger.record(
            {
                "type": "end",
                "reason": reason,
                "step": counters["step"],
                "env_steps": counters["env_steps"],
                "updates": counters["updates"],
                "elapsed_seconds": counters["elapsed_seconds"],
                "eval_env_steps": counters["eval_env_steps"],
            }
        )
    except BaseException as error:
        logger.record(
            {
                "type": "failure",
                "step": counters["step"],
                "error": type(error).__name__,
                "message": str(error),
            }
        )
        raise
    finally:
        env.close()
        eval_env.close()
        logger.close()


if __name__ == "__main__":
    main()
