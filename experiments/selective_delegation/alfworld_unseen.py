"""Frozen unseen-game inventory using unchanged source026/source032 policy functions."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import alfworld_local_reason as local

ROOT = local.SOURCE_OUTPUT.parent
INPUT = ROOT / "alfworld-unseen-inputs-001/MANIFEST.json"
INPUT_SHA = "4d69b397bb98e77f988af445f7fbdb7310e0369531495f7df4886d575cbcd7a9"
POLICIES = ("flat", "manager_worker", "local_reason")
SEEDS = (2026092195, 2026092196)
save, probe = local.save, local.probe


def play_episode(client, env, job, output, stopping=lambda: False):
    if job["policy"] == "local_reason":
        return local.play_episode(client, env, job, output, stopping)
    if job["policy"] in ("flat", "manager_worker"):
        return local.base.play_episode(client, env, job, output, stopping)
    raise ValueError("unknown frozen policy")


class CheckedBridge(local.base.Bridge):
    def reset(self):
        event = super().reset()
        if event["public"] != self.expected_public:
            raise ValueError("unseen reset differs from frozen public observation")
        return event


def prepare(output, hours):
    if not 0 < hours <= 2:
        raise ValueError("at most two cumulative hours")
    if probe.campaign.sha(INPUT) != INPUT_SHA:
        raise ValueError("frozen unseen panel changed")
    panel = json.loads(INPUT.read_text())
    if (
        panel["split"] != "valid_unseen"
        or panel["game_count"] != 12
        or panel["selection_seed"] != 2026092194
        or panel["reset_failures_retained"] != 0
        or panel["proposed_sampling_seeds"] != list(SEEDS)
    ):
        raise ValueError("frozen unseen inventory contract differs")
    if probe.campaign.sha(local.base.MANIFEST) != local.base.MANIFEST_SHA:
        raise ValueError("original readiness manifest changed")
    readiness = json.loads(local.base.MANIFEST.read_text())
    source032 = ROOT / "alfworld-local-reason-001/PLAN.json"
    original = json.loads(source032.read_text())
    if (
        probe.campaign.sha(source032)
        != "53f6e84f7a0ce9e62564c873b44d6d995591cf9191ec192861e3df598109ec69"
    ):
        raise ValueError("source032 plan changed")
    for source, digest in original["source_sha256"].items():
        if probe.campaign.sha(Path(__file__).with_name(Path(source).name)) != digest:
            raise ValueError("reused frozen policy/client dependency changed")
    for path, digest in panel["source_environment_sha256"].items():
        if probe.campaign.sha(path) != digest:
            raise ValueError("unseen preparation/environment input changed")
    for name, digest in panel["reset_receipt_sha256"].items():
        if probe.campaign.sha(INPUT.parent / name) != digest:
            raise ValueError("frozen public reset receipt changed")
    jobs = []
    for game in panel["games"]:
        if probe.campaign.sha(game["game"]) != game["game_sha256"]:
            raise ValueError("selected game changed")
        for seed in SEEDS:
            for policy in POLICIES:
                jobs.append(
                    {
                        "episode_id": f"game-{game['game_index']:02d}-seed-{seed}-{policy}",
                        "game_index": game["game_index"],
                        "game": game,
                        "seed": seed,
                        "policy": policy,
                    }
                )
    counts = Counter(g["family"] for g in panel["games"])
    if len(jobs) != 72 or len(counts) != 6 or any(n != 2 for n in counts.values()):
        raise ValueError("balanced twelve-game inventory differs")
    model = Path(original["model"])
    if (
        probe.campaign.sha(model / "local-research-manifest.json")
        != original["model_manifest_sha256"]
    ):
        raise ValueError("model manifest changed")
    plan = {
        "schema": "alfworld-unseen-frozen-policies-v1",
        "input_manifest": str(INPUT),
        "input_manifest_sha256": INPUT_SHA,
        "source032_plan_sha256": probe.campaign.sha(source032),
        "split": "valid_unseen",
        "cases": jobs,
        "planned_episodes": 72,
        "policies": list(POLICIES),
        "seeds": list(SEEDS),
        "budget_seconds": hours * 3600,
        "model": str(model),
        "model_manifest_sha256": original["model_manifest_sha256"],
        "adapters": None,
        "training": False,
        "action_limit": 50,
        "token_limit": 2048,
        "request_cap": 128,
        "context_limit": 8192,
        "alfworld_python": str(Path(readiness["environment"]["path"]) / ".venv/bin/python"),
        "data_root": original["data_root"],
        "scene_ids": panel["selected_scene_ids"],
        "smoke_rule": "first game/seed: all three policies execute at least one action; no won gate",
        "interpretation": "Balanced unseen panel, only four scenes. Exact source026/032 policies; "
        "not a new architecture or estimate of the natural benchmark mixture.",
        "source_sha256": {
            str(p.resolve()): probe.campaign.sha(p)
            for p in (
                Path(__file__),
                Path(local.__file__),
                Path(local.base.__file__),
                Path(local.base.native.__file__),
                Path(__file__).with_name("alfworld_bridge.py"),
                Path(local.base.evaluation.__file__),
                Path(probe.__file__),
            )
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        },
    }
    path = Path(output) / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable PLAN differs")
    else:
        save(path, plan)
    return plan, jobs


def summarize(output, plan):
    episodes = [json.loads(p.read_text()) for p in (output / "episodes").glob("*.json")]
    calls = [json.loads(p.read_text()) for p in (output / "calls").glob("*.json")]
    groups = {}
    for policy in POLICIES:
        rows = [r for r in episodes if r["policy"] == policy]
        groups[policy] = {
            "planned": 24,
            "recorded": len(rows),
            "observed": sum(r["observed"] for r in rows),
            "missing_or_unobserved": 24 - sum(r["observed"] for r in rows),
            "won": sum(r["observed"] and r["won"] for r in rows),
            "terminations": dict(Counter(r["termination"] for r in rows)),
            "actions": sum(r["actions"] for r in rows),
            "cost": local.base.evaluation.cost([c for c in calls if c["condition"] == policy]),
        }
    return {
        "plan_sha256": probe.campaign.sha(output / "PLAN.json"),
        "planned_episodes": 72,
        "recorded_episodes": len(episodes),
        "groups": groups,
        "physical_cost": local.base.evaluation.cost(calls),
        "all_slots_recorded": len(episodes) == 72,
        "note": "Twelve balanced unseen games, four scenes; missing is unobserved, not failure.",
    }


def run(args):
    output = args.output.resolve()
    plan, jobs = prepare(output, args.hours)
    if args.prepare_only:
        print(json.dumps({"planned": len(jobs), "model_loaded": False, "output": str(output)}))
        return
    if list(output.glob("OWNER-*.json")):
        raise ValueError("existing owner: no implicit retry/resume")
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600, lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient allocation budget")
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    stopped, failure, model = False, None, None
    try:
        save(
            output / f"OWNER-{invocation}.json",
            {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "started": started,
                "deadline": deadline,
                "allocation_end": lease,
                "source": str(Path(__file__).resolve()),
            },
        )

        def stop(*_):
            nonlocal stopped
            stopped = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        save(
            output / f"LOAD-{invocation}.json",
            {
                "adapters_loaded": False,
                "optimizer_created": False,
                "trainable_parameters": 0,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        client = local.base.BaseClient(model, tokenizer, output, deadline)
        smoke = []

        def stopping():
            return stopped or time.time() >= deadline - 5 or (output / "STOP").exists()

        for index, job in enumerate(jobs):
            if stopping():
                stopped = True
                break
            env = CheckedBridge(
                job["game"],
                plan["alfworld_python"],
                plan["data_root"],
                deadline,
                output / (job["episode_id"] + "-bridge.stderr"),
            )
            env.expected_public = job["game"]["public"]
            try:
                result = play_episode(client, env, job, output, stopping)
            finally:
                env.close()
            probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
            if index < 3:
                smoke.append(result)
            if index == 2:
                passed = all(r["actions"] > 0 for r in smoke)
                save(
                    output / "SMOKE.json",
                    {
                        "passed": passed,
                        "episode_ids": [r["episode_id"] for r in smoke],
                        "rule": plan["smoke_rule"],
                    },
                )
                if not passed:
                    stopped = True
                    break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": stopped,
                "elapsed_seconds": time.time() - started,
                "ended": time.time(),
                "deadline": deadline,
            },
        )
        probe.campaign.snapshot(
            output / "STATUS.json",
            {
                "state": "failed" if failure else "finished_or_capped",
                "failure": failure,
                "updated": time.time(),
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument(
        "--prepare-only", "--validate-only", dest="prepare_only", action="store_true"
    )
    run(parser.parse_args())
