"""Small TRAIN direct baseline sharing one physical direct call across candidate slots."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import signal
import time
import uuid
from pathlib import Path

import eval_planner as evaluation
import plan_only_probe as runtime
import probe
import training_execution_credit as credit

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
MAX_HOURS = 1 / 3


def unique_jobs(jobs: list[dict]) -> list[dict]:
    """Collapse candidate copies only after proving their saved final seed is identical."""
    grouped = {}
    for job in jobs:
        key = job["case_id"], job["repeat"]
        grouped.setdefault(key, []).append(job)
    unique = []
    for (case_id, repeat), group in sorted(grouped.items()):
        if {job["policy"] for job in group} != {"c0", "c1", "c2", "c3"}:
            raise ValueError("four candidate slots required")
        if len({job["seed"] for job in group}) != 1:
            raise ValueError("saved final seeds must be candidate-independent")
        unique.append({"case_id": case_id, "repeat": repeat, "seed": group[0]["seed"]})
    return unique


def logical_slots(unique: list[dict]) -> list[dict]:
    return [
        {
            "case_id": job["case_id"],
            "repeat": job["repeat"],
            "candidate": candidate,
            "seed": job["seed"],
            "physical_identity": f"{job['case_id']}-s{job['repeat']}-direct",
        }
        for job in unique
        for candidate in range(4)
    ]


def prepare(root: Path, output: Path, hours: float) -> tuple[dict, dict, list[dict]]:
    if not 0 < hours <= MAX_HOURS:
        raise ValueError("maximum 20 minutes required")
    # The execution-credit preparer validates its source only; never let it write
    # its incompatible PLAN into this direct-control target.
    source_plan, cases, source_jobs = credit.prepare(
        root, root / "direct-control-source-audit", hours
    )
    unique = unique_jobs(source_jobs)
    if len(unique) != 80 or len(logical_slots(unique)) != 320:
        raise ValueError("requires 16 parents x five settings and 320 mapped logical slots")
    adapter = Path(source_plan["inert_adapter"])
    plan = {
        "schema": "training-direct-control-v1",
        "source_execution_credit_plan_sha256": probe.campaign.sha(
            root / "training-execution-credit-001/PLAN.json"
        ),
        "parents": source_plan["parents"],
        "settings": 5,
        "candidates": 4,
        "planned_unique_direct_calls": 80,
        "planned_logical_slots": 320,
        "maximum_new_calls": 80,
        "budget_seconds": int(hours * 3600),
        "cases": source_plan["cases"],
        "cases_sha256": source_plan["cases_sha256"],
        "model": source_plan["model"],
        "model_manifest_sha256": source_plan["model_manifest_sha256"],
        "inert_adapter": str(adapter),
        "inert_adapter_binding": source_plan["inert_adapter_binding"],
        "sampling": source_plan["sampling"],
        "source_hashes": source_plan["source_hashes"],
        "collector_sha256": probe.campaign.sha(__file__),
        "policy": "Base full-source direct prompt; no root plan, helper, or adapter-enabled call. "
        "One physical response per parent/setting maps to all four candidate logical slots; "
        "logical slots are correlated and never treated as independent.",
    }
    return plan, cases, unique


def summarize(output: Path, plan: dict) -> dict:
    calls = [runtime.read(path) for path in (output / "calls").glob("*.json")]
    episodes = [runtime.read(path) for path in (output / "episodes").glob("*.json")]
    return {
        "planned_unique_direct_calls": plan["planned_unique_direct_calls"],
        "recorded_unique_direct_calls": len(calls),
        "missing_unique_direct_calls": plan["planned_unique_direct_calls"] - len(calls),
        "planned_logical_slots": plan["planned_logical_slots"],
        "mapped_logical_slots": len(episodes) * 4,
        "status_counts": {
            key: sum(row["status"] == key for row in episodes)
            for key in sorted({row["status"] for row in episodes})
        },
        "physical_cost": evaluation.cost(calls),
    }


def run(args):
    plan, cases, jobs = prepare(args.root.resolve(), args.output.resolve(), args.hours)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime.immutable(output / "PLAN.json", plan)
    if args.prepare_only:
        print(json.dumps({"unique_calls": 80, "logical_slots": 320, "maximum_calls": 80}))
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if list(output.glob("OWNER-*.json")) or any(
        any((output / name).glob("*.json")) for name in ("calls", "episodes")
    ):
        raise ValueError("existing owner or receipts require explicit review")
    deadline = min(
        time.time() + plan["budget_seconds"], int(os.environ["SLURM_JOB_END_TIME"]) - 600
    )
    if (
        deadline < time.time() + 180
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("insufficient bounded allocation or exactly one GPU required")
    import psutil

    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started, failure, model, stopped = (
        uuid.uuid4().hex[:12],
        time.time(),
        None,
        None,
        False,
    )
    probe.runtime.save(
        output / f"OWNER-{invocation}.json",
        {
            "pid": os.getpid(),
            "create_time": psutil.Process().create_time(),
            "started": started,
            "deadline": deadline,
            "allocation_end": int(os.environ["SLURM_JOB_END_TIME"]),
            "source": str(Path(__file__).resolve()),
        },
    )
    try:

        def stop(*_):
            nonlocal stopped
            stopped = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        tokenizer = AutoTokenizer.from_pretrained(
            evaluation.planner.BASE, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            evaluation.planner.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base, plan["inert_adapter"], is_trainable=False, autocast_adapter_dtype=True
        )
        model.eval()
        client = runtime.FinalClient(
            model,
            tokenizer,
            output,
            deadline,
            plan["inert_adapter_binding"]["adapter_model.safetensors"],
        )
        for job in jobs:
            if stopped or time.time() >= deadline - 5 or (output / "STOP").exists():
                stopped = True
                break
            identity = f"{job['case_id']}-s{job['repeat']}-direct"
            call = client.call(
                identity + "-final",
                evaluation.direct_prompt(cases[job["case_id"]]),
                "base",
                "final",
                job["seed"],
                max_new_tokens=128,
            )
            row = {
                **job,
                "episode_id": identity,
                "call_ids": [call["call_id"]],
                "observed": call["available"],
                "status": "scored" if call["available"] else "unavailable",
                "new": probe.grade(
                    call["text"] if call["available"] else "", cases[job["case_id"]]
                ),
            }
            probe.runtime.save(output / "episodes" / f"{identity}.json", row)
            probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
            probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
            if not call["available"]:
                raise RuntimeError("unavailable direct call; no implicit retry")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
        gc.collect()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": stopped,
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
            },
        )
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        probe.campaign.snapshot(
            output / "STATUS.json",
            {
                "state": "failed" if failure else "finished_or_capped",
                "failure": failure,
                "stopped": stopped,
                "updated": time.time(),
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=MAX_HOURS)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
