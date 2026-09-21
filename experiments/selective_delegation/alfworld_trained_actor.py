"""Fixed-checkpoint action-adapter readout on all frozen 035 unseen slots."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import alfworld_unseen as unseen

ROOT = unseen.ROOT
CHECKPOINT = ROOT / "alfworld-action-sft-002/checkpoint-0033"
POLICIES = ("flat", "manager_worker")
SEEDS = unseen.SEEDS
save, probe = unseen.save, unseen.probe


def role_identity(role: str, adapter_sha256: str) -> dict:
    if role not in ("flat", "manager", "worker"):
        raise ValueError("unknown ALFWorld call role")
    enabled = role in ("flat", "worker")
    return {"adapter_enabled": enabled, "adapter_sha256": adapter_sha256 if enabled else None}


@contextlib.contextmanager
def role_adapter_context(model, role: str):
    if role_identity(role, "identity")["adapter_enabled"]:
        yield
        return
    # PEFT re-enables LoRA trainability while restoring adapter state.  This readout has no
    # optimizer, but preserve the explicit frozen-parameter contract after every manager call.
    with model.disable_adapter():
        yield
    for parameter in model.parameters():
        parameter.requires_grad_(False)


class ActionAdapterClient(unseen.local.base.BaseClient):
    """The unchanged native decoder, with adapter state recorded before every request."""

    def __init__(self, model, tokenizer, output, deadline, adapter_sha256):
        super().__init__(model, tokenizer, output, deadline)
        self.adapter_sha256 = adapter_sha256

    def call(self, identity, prompt, policy, role, seed, cap, trimming):
        import torch

        ids = self.ids(prompt)
        identity_fields = role_identity(role, self.adapter_sha256)
        request = {
            "prompt": prompt,
            "input_token_ids": ids,
            "condition": policy,
            "role": role,
            "model": str(unseen.local.base.evaluation.planner.BASE),
            "adapter_checkpoint": str(CHECKPOINT),
            **identity_fields,
            "seed": seed,
            "sampling": {
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": cap,
                "max_time": 90.0,
                "do_sample": True,
            },
        }
        digest, path = probe.runtime.digest(request), self.output / "calls" / (identity + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if (
                saved["request_digest"] != digest
                or probe.runtime.digest(saved["request"]) != digest
            ):
                raise ValueError("saved request differs; no implicit retry")
            return saved
        row = {
            "call_id": identity,
            "request": request,
            "request_digest": digest,
            "condition": policy,
            "role": role,
            "available": False,
            "text": None,
            "input_token_ids": ids,
            "usage": {},
            "trimming": trimming,
            "started": time.time(),
            **identity_fields,
        }
        save(self.output / "starts" / (identity + ".json"), row)
        try:
            remaining = min(90.0, self.deadline - time.time() - 1)
            if remaining <= 0 or len(ids) + cap > unseen.local.base.CONTEXT or not 1 <= cap <= 128:
                raise ValueError("generation deadline/context/cap violated")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            inputs = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            row["effective_max_time"] = remaining
            row["usage"]["prompt_tokens"] = len(ids)
            with role_adapter_context(self.model, role), torch.no_grad():
                generated = self.model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    do_sample=True,
                    temperature=0.5,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=cap,
                    max_time=remaining,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            values = generated[0].detach().cpu().tolist()
            if values[: len(ids)] != ids or len(values) <= len(ids):
                raise ValueError("not a real continuation of saved prompt")
            completion = values[len(ids) :]
            row.update(
                available=True,
                output_token_ids=completion,
                text=self.tokenizer.decode(
                    completion, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
                usage={"prompt_tokens": len(ids), "completion_tokens": len(completion)},
                finish_reason="eos"
                if completion[-1] == self.tokenizer.eos_token_id
                else "length_or_time",
            )
            if not row["text"].strip():
                raise ValueError("empty scientific model response")
            self.returned += 1
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.failed += 1
        row["ended"] = time.time()
        save(path, row)
        return row


def checkpoint_identity() -> dict:
    commit = CHECKPOINT / "COMMIT.json"
    if not commit.exists():
        raise ValueError("fixed action-SFT checkpoint0033 is not available")
    receipt = json.loads(commit.read_text())
    for name, digest in receipt["files"].items():
        if probe.campaign.sha(CHECKPOINT / name) != digest:
            raise ValueError("fixed checkpoint COMMIT mismatch")
    state = json.loads((CHECKPOINT / "STATE.json").read_text())
    if state.get("step") != 33 or state.get("epoch") != 1 or state.get("cursor") != 0:
        raise ValueError("only complete fixed checkpoint0033 is admissible")
    adapter = CHECKPOINT / "adapter_model.safetensors"
    return {
        "commit_sha256": probe.campaign.sha(commit),
        "adapter_sha256": probe.campaign.sha(adapter),
    }


def prepare(output: Path, hours: float):
    if not 0 < hours <= 1.5:
        raise ValueError("at most 90 cumulative minutes")
    if probe.campaign.sha(unseen.INPUT) != unseen.INPUT_SHA:
        raise ValueError("frozen 035 unseen panel changed")
    panel = json.loads(unseen.INPUT.read_text())
    identity = checkpoint_identity()
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
    if len(jobs) != 48:
        raise ValueError("exact twelve games, two seeds and two roles required")
    base_plan = json.loads((ROOT / "alfworld-unseen-001/PLAN.json").read_text())
    expected = [job for job in base_plan["cases"] if job["policy"] in POLICIES]
    if jobs != expected:
        raise ValueError("all exact 035 flat/manager slots required")
    plan = {
        "schema": "alfworld-fixed-action-adapter-readout-v1",
        "baseline_output": str(ROOT / "alfworld-unseen-001"),
        "baseline_plan_sha256": probe.campaign.sha(ROOT / "alfworld-unseen-001/PLAN.json"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_identity": identity,
        "cases": jobs,
        "planned_episodes": 48,
        "policies": list(POLICIES),
        "seeds": list(SEEDS),
        "budget_seconds": hours * 3600,
        "action_limit": 50,
        "token_limit": 2048,
        "request_cap": 128,
        "context_limit": 8192,
        "alfworld_python": base_plan["alfworld_python"],
        "data_root": base_plan["data_root"],
        "role_adapter": {"flat": "enabled", "worker": "enabled", "manager": "disabled_base"},
        "interpretation": "Exposed 035 slots; fixed checkpoint, not selection or fresh "
        "generalization.",
        "source_sha256": {str(Path(__file__).resolve()): probe.campaign.sha(Path(__file__))},
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
    }
    path = Path(output) / "PLAN.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("immutable PLAN differs")
    if not path.exists():
        save(path, plan)
    return plan, jobs


def summarize(output: Path, plan: dict) -> dict:
    episodes = [json.loads(path.read_text()) for path in (output / "episodes").glob("*.json")]
    calls = [json.loads(path.read_text()) for path in (output / "calls").glob("*.json")]
    groups = {}
    for policy in POLICIES:
        rows = [row for row in episodes if row["policy"] == policy]
        groups[policy] = {
            "planned": 24,
            "recorded": len(rows),
            "observed": sum(row["observed"] for row in rows),
            "missing_or_unobserved": 24 - sum(row["observed"] for row in rows),
            "won": sum(row["observed"] and row["won"] for row in rows),
            "cost": unseen.local.base.evaluation.cost(
                [c for c in calls if c["condition"] == policy]
            ),
        }
    return {
        "plan_sha256": probe.campaign.sha(output / "PLAN.json"),
        "planned_episodes": 48,
        "recorded_episodes": len(episodes),
        "groups": groups,
        "physical_cost": unseen.local.base.evaluation.cost(calls),
    }


def run(args):
    output = args.output.resolve()
    plan, jobs = prepare(output, args.hours)
    if args.prepare_only:
        return
    if list(output.glob("OWNER-*.json")):
        raise ValueError("existing owner: no implicit retry")
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lease, started = int(os.environ["SLURM_JOB_END_TIME"]), time.time()
    deadline = min(started + args.hours * 3600, lease - 600)
    if (
        deadline < time.time() + 180
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("requires one usable GPU and allocation margin")
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, stopped, failure, model = uuid.uuid4().hex[:12], False, None, None
    try:
        save(
            output / f"OWNER-{invocation}.json",
            {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "started": started,
                "deadline": deadline,
                "source": str(Path(__file__).resolve()),
            },
        )

        def stop(*_):
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        tokenizer = AutoTokenizer.from_pretrained(
            unseen.local.base.evaluation.planner.BASE,
            local_files_only=True,
            trust_remote_code=False,
        )
        base = AutoModelForCausalLM.from_pretrained(
            unseen.local.base.evaluation.planner.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base, CHECKPOINT, is_trainable=False, autocast_adapter_dtype=True
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        client = ActionAdapterClient(
            model, tokenizer, output, deadline, plan["checkpoint_identity"]["adapter_sha256"]
        )

        def stopping():
            return stopped or time.time() >= deadline - 5

        for job in jobs:
            if stopping() or (output / "STOP").exists():
                stopped = True
                break
            env = unseen.CheckedBridge(
                job["game"],
                plan["alfworld_python"],
                plan["data_root"],
                deadline,
                output / (job["episode_id"] + "-bridge.stderr"),
            )
            env.expected_public = job["game"]["public"]
            try:
                unseen.play_episode(client, env, job, output, stopping)
            finally:
                env.close()
            probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model
        gc.collect()
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": stopped,
                "elapsed_seconds": time.time() - started,
                "ended": time.time(),
                "endpoint": "complete" if not stopped and failure is None else "stopped_or_failed",
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1.5)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
