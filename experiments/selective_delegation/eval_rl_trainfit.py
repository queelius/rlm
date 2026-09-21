"""Frozen last-RL-checkpoint TRAIN fit, exactly paired to the original batch-1 rollout."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import signal
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import eval_helper
import frozen_execution_probe as frozen
import probe
import rl_planner as rl


def expected_request(prompt, role, seed, cap, root_sha, helper):
    enabled = role != "final"
    return {
        "prompt": prompt,
        "role": role,
        "model": str(rl.evaluation.planner.BASE),
        "adapter_enabled": enabled,
        "adapter_sha256": root_sha
        if role == "root"
        else helper["adapter_binding"]["adapter_model.safetensors"]
        if enabled
        else None,
        "model_instance": "helper" if role == "helper" else "root",
        "helper_contract": helper,
        "seed": seed,
        "temperature": 0.8 if role == "root" else 0.5,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": cap,
        "max_time": 90.0,
    }


def replay_episode(source, case, parent_index, candidate, root_sha, helper):
    """Rebuild actual prompts from predicted answers without modifying immutable source."""
    records = []
    with tempfile.TemporaryDirectory(prefix="rl-trainfit-replay-") as temporary:

        class Replay:
            output = Path(temporary)
            helper_contract = helper

            def call(self, identity, prompt, role, seed, cap):
                call = eval_helper.read(Path(source) / "calls" / (identity + ".json"))
                request = call["request"]
                expected = expected_request(prompt, role, seed, cap, root_sha, helper)
                if (
                    call["call_id"] != identity
                    or probe.runtime.digest(request) != call["request_digest"]
                    or any(request.get(k) != v for k, v in expected.items())
                    or any(
                        call.get(k) != expected[k]
                        for k in (
                            "role",
                            "adapter_enabled",
                            "adapter_sha256",
                            "model",
                            "model_instance",
                        )
                    )
                    or call["input_token_ids"] != request["input_token_ids"]
                    or call["usage"]["prompt_tokens"] != len(call["input_token_ids"])
                    or (
                        call["available"]
                        and call["usage"]["completion_tokens"] != len(call["output_token_ids"])
                    )
                ):
                    raise RuntimeError("saved native request/identity/token mismatch")
                start = eval_helper.read(Path(source) / "starts" / (identity + ".json"))
                if start["request_digest"] != call["request_digest"]:
                    raise RuntimeError("saved start/request mismatch")
                records.append(call)
                if not call["available"]:
                    raise rl.MissingGroup(call.get("error", "unavailable"), call)
                return call

        try:
            row, _ = rl.rollout(Replay(), case, 1, parent_index, candidate)
        except rl.MissingGroup:
            identity = f"u01-{case['id']}-c{candidate}"
            row = eval_helper.read(Path(temporary) / "episodes" / (identity + ".json"))
    path = Path(source) / "episodes" / (row["episode_id"] + ".json")
    if eval_helper.read(path) != row:
        raise RuntimeError("saved episode differs from rollout replay and official grade")
    return row, records


def prepare(source, cases_path):
    source, cases_path = Path(source).resolve(), Path(cases_path).resolve()
    owners = {p.stem.removeprefix("OWNER-") for p in source.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in source.glob("TERMINAL-*.json")}
    if not owners or owners != terminals:
        raise ValueError("source training must have terminal owners before checkpoint selection")
    training = eval_helper.read(source / "PLAN.json")
    terminal_paths = sorted(
        source.glob("TERMINAL-*.json"), key=lambda p: eval_helper.read(p)["ended"]
    )
    terminal = eval_helper.read(terminal_paths[-1])
    if terminal["state"] not in (
        "completed_updates",
        "admission_failed_no_update",
        "stopped_at_update_boundary",
        "failed_or_capped",
        "missing_group_halt",
    ):
        raise ValueError("source stopping rule has not terminated")
    checkpoints = sorted(p for p in source.glob("checkpoint-*") if (p / "COMMIT.json").exists())
    if not checkpoints:
        raise ValueError("no committed RL checkpoint")
    checkpoint = checkpoints[-1]
    state = rl.checkpoint_valid(checkpoint)
    rl.validate_component_resume(state, training)
    if state["step"] != terminal["optimizer_steps"]:
        raise ValueError("last committed checkpoint differs from terminal step")
    baseline = frozen.prepare(source / "batch-0001", cases_path)
    cases = {c["id"]: c for c in map(json.loads, cases_path.read_text().splitlines())}
    root_sha = training["adapter_binding"]["adapter_model.safetensors"]
    outcomes, hashes = [], dict(baseline["source_hashes"])
    for index, cid in enumerate(baseline["case_ids"]):
        for candidate in range(4):
            row, records = replay_episode(
                source / "batch-0001",
                cases[cid],
                index,
                candidate,
                root_sha,
                training["helper_contract"],
            )
            outcomes.append(row)
            for record in records:
                for directory in ("calls", "starts"):
                    path = source / "batch-0001" / directory / (record["call_id"] + ".json")
                    hashes[str(path)] = frozen.sha(path)
    if sum(row["reward"] for row in outcomes) != 45:
        raise ValueError("expected original pre-update batch1 baseline45/64")
    for path in [*source.glob("OWNER-*.json"), *terminal_paths]:
        hashes[str(path)] = frozen.sha(path)
    return {
        "schema": "frozen-rl-trainfit-v1",
        "source": str(source),
        "cases_sha256": frozen.sha(cases_path),
        "case_ids": baseline["case_ids"],
        "checkpoint": str(checkpoint),
        "checkpoint_binding": rl.evaluation.adapter_identity(checkpoint),
        "checkpoint_step": state["step"],
        "checkpoint_selection": "last committed after terminal, not score-selected",
        "terminal": terminal,
        "helper_contract": training["helper_contract"],
        "baseline_adapter_sha256": root_sha,
        "baseline_correct": 45,
        "source_hashes": hashes,
        "planned_episodes": 64,
        "maximum_new_calls": 640,
        "seed_policy": "exact rl_planner.seed_schedule(update=1,parent_index,candidate)",
        "root_temperature": 0.8,
        "downstream_temperature": 0.5,
        "caps": {"root": 128, "helper_total": 384, "final": 128},
        "no_training": True,
        "model": training["model"],
    }


def freeze(model):
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


class FrozenClient(rl.Client):
    """Only adds permanent freezing and no-retry protection to the original native client."""

    def call(self, identity, prompt, role, seed, cap):
        if (self.output / "starts" / (identity + ".json")).exists():
            raise RuntimeError("started attempt already exists; no retry")
        for model in (self.model, self.helper_model):
            freeze(model)
        try:
            return super().call(identity, prompt, role, seed, cap)
        finally:
            for model in (self.model, self.helper_model):
                freeze(model)


def summary(output):
    rows = [eval_helper.read(p) for p in output.glob("episodes/*.json")]
    calls = [eval_helper.read(p) for p in output.glob("calls/*.json")]
    known = {r["call_id"] for r in calls}
    unresolved = [eval_helper.read(p) for p in output.glob("starts/*.json") if p.stem not in known]
    return {
        "planned": 64,
        "recorded": len(rows),
        "missing": 64 - len(rows),
        "unavailable": sum(r["reward"] is None for r in rows),
        "correct_observed": sum(r["reward"] or 0 for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "physical_cost": rl.evaluation.cost(calls + unresolved),
        "unresolved_starts": len(unresolved),
    }


def run(args):
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rl.STOP = False
    if not 0 < args.hours <= 0.5 or args.output.resolve() == args.source.resolve():
        raise ValueError("separate output and at most30 cumulative minutes required")
    plan = prepare(args.source, args.cases)
    plan.update(
        budget_seconds=args.hours * 3600,
        environment={
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
        dependencies={
            str(path): frozen.sha(path)
            for path in (
                Path(__file__),
                Path(rl.__file__),
                Path(rl.train_planner.__file__),
                Path(frozen.__file__),
                Path(eval_helper.__file__),
                Path(rl.evaluation.__file__),
                Path(rl.evaluation.planner.__file__),
                Path(probe.__file__),
                probe.MUSIQUE / "metrics/answer.py",
            )
        },
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if eval_helper.read(output / "PLAN.json") != plan:
            raise ValueError("immutable train-fit PLAN differs")
    else:
        probe.runtime.save(output / "PLAN.json", plan)
    owners = {p.stem.removeprefix("OWNER-") for p in output.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in output.glob("TERMINAL-*.json")}
    if owners != terminals or summary(output)["unresolved_starts"]:
        raise ValueError("unresolved previous owner/request; no implicit retry")
    spent = sum(eval_helper.read(p)["elapsed_seconds"] for p in output.glob("TERMINAL-*.json"))
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + max(0, args.hours * 3600 - spent), lease - 600)
    rl.guard(deadline, 180)
    cases = {c["id"]: c for c in map(json.loads, args.cases.read_text().splitlines())}
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    root = helper = None
    failure = None
    try:
        probe.runtime.save(
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
            rl.STOP = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            plan["model"], local_files_only=True, trust_remote_code=False
        )

        def load(adapter):
            base = AutoModelForCausalLM.from_pretrained(
                plan["model"],
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map={"": "cuda:0"},
            )
            model = PeftModel.from_pretrained(
                base, adapter, is_trainable=False, autocast_adapter_dtype=True
            )
            freeze(model)
            return model

        root, helper = load(plan["checkpoint"]), load(plan["helper_contract"]["adapter"])
        rl.freeze_helper_model(helper, root)
        probe.runtime.save(
            output / f"LOAD-{invocation}.json",
            {
                "checkpoint_binding": plan["checkpoint_binding"],
                "helper_contract": plan["helper_contract"],
                "trainable_parameters": sum(
                    p.numel() for m in (root, helper) for p in m.parameters() if p.requires_grad
                ),
                "optimizer_created": False,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        client = FrozenClient(
            root,
            tokenizer,
            output,
            deadline,
            plan["checkpoint_binding"]["adapter_model.safetensors"],
            helper_contract=plan["helper_contract"],
            helper_model=helper,
        )
        for index, cid in enumerate(plan["case_ids"]):
            for candidate in range(4):
                rl.guard(deadline, 1)
                if (output / "STOP").exists():
                    raise TimeoutError("explicit readout stop")
                identity = f"u01-{cid}-c{candidate}"
                if (output / "episodes" / (identity + ".json")).exists():
                    row, _ = replay_episode(
                        output,
                        cases[cid],
                        index,
                        candidate,
                        client.adapter_sha,
                        client.helper_contract,
                    )
                    if not row["available"]:
                        raise RuntimeError("saved unavailable episode; no retry")
                    continue
                baseline_root = eval_helper.read(
                    args.source / "batch-0001/calls" / (identity + "-root.json")
                )
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": rl.evaluation.planner_prompt(cases[cid])}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if tokens != baseline_root["input_token_ids"]:
                    raise ValueError("root native prompt tokens differ from historical baseline")
                rl.rollout(client, cases[cid], 1, index, candidate)
                probe.campaign.snapshot(output / "SUMMARY.json", summary(output))
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del root, helper
        gc.collect()
        torch.cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summary(output))
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
                "optimizer_steps": 0,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.5)
    run(parser.parse_args())
