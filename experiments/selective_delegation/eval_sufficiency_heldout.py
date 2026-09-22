"""Held32 paired readout of warm joint32 and actual committed RL/SFT endpoints."""

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
from pathlib import Path

import eval_sufficiency as adapter_runtime
import rl_sufficiency as training
import sufficiency_probe as baseline

native, panel = baseline.native, baseline.panel
sha = panel.sha256
CONDITIONS = ("warm_joint32", "rl_terminal", "matched_sft_terminal")
CASES_SHA = "52d4c74fedbb89b3b6b56d26538d95d2d512445d0bf7c52883d765c7b1195150"


def read(path):
    return json.loads(path.read_text())


def endpoint_identity(output, mode, warmstart):
    """Authenticate last committed boundary, including a naturally capped endpoint."""
    import psutil

    owners = sorted(output.glob("OWNER-*.json"))
    if not owners:
        raise ValueError("training owner absent")
    terminals = []
    for path in owners:
        owner = read(path)
        terminal = path.with_name(path.name.replace("OWNER-", "TERMINAL-"))
        if not terminal.exists():
            raise ValueError("training owner not terminal")
        try:
            process = psutil.Process(owner["pid"])
            if (
                abs(process.create_time() - owner["create_time"]) < 0.01
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                raise ValueError("training owner still live")
        except psutil.NoSuchProcess:
            pass
        receipt = read(terminal)
        if receipt.get("failure") or receipt.get("stopped"):
            raise ValueError("failed/interrupted training needs explicit review")
        terminals.append((receipt, terminal))
    plan, summary = read(output / "PLAN.json"), read(output / "SUMMARY.json")
    if plan["mode"] != mode or plan["warmstart"] != warmstart:
        raise ValueError("training mode/warmstart differs")
    plan_hash = sha(output / "PLAN.json")
    boundaries = []
    for directory in sorted((output / "boundaries").glob("sample-*")):
        b = read(directory / "BOUNDARY.json")
        cp = Path(b["checkpoint"])
        if cp.parent.resolve() != directory.resolve():
            raise ValueError("checkpoint escaped boundary")
        commit, state = read(cp / "COMMIT.json"), read(cp / "STATE.json")
        if (
            sha(cp / "COMMIT.json") != b["commit_sha256"]
            or sha(cp / "STATE.json") != commit["files"]["STATE.json"]
            or b["state"] != state
            or state["sample_cursor"] != len(boundaries)
            or state["step"] != commit["step"]
            or state["plan_sha256"] != plan_hash
        ):
            raise ValueError("boundary provenance/order differs")
        training.validate_boundary_state(state, boundaries[-1]["state"] if boundaries else None)
        boundaries.append(b)
    if not boundaries:
        raise ValueError("no committed boundary")
    last = boundaries[-1]
    checkpoint, state = Path(last["checkpoint"]), last["state"]
    terminal, terminal_path = max(terminals, key=lambda t: t[0]["sample_cursor"])
    if (
        terminal["endpoint"] != str(checkpoint)
        or summary["endpoint"] != str(checkpoint)
        or terminal["step"] != state["step"]
        or terminal["sample_cursor"] != state["sample_cursor"]
        or summary["actual_optimizer_steps"] != state["step"]
        or summary["committed_sampled_blocks"] != state["sample_cursor"]
        or summary.get("failure")
        or terminal["endpoint_selection"] != "last committed boundary, never held score"
    ):
        raise ValueError("terminal/summary is not last committed endpoint")
    if mode == "sft_control" and not summary.get("matched_control_complete"):
        raise ValueError("matched SFT control not complete")
    commit = read(checkpoint / "COMMIT.json")
    required = (
        "STATE.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng.pt",
    )
    files = {name: sha(checkpoint / name) for name in required}
    if any(commit["files"].get(name) != digest for name, digest in files.items()):
        raise ValueError("checkpoint file differs from commit")
    return {
        "path": str(checkpoint),
        "files": files,
        "commit_sha256": sha(checkpoint / "COMMIT.json"),
        "training_plan_sha256": plan_hash,
        "training_output": str(output),
        "mode": mode,
        "step": state["step"],
        "sample_cursor": state["sample_cursor"],
        "boundaries": boundaries,
        "terminal_sha256": sha(terminal_path),
        "summary_sha256": sha(output / "SUMMARY.json"),
        "training_plan": plan,
    }


def skip_reason(identity):
    return (
        "zero RL optimizer updates; no duplicate triple readout" if identity["step"] == 0 else None
    )


def jobs(cases):
    return [
        {
            "episode_id": f"{c['id']}-{seed}",
            "case_id": c["id"],
            "parent_id": c["parent_id"],
            "seed": seed,
        }
        for c in cases
        for seed in baseline.SEEDS
    ]


def activate(model, condition):
    model.set_adapter(condition)
    # PEFT set_adapter may enable gradients; every readout parameter stays frozen.
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def save_plan(path, plan):
    if path.exists():
        if read(path) != plan:
            raise ValueError("immutable PLAN differs")
    else:
        native.save(path, plan)


def validate_panel(cases_path, manifest, profile=None):
    expected = CASES_SHA if profile is None else profile["cases_sha256"]
    seed = 2026092198 if profile is None else profile["selection_seed"]
    if sha(cases_path) != expected or manifest["cases_sha256"] != expected:
        raise ValueError("fixed paired32 panel differs")
    if manifest["selection_seed"] != seed:
        raise ValueError("selection differs")
    if profile is not None and (
        sha(cases_path.with_name("MANIFEST.json")) != profile["manifest_sha256"]
        or manifest["component_cluster_count"] != profile["component_cluster_count"]
    ):
        raise ValueError("panel manifest/component inventory differs")
    metrics = manifest["official_metric_sha256"] if profile is None else profile["metric_sha256"]
    for path, digest in metrics.items():
        if sha(Path(path)) != digest:
            raise ValueError("official grader changed")
    return expected


class AdditiveDoseMismatch(ValueError):
    """Authenticated terminal is not the fixed eight-update comparison dose."""


def validate_additive_control(product, additive):
    rp, ap = product["training_plan"], additive["training_plan"]
    if (
        rp.get("reward_objective", "product") != "product"
        or rp.get("estimator", "diagonal") != "diagonal"
        or ap.get("reward_objective") != "additive"
        or ap.get("estimator") != "diagonal"
    ):
        raise ValueError("explicit product versus additive diagonal objectives required")
    common = (
        "mode",
        "cases_sha256",
        "input_manifest_sha256",
        "parent_blocks",
        "model",
        "warmstart",
        "base_manifest_sha256",
        "official_metric_sha256",
        "seed",
        "learning_rate",
        "fresh_optimizer",
        "weight_decay",
        "gradient_clip",
        "microbatch",
        "candidate_pairs_per_parent",
        "pair_denominator",
        "max_sampled_blocks",
        "max_calls",
        "max_consecutive_zero_blocks",
        "budget_seconds",
        "sampling_temperature",
        "sampling_top_p",
        "sampling_top_k",
        "max_new_tokens",
        "max_context",
        "environment_lock_sha256",
    )
    if any(rp[key] != ap[key] for key in common):
        raise ValueError("additive control common training contract differs")
    if (product["step"], product["sample_cursor"]) != (8, 8):
        raise ValueError("fixed product endpoint requires eight committed updates/blocks")
    if (additive["step"], additive["sample_cursor"]) != (8, 8):
        raise AdditiveDoseMismatch("additive endpoint dose mismatch; no checkpoint substitution")


def prepare(args, *, profile=None):
    warm = adapter_runtime.adapter_identity(args.warm_adapter.resolve(), "joint")
    rl = endpoint_identity(args.rl_output.resolve(), "rl", warm)
    if skip_reason(rl):
        save_plan(
            args.output / "SKIPPED.json",
            {
                "reason": skip_reason(rl),
                "endpoint": rl,
                "chain_resolved": True,
                "scientific_readout": False,
            },
        )
        return None, None, None
    sft = endpoint_identity(args.sft_output.resolve(), "sft_control", warm)
    rp, sp = rl["training_plan"], sft["training_plan"]
    if (
        sp["rl_plan_sha256"] != rl["training_plan_sha256"]
        or sp["matched_rl_boundaries"] != rl["boundaries"]
        or sp["cases_sha256"] != rp["cases_sha256"]
        or sp["parent_blocks"] != rp["parent_blocks"]
        or (sft["step"], sft["sample_cursor"]) != (rl["step"], rl["sample_cursor"])
    ):
        raise ValueError("matched control differs from actual RL update inventory")
    identities = dict(zip(CONDITIONS, (warm, rl, sft), strict=True))
    additive_output = getattr(args, "additive_rl_output", None)
    if additive_output is not None:
        if profile is None:
            raise ValueError("additive readout requires explicit frozen panel profile")
        additive = endpoint_identity(additive_output.resolve(), "rl", warm)
        try:
            validate_additive_control(rl, additive)
        except AdditiveDoseMismatch as exc:
            save_plan(
                args.output / "SKIPPED.json",
                {
                    "reason": str(exc),
                    "endpoint": additive,
                    "required_steps": 8,
                    "required_sample_cursor": 8,
                    "actual_additive_steps": additive["step"],
                    "actual_additive_sample_cursor": additive["sample_cursor"],
                    "chain_resolved": True,
                    "scientific_readout": False,
                },
            )
            return None, None, None
        identities["additive_rl_terminal"] = additive
    base = native.evaluation.planner.BASE
    if any(
        p["model"] != str(base)
        or p["base_manifest_sha256"] != sha(base / "local-research-manifest.json")
        for p in (rp, sp)
    ):
        raise ValueError("training base differs from readout base")
    if not 0 < args.hours <= 1:
        raise ValueError("at most one hour for fixed readout")
    cases_path = args.cases.resolve()
    manifest = read(cases_path.with_name("MANIFEST.json"))
    cases_sha = validate_panel(cases_path, manifest, profile)
    from transformers import AutoTokenizer

    cases = panel.read_jsonl(cases_path)
    if len(cases) != 64 or len({c["parent_id"] for c in cases}) != 32:
        raise ValueError("held32 inventory differs")
    tokenizer = AutoTokenizer.from_pretrained(
        native.evaluation.planner.BASE, local_files_only=True, trust_remote_code=False
    )
    stub = object.__new__(native.BaseClient)
    stub.tokenizer = tokenizer
    lengths = {c["id"]: len(stub.ids(baseline.prompt(c))) for c in cases}
    if lengths != manifest["prompt_token_counts"] or max(lengths.values()) + 128 > 8192:
        raise ValueError("native tokenization changed; no truncation")
    plan = {
        "schema": "paired-sufficiency-held32-terminal-v1",
        "cases": str(cases_path),
        "cases_sha256": cases_sha,
        "manifest_sha256": sha(cases_path.with_name("MANIFEST.json")),
        "conditions": list(identities),
        "adapters": identities,
        "jobs": jobs(cases),
        "planned_calls": 128 * len(identities),
        "planned_calls_per_condition": 128,
        "seeds": list(baseline.SEEDS),
        "prompt_instruction": baseline.INSTRUCTION,
        "temperature": 0.5,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 128,
        "context_limit": 8192,
        "truncation": False,
        "prompt_token_counts": lengths,
        "model": str(native.evaluation.planner.BASE),
        "budget_seconds": args.hours * 3600,
        "source_sha256": {
            str(Path(m.__file__).resolve()): sha(Path(m.__file__))
            for m in (baseline, adapter_runtime, native, panel, training)
        },
        "runner_sha256": sha(Path(__file__)),
        "endpoint_selection": "Last committed TRAIN boundary, not held-score selected",
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
    }
    if profile is not None:
        plan["panel_profile"] = profile
        plan["schema"] = profile["readout_schema"]
    if additive_output is not None:
        plan["control_binding"] = {
            "matched_sft_control_for": "rl_terminal",
            "rl_terminal_reward": "product",
            "additive_rl_terminal_reward": "additive",
            "equal_actual_steps_and_sample_cursors": True,
            "dose_caveat": "Not matched information, nonzero-credit tokens, or FLOPs",
        }
    save_plan(args.output / "PLAN.json", plan)
    for condition in plan["conditions"]:
        save_plan(
            args.output / condition / "PLAN.json",
            {
                **plan,
                "comparison_arm": condition,
                "adapters": identities[condition],
                "planned_calls": 128,
            },
        )
    return plan, cases, tokenizer


def summaries(output, plan, cases):
    groups = {c: baseline.summarize(output / c, plan, cases) for c in plan["conditions"]}
    return {
        "planned_calls": plan["planned_calls"],
        "conditions": groups,
        "physical_cost": {
            k: sum(g["physical_cost"][k] for g in groups.values())
            for k in next(iter(groups.values()))["physical_cost"]
        },
        "native_call_seconds": sum(g["native_call_seconds"] for g in groups.values()),
    }


def run(args, *, profile=None):
    args.output = args.output.resolve()
    plan, cases, tokenizer = prepare(args, profile=profile)
    if plan is None or args.prepare_only:
        print(
            json.dumps(
                {
                    "model_loaded": False,
                    "planned_calls": 0 if plan is None else plan["planned_calls"],
                }
            )
        )
        return
    conditions = plan["conditions"]
    if list(args.output.glob("OWNER-*.json")):
        raise ValueError("existing owner; no implicit retry")
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600, lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient allocation")
    lock = (native.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open(
        "a"
    )
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    stopped, failure, model, clients = False, None, None, {}
    try:
        native.save(
            args.output / f"OWNER-{invocation}.json",
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
            raise RuntimeError("requires exactly one GPU assigned by main")
        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            model,
            plan["adapters"][conditions[0]]["path"],
            adapter_name=conditions[0],
            is_trainable=False,
        )
        for c in conditions[1:]:
            model.load_adapter(plan["adapters"][c]["path"], adapter_name=c, is_trainable=False)
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        native.save(
            args.output / f"LOAD-{invocation}.json",
            {
                "adapters": plan["adapters"],
                "optimizer_created": False,
                "gpu": torch.cuda.get_device_name(),
                "cuda": torch.version.cuda,
            },
        )
        clients = {
            c: adapter_runtime.AdapterClient(
                model, tokenizer, args.output / c, deadline, plan["adapters"][c]
            )
            for c in conditions
        }
        lookup = {c["id"]: c for c in cases}
        for job in plan["jobs"]:
            for condition in conditions:
                if stopped or time.time() >= deadline - 5 or (args.output / "STOP").exists():
                    stopped = True
                    break
                activate(model, condition)
                call = clients[condition].call(
                    job["episode_id"],
                    baseline.prompt(lookup[job["case_id"]]),
                    condition,
                    "sufficiency",
                    job["seed"],
                    128,
                    {"truncation": False},
                )
                prediction, error = None, None
                if call["available"]:
                    try:
                        prediction = baseline.parse_output(call["text"])
                    except (ValueError, TypeError) as exc:
                        error = str(exc)
                native.save(
                    args.output / condition / "episodes" / (job["episode_id"] + ".json"),
                    {
                        **job,
                        "call_id": call["call_id"],
                        "available": call["available"],
                        "prediction": prediction,
                        "protocol_error": error,
                        "request_digest": call["request_digest"],
                    },
                )
                if not call["available"]:
                    raise RuntimeError("native inference failed; halt without retry")
            if stopped:
                break
            native.probe.campaign.snapshot(
                args.output / "SUMMARY.json", summaries(args.output, plan, cases)
            )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        clients.clear()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        native.probe.campaign.snapshot(
            args.output / "SUMMARY.json", summaries(args.output, plan, cases)
        )
        native.save(
            args.output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": stopped,
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("cases", "warm-adapter", "rl-output", "sft-output", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument(
        "--prepare-only", "--validate-only", dest="prepare_only", action="store_true"
    )
    run(parser.parse_args())
