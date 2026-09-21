"""Paired base versus helper-SFT36 direct answers: one call, no planner or extra final."""

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

import eval_helper
import eval_planner as evaluation
import probe

CONDITIONS = ("base_direct", "helper_sft_direct")
CASES_SHA = "6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153"
HOTPOT_CASES_SHA = "f16bc99b6b786920a5fecc516d6e2ecfc7c566cf0c38377764e7a8469e424417"
SAMPLING = {
    "do_sample": True,
    "temperature": 0.5,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 128,
    "max_time": 90.0,
}
STOP = False


def panel_spec(panel="fresh003"):
    if panel == "fresh003":
        return {
            "parents": 64,
            "split": "development",
            "dataset": "musique",
            "cases_sha256": CASES_SHA,
            "seed": evaluation.SEED,
            "metric": "official_musique_alias_max_em_f1",
            "hours": 1.0,
        }
    if panel == "hotpot_explorer32":
        return {
            "parents": 32,
            "split": "transfer",
            "dataset": "hotpotqa",
            "cases_sha256": HOTPOT_CASES_SHA,
            "seed": eval_helper.SEED,
            "metric": "official_hotpotqa_em_f1",
            "hours": 1 / 3,
        }
    raise ValueError("unknown fixed direct panel")


def validate_panel(cases, panel="fresh003"):
    spec = panel_spec(panel)
    if (
        len(cases) != spec["parents"]
        or len({c["id"] for c in cases}) != spec["parents"]
        or any(
            c["split"] != spec["split"] or c.get("dataset", "musique") != spec["dataset"]
            for c in cases
        )
    ):
        raise ValueError(
            f"expected all{spec['parents']} unique {spec['dataset']} {spec['split']} parents"
        )


class DirectClient:
    """Native HF receipts with explicit direct-answer identities; no hidden root calls."""

    def __init__(self, model, tokenizer, output, deadline, adapter_sha, seed_base=evaluation.SEED):
        self.model, self.tokenizer = model, tokenizer
        self.output, self.deadline, self.adapter_sha = Path(output), deadline, adapter_sha
        self.returned = self.failed = self.consecutive_failures = 0
        self.seed_base = seed_base

    def call(self, case, condition, repeat):
        import torch

        if condition not in CONDITIONS or repeat not in (0, 1):
            raise ValueError("unknown direct arm/repeat")
        enabled = condition == "helper_sft_direct"
        identity = f"{case['id']}-r{repeat}-{condition}-answer"
        prompt = evaluation.direct_prompt(case)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        seed = self.seed_base + int(probe.runtime.digest(case["id"])[:6], 16) + repeat * 100 + 2
        request = {
            "prompt": prompt,
            "input_token_ids": ids,
            "condition": condition,
            "role": "direct_answer",
            "model": str(evaluation.planner.BASE),
            "adapter_enabled": enabled,
            "adapter_name": eval_helper.ADAPTER_NAME if enabled else None,
            "adapter_sha256": self.adapter_sha if enabled else None,
            "seed": seed,
            "sampling": SAMPLING,
        }
        digest = probe.runtime.digest(request)
        path, start = (
            self.output / "calls" / (identity + ".json"),
            self.output / "starts" / (identity + ".json"),
        )
        if path.exists():
            old = eval_helper.read(path)
            if old["request_digest"] != digest or probe.runtime.digest(old["request"]) != digest:
                raise ValueError("cached request differs")
            return old
        if start.exists():
            raise RuntimeError("unresolved direct start; no implicit retry")
        row = {
            "call_id": identity,
            "condition": condition,
            "role": "direct_answer",
            "model": str(evaluation.planner.BASE),
            "adapter_enabled": enabled,
            "adapter_name": request["adapter_name"],
            "adapter_sha256": request["adapter_sha256"],
            "request": request,
            "request_digest": digest,
            "input_token_ids": ids,
            "available": False,
            "text": None,
            "usage": {},
            "started": time.time(),
        }
        probe.runtime.save(start, row)
        try:
            remaining = min(90.0, self.deadline - time.time() - 1)
            if STOP or remaining <= 0:
                raise TimeoutError("direct readout deadline")
            if len(ids) + 128 > 8192:
                raise ValueError("context limit exceeded; no truncation")
            torch.manual_seed(seed)
            if str(self.model.device).startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            tensor = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            row["usage"]["prompt_tokens"] = len(ids)
            row["effective_max_time"] = remaining
            # Reuse tested named-adapter freezing, not helper prompts/role receipts.
            routing = "trained_helper" if enabled else "base_helper"
            with eval_helper.route(self.model, routing, "helper"), torch.no_grad():
                sequence = self.model.generate(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    **{**SAMPLING, "max_time": remaining},
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            actual = sequence[0].detach().cpu().tolist()
            if actual[: len(ids)] != ids or len(actual) <= len(ids):
                raise ValueError("native answer did not extend saved prompt")
            emitted = actual[len(ids) :]
            row.update(
                output_token_ids=emitted,
                usage={"prompt_tokens": len(ids), "completion_tokens": len(emitted)},
            )
            answer = self.tokenizer.decode(
                emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if not answer.strip():
                raise ValueError("empty scientific response")
            if self.returned == 0 and time.time() - row["started"] > 90:
                raise TimeoutError("first scientific response exceeded90 seconds")
            row.update(
                available=True,
                text=answer,
                finish_reason="eos"
                if emitted[-1] == self.tokenizer.eos_token_id
                else "length"
                if len(emitted) >= 128
                else "max_time_or_other_stop",
            )
            self.returned += 1
            self.consecutive_failures = 0
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.failed += 1
            self.consecutive_failures += 1
        row["ended"] = time.time()
        probe.runtime.save(path, row)
        print(
            json.dumps(
                {
                    "call_id": identity,
                    "available": row["available"],
                    "usage": row["usage"],
                    "seconds": row["ended"] - row["started"],
                }
            ),
            flush=True,
        )
        return row


def collect_episode(client, case, condition, repeat):
    call = client.call(case, condition, repeat)
    identity = f"{case['id']}-r{repeat}-{condition}"
    grade = eval_helper.grade_final(call["text"] if call["available"] else "", case)
    row = {
        "episode_id": identity,
        "case_id": case["id"],
        "split": case["split"],
        "condition": condition,
        "repeat": repeat,
        "mode": "direct_adapted",
        "planner_called": False,
        "helper_called": False,
        "extra_final_called": False,
        "call_ids": [call["call_id"]],
        "available": call["available"],
        **grade,
        "status": "generation_failure"
        if not call["available"]
        else "scored"
        if grade["valid"]
        else "invalid_answer",
        "deployed_cost": evaluation.cost([call]),
    }
    path = client.output / "episodes" / (identity + ".json")
    if path.exists():
        if eval_helper.read(path) != row:
            raise ValueError("resumed direct episode changed")
    else:
        probe.runtime.save(path, row)
    return row


def summarize(output, plan):
    rows = [eval_helper.read(p) for p in (output / "episodes").glob("*.json")]
    calls = [eval_helper.read(p) for p in (output / "calls").glob("*.json")]
    ids = {r["call_id"] for r in calls}
    unresolved = [
        eval_helper.read(p) for p in (output / "starts").glob("*.json") if p.stem not in ids
    ]
    denominator = len(plan["case_ids"]) * plan["repeats"]
    groups = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        groups[condition] = {
            "planned": denominator,
            "recorded": len(subset),
            "missing": denominator - len(subset),
            "correct": sum(r["correct"] for r in subset),
            "em": sum(r["correct"] for r in subset) / denominator,
            "f1": sum(r["f1"] for r in subset) / denominator,
            "status_counts": dict(Counter(r["status"] for r in subset)),
            "physical_cost": evaluation.cost(
                [r for r in calls + unresolved if r["condition"] == condition]
            ),
        }
    indexed = {(r["case_id"], r["repeat"], r["condition"]): r for r in rows}
    changes = Counter()
    for parent in plan["case_ids"]:
        for repeat in range(plan["repeats"]):
            a, b = [indexed.get((parent, repeat, c), {}) for c in CONDITIONS]
            change = int(b.get("correct", False)) - int(a.get("correct", False))
            if change:
                category = (
                    "missing_involved"
                    if not a or not b
                    else "both_valid"
                    if a["valid"] and b["valid"]
                    else "protocol_or_generation_involved"
                )
                changes[("win_" if change > 0 else "loss_") + category] += 1
    return {
        "conditions": groups,
        "metric": plan.get("metric", "official_musique_alias_max_em_f1"),
        "physical_cost": evaluation.cost(calls + unresolved),
        "unresolved_start_receipts": len(unresolved),
        "paired_changes": dict(changes),
        "all_planned_complete": len(rows) == denominator * 2,
        "parents": len(plan["case_ids"]),
        "episodes_per_condition": denominator,
        "note": "One direct-answer call per attempt; no planner/helper/extra final. All planned "
        "attempts remain in denominators. Missing outcomes are incomplete lower bounds, "
        "not observed errors. Unknown cost is not zero. Repeats are not independent parents.",
        "updated": time.time(),
    }


def run(args):
    global STOP
    STOP = False
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    panel = getattr(args, "panel", "fresh003")
    spec = panel_spec(panel)
    if args.hours is None:
        args.hours = spec["hours"]
    if not 0 < args.hours <= spec["hours"]:
        raise ValueError(f"at most {spec['hours'] * 60:g} cumulative minutes for {panel}")
    cases_path, adapter, output = (
        args.cases.resolve(),
        args.helper_adapter.resolve(),
        args.output.resolve(),
    )
    if probe.campaign.sha(cases_path) != spec["cases_sha256"]:
        raise ValueError("requires authoritative fixed panel cases: " + panel)
    cases = [json.loads(line) for line in cases_path.open()]
    validate_panel(cases, panel)
    binding = evaluation.adapter_identity(adapter)
    state, training = (
        eval_helper.read(adapter / "STATE.json"),
        eval_helper.read(adapter.parent / "PLAN.json"),
    )
    model_path = evaluation.planner.BASE
    eval_helper.validate_helper_training(training, model_path)
    manifest_sha = probe.campaign.sha(model_path / "local-research-manifest.json")
    if (
        state["step"] != 36
        or state.get("epoch") != 1
        or training.get("model_manifest_sha256") != manifest_sha
    ):
        raise ValueError("fixed helper-SFT36 one-epoch/base binding required")
    plan = {
        "schema": "paired-direct-helper-adapter-v1",
        "mode": "direct_adapted",
        "cases": str(cases_path),
        "cases_sha256": spec["cases_sha256"],
        "case_ids": [c["id"] for c in cases],
        "split": spec["split"],
        "parents": spec["parents"],
        "repeats": 2,
        "conditions": list(CONDITIONS),
        "maximum_calls": spec["parents"] * 4,
        "model": str(model_path),
        "model_manifest_sha256": manifest_sha,
        "generation_config_sha256": probe.campaign.sha(model_path / "generation_config.json"),
        "helper_adapter": str(adapter),
        "helper_adapter_binding": binding,
        "helper_training_plan_sha256": probe.campaign.sha(adapter.parent / "PLAN.json"),
        "root_adapter_loaded": False,
        "seed": spec["seed"],
        "sampling": SAMPLING,
        "seed_policy": ("eval_planner.SEED" if panel == "fresh003" else "eval_helper.SEED")
        + " + int(digest(case_id)[:6],16) + repeat*100 + 2",
        "metric": spec["metric"],
        "budget_seconds": args.hours * 3600,
        "architecture": "Same direct_prompt for both arms; base versus helper-SFT36 "
        "enabled for one direct_answer call. No planner, decomposition, helper or extra final.",
        "baseline_reuse": False,
        "dependencies": {
            str(p): probe.campaign.sha(p)
            for p in (
                Path(__file__),
                Path(evaluation.__file__),
                Path(evaluation.planner.__file__),
                Path(eval_helper.__file__),
                Path(probe.__file__),
                *eval_helper.metric_sources(spec["dataset"]),
            )
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
    }
    if panel != "fresh003":
        plan.update(panel=panel, dataset=spec["dataset"], summary_every_episodes=16)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if eval_helper.read(output / "PLAN.json") != plan:
            raise ValueError("immutable direct-adapted PLAN differs")
    else:
        probe.runtime.save(output / "PLAN.json", plan)
    owners = {p.stem.removeprefix("OWNER-") for p in output.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in output.glob("TERMINAL-*.json")}
    if owners != terminals:
        raise ValueError("previous direct owner unresolved")
    spent = sum(eval_helper.read(p)["elapsed_seconds"] for p in output.glob("TERMINAL-*.json"))
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + max(0, args.hours * 3600 - spent), lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient remaining cumulative/allocation budget")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    model, failure = None, None
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
            global STOP
            STOP = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base,
            adapter,
            adapter_name=eval_helper.ADAPTER_NAME,
            is_trainable=False,
            autocast_adapter_dtype=True,
        )
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        probe.runtime.save(
            output / f"LOAD-{invocation}.json",
            {
                "loaded_adapter_names": list(model.peft_config),
                "root_adapter_loaded": False,
                "helper_binding": binding,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "model_training": model.training,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        client = DirectClient(
            model,
            tokenizer,
            output,
            deadline,
            binding["adapter_model.safetensors"],
            seed_base=spec["seed"],
        )
        visited = 0
        for case in cases:
            for repeat in range(2):
                order = (
                    CONDITIONS
                    if (int(probe.runtime.digest(case["id"])[:4], 16) + repeat) % 2 == 0
                    else CONDITIONS[::-1]
                )
                for condition in order:
                    if STOP or time.time() >= deadline - 5 or (output / "STOP").exists():
                        STOP = True
                        break
                    collect_episode(client, case, condition, repeat)
                    visited += 1
                    if panel == "fresh003" or visited % 16 == 0:
                        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
                    if panel != "fresh003":
                        probe.campaign.snapshot(
                            output / "STATUS.json",
                            {
                                "visited_episodes_this_owner": visited,
                                "returned_this_owner": client.returned,
                                "failed_this_owner": client.failed,
                                "updated": time.time(),
                            },
                        )
                    if (client.returned == 0 and client.failed) or client.consecutive_failures >= 2:
                        raise RuntimeError("scientific response check failed")
                if STOP:
                    break
            if STOP:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": STOP,
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--helper-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("fresh003", "hotpot_explorer32"), default="fresh003")
    parser.add_argument("--hours", type=float, help="default/cap: fresh003=1; Hotpot=1/3")
    run(parser.parse_args())
