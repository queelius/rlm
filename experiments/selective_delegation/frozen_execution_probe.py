"""Conditional downstream-only execution of the 64 frozen batch-0001 plans."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
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

CONDITIONS = ("trained_helper", "base_helper")
SEED = 2026092161
SEEDS = tuple(SEED + 1000 * r for r in range(4))
STOP = False


def execution_seed(parent_index, execution_repeat):
    if not 0 <= parent_index < 16 or execution_repeat not in range(4):
        raise ValueError("bounded16-parent four-seed inventory required")
    return SEED + parent_index * 10000 + execution_repeat * 1000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_slots(source):
    rows = [json.loads(p.read_text()) for p in sorted(Path(source).glob("episodes/*.json"))]
    if len(rows) != 64 or any(not r.get("plan_valid") for r in rows):
        raise ValueError("expected exactly64 valid frozen batch-0001 plans")
    keys = {(r["case_id"], r["candidate"]) for r in rows}
    parents = {p for p, _ in keys}
    if len(parents) != 16 or keys != {(p, c) for p in parents for c in range(4)}:
        raise ValueError("expected16 parents by four candidate plans")
    steps = Counter(len(r["plan"]["subquestions"]) for r in rows)
    if any(n < 1 or n > 4 for n in steps):
        raise ValueError("unexpected frozen plan length")
    return rows, steps


def prepare(source, cases_path):
    source = Path(source).resolve()
    slots, lengths = frozen_slots(source)
    if source.name != "batch-0001":
        raise ValueError("only frozen batch-0001 is accepted")
    source_plan = eval_helper.read(source.parent / "PLAN.json")
    parents = source_plan["case_ids_by_update"][0]
    cases = {r["id"]: r for r in map(json.loads, Path(cases_path).read_text().splitlines())}
    if (
        len(parents) != 16
        or set(parents) != {s["case_id"] for s in slots}
        or source_plan["cases_sha256"] != sha(cases_path)
        or any(s["case_id"] not in cases or cases[s["case_id"]]["split"] != "train" for s in slots)
    ):
        raise ValueError("frozen slots must bind train cases")
    helper = source_plan["helper_contract"]
    if helper["mode"] != "trained_helper":
        raise ValueError("saved source must use fixed trained helper")
    adapter = Path(helper["adapter"])
    if evaluation.adapter_identity(adapter) != helper["adapter_binding"]:
        raise ValueError("helper checkpoint identity changed")
    state = eval_helper.read(adapter / "STATE.json")
    training = eval_helper.read(adapter.parent / "PLAN.json")
    eval_helper.validate_helper_training(training, evaluation.planner.BASE)
    if state["step"] != 36 or state["epoch"] != 1:
        raise ValueError("fixed helper-SFT36 required")
    manifest = sha(evaluation.planner.BASE / "local-research-manifest.json")
    if helper["base_manifest_sha256"] != manifest or training["model_manifest_sha256"] != manifest:
        raise ValueError("helper base manifest changed")
    if sha(adapter.parent / "PLAN.json") != helper["training_plan_sha256"]:
        raise ValueError("helper training PLAN changed")
    hashes = {
        str(source.parent / "PLAN.json"): sha(source.parent / "PLAN.json"),
        str(source / "BATCH.json"): sha(source / "BATCH.json"),
    }
    for path, digest in source_plan["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("source dependency changed")
        hashes[path] = digest
    for slot in slots:
        path = source / "episodes" / (slot["episode_id"] + ".json")
        root_path = source / "calls" / (slot["episode_id"] + "-root.json")
        root = eval_helper.read(root_path)
        request = root["request"]
        if (
            slot["update"] != 1
            or probe.runtime.digest(request) != root["request_digest"]
            or not root["available"]
            or request["role"] != "root"
            or request["prompt"] != evaluation.planner_prompt(cases[slot["case_id"]])
            or request["adapter_sha256"]
            != source_plan["adapter_binding"]["adapter_model.safetensors"]
            or evaluation.parse_plan(root["text"]) != slot["plan"]
        ):
            raise ValueError("saved root plan/request differs")
        hashes.update({str(path): sha(path), str(root_path): sha(root_path)})
    helper_calls = sum(length * count for length, count in lengths.items())
    if lengths != Counter({2: 47, 3: 14, 4: 2, 1: 1}) or 4 * (helper_calls + 64) != 836:
        raise ValueError("expected exact836-call batch-0001 plan inventory")
    old_seeds = {eval_helper.read(p)["request"]["seed"] for p in source.glob("calls/*.json")}
    new_seeds = {
        execution_seed(i, r) + offset
        for i in range(16)
        for r in range(4)
        for offset in (1, 2, 3, 4, 100)
    }
    if old_seeds & new_seeds:
        raise ValueError("new diagnostic seeds overlap original rollout seeds")
    return {
        "schema": "frozen-execution-noise-v1",
        "source": str(Path(source).resolve()),
        "source_batch_sha256": sha(Path(source) / "BATCH.json"),
        "cases_sha256": sha(cases_path),
        "case_ids": parents,
        "source_hashes": hashes,
        "helper_contract": helper,
        "model": source_plan["model"],
        "model_manifest_sha256": manifest,
        "seed": SEED,
        "seed_policy": "SEED + source-parent-index*10000 + execution-repeat*1000; "
        "helper+i+1, final+100; shared across four candidates",
        "seeds": list(SEEDS),
        "slots": [
            {
                "case_id": s["case_id"],
                "candidate": s["candidate"],
                "plan": s["plan"],
                "parent_index": parents.index(s["case_id"]),
            }
            for s in sorted(slots, key=lambda x: (parents.index(x["case_id"]), x["candidate"]))
        ],
        "caps": {"helper_total": 384, "final": 128, "temperature": 0.5},
        "realized_plan_lengths": {str(k): v for k, v in sorted(lengths.items())},
        "planned_new_calls": 4 * (helper_calls + 64),
        "no_root_generation": True,
    }


def execute_slot(client, case, slot, execution_repeat):
    """Execute saved plan with an eval_helper-compatible native client; no gold enters prompts."""
    plan, seed = slot["plan"], execution_seed(slot["parent_index"], execution_repeat)
    key = f"{case['id']}-c{slot['candidate']}-e{execution_repeat}"
    row = {
        "episode_id": key,
        "case_id": case["id"],
        "candidate": slot["candidate"],
        "execution_repeat": execution_repeat,
        "plan": plan,
        "seed": seed,
        "status": "invalid_dependency",
        "available": True,
        "reward": 0,
        "call_ids": [],
    }
    answers, trace, records = [], [], []

    def finish():
        row.update(helper_trace=trace, cost=evaluation.cost(records))
        return row

    try:
        cap = 384 // len(plan["subquestions"])
        for index, question in enumerate(plan["subquestions"]):
            row["status"] = "invalid_dependency"
            resolved = evaluation.bind_question(question, answers)
            prompt = evaluation.isolated_helper_prompt(case, resolved)
            call = client.call(
                key + f"-helper-{index + 1}",
                prompt,
                "trained_helper",
                "helper",
                seed + index + 1,
                max_new_tokens=cap,
            )
            row["call_ids"].append(call["call_id"])
            records.append(call)
            if not call.get("available"):
                row.update(status="missing_generation", available=False, reward=None)
                return finish()
            row["status"] = "invalid_helper"
            answer = evaluation.parse_helper_answer(call["text"])
            answers.append(answer)
            trace.append(
                {
                    "step": index + 1,
                    "question": question,
                    "resolved_question": resolved,
                    "answer": answer,
                }
            )
        row["status"] = "final"
        prompt = evaluation.final_prompt(case, plan, {"execution": "isolated", "steps": trace})
        call = client.call(
            key + "-final", prompt, "base_helper", "final", seed + 100, max_new_tokens=128
        )
        row["call_ids"].append(call["call_id"])
        records.append(call)
        if not call.get("available"):
            row.update(status="missing_generation", available=False, reward=None)
            return finish()
        score = probe.grade(call["text"], case)
        row.update(score=score, reward=int(score["correct"]), trace=trace)
        row["status"] = "scored" if score["valid"] else "invalid_final"
    except (ValueError, TypeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return finish()


def summarize(output, plan):
    rows = [eval_helper.read(p) for p in (output / "episodes").glob("*.json")]
    calls = [eval_helper.read(p) for p in (output / "calls").glob("*.json")]
    known = [r for r in rows if r["reward"] is not None]
    call_ids = {r["call_id"] for r in calls}
    unresolved = [
        eval_helper.read(p) for p in (output / "starts").glob("*.json") if p.stem not in call_ids
    ]
    return {
        "planned_episodes": 256,
        "recorded_episodes": len(rows),
        "missing_episodes": 256 - len(rows),
        "available_outcomes": len(known),
        "unavailable_outcomes": sum(r["reward"] is None for r in rows),
        "correct_observed": sum(r["reward"] for r in known),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "physical_cost": evaluation.cost(calls + unresolved),
        "maximum_calls": plan["planned_new_calls"],
        "unresolved_starts": len(unresolved),
        "all_planned_complete": len(known) == 256 and not unresolved,
        "updated": time.time(),
    }


def run(args):
    global STOP
    STOP = eval_helper.STOP = False
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source, output = args.source.resolve(), args.output.resolve()
    if source == output or not 0 < args.hours <= 1:
        raise ValueError("separate output and at most one cumulative hour required")
    plan = prepare(source, args.cases)
    plan.update(
        budget_seconds=args.hours * 3600,
        dependencies={
            str(p): sha(p)
            for p in (
                Path(__file__),
                Path(eval_helper.__file__),
                Path(evaluation.__file__),
                Path(evaluation.planner.__file__),
                Path(probe.__file__),
                probe.MUSIQUE / "metrics/answer.py",
            )
        },
        environment={
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
    )
    cases = {c["id"]: c for c in map(json.loads, args.cases.read_text().splitlines())}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if eval_helper.read(output / "PLAN.json") != plan:
            raise ValueError("immutable frozen-execution PLAN differs")
    else:
        probe.runtime.save(output / "PLAN.json", plan)
    owners = {p.stem.removeprefix("OWNER-") for p in output.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in output.glob("TERMINAL-*.json")}
    if owners != terminals:
        raise ValueError("previous frozen-execution owner unresolved")
    spent = sum(eval_helper.read(p)["elapsed_seconds"] for p in output.glob("TERMINAL-*.json"))
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + max(0, args.hours * 3600 - spent), lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient remaining cumulative/allocation budget")
    tokenizer = AutoTokenizer.from_pretrained(
        evaluation.planner.BASE, local_files_only=True, trust_remote_code=False
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
            STOP = eval_helper.STOP = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        base = AutoModelForCausalLM.from_pretrained(
            evaluation.planner.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        helper = plan["helper_contract"]
        model = PeftModel.from_pretrained(
            base,
            helper["adapter"],
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
                "helper_contract": helper,
                "root_adapter_loaded": False,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "loaded_adapter_names": list(model.peft_config),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        client = eval_helper.HelperClient(
            model,
            tokenizer,
            output,
            deadline,
            helper["adapter_binding"]["adapter_model.safetensors"],
        )
        for parent in plan["case_ids"]:
            parent_slots = [slot for slot in plan["slots"] if slot["case_id"] == parent]
            for repeat in range(4):
                for slot in parent_slots:
                    if STOP or time.time() >= deadline - 5 or (output / "STOP").exists():
                        STOP = True
                        break
                    row = execute_slot(client, cases[parent], slot, repeat)
                    path = output / "episodes" / (row["episode_id"] + ".json")
                    if path.exists():
                        if eval_helper.read(path) != row:
                            raise ValueError("resumed frozen execution changed")
                    else:
                        probe.runtime.save(path, row)
                    probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
                    if row["reward"] is None:
                        raise RuntimeError(
                            "missing scientific generation; preserve attempt and halt"
                        )
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    run(parser.parse_args())
