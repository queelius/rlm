"""Frozen-base native TextCraft flat/recursive screen with one global tree budget."""

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

import prepare_textcraft as inputs
import probe
import textcraft_bridge as bridge

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
BASE = Path(
    "/project/alex_phd/research-cache/models/"
    "Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554"
)
INPUT_SHA = "79ac9326c209e3df3d31cf3a479fec264506fb15e19445f06370a1c3d5baf405"
SEEDS = (2026092204, 2026092205)
STOP = False
save = inputs.save
INSTRUCTION_REMINDER = (
    "\nChoose exactly ONE next action: return ONE JSON object, then stop. "
    "Use current_inventory, not previous action text, to decide what is still missing. "
    "can_craft means a recipe exists, not that its ingredients are present. "
    "For EVERY item in target_items, compare current_inventory[item] minus "
    "inventory_at_task_start[item] with target_items[item] (absent inventory items count as0). "
    "If every difference is at least the requested amount, return "
    '{"action":"finish","message":"done"} now. Otherwise choose one useful action.'
)
PROCEDURAL_INSTRUCTION = (
    "\nFirst query each requested target's recipe. Discover needed ingredients from returned "
    "recipes before crafting. Use current inventory and returned batch sizes; do not repeatedly "
    "query an item whose recipe is already known. Finish only after the net target is met."
)


def render_prompt(frame, history, context="", goal=None, profile="original"):
    if profile not in ("original", "instruction_control", "procedure_control"):
        raise ValueError("unknown TextCraft prompt profile")
    if profile != "original" and frame.max_depth != 0:
        raise ValueError("instruction control is flat only")
    prompt = bridge.public_prompt(frame, history, context=context, goal=goal)
    if profile == "procedure_control":
        return prompt + PROCEDURAL_INSTRUCTION
    return prompt + (INSTRUCTION_REMINDER if profile == "instruction_control" else "")


def cost(calls):
    return {
        "calls": len(calls),
        "prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in calls),
        "completion_tokens": sum(c.get("usage", {}).get("completion_tokens", 0) for c in calls),
        "failed_calls": sum(not c.get("available", False) for c in calls),
        "unknown_completion_tokens_calls": sum(
            "completion_tokens" not in c.get("usage", {}) for c in calls
        ),
        "native_service_seconds": sum(c["ended"] - c["started"] for c in calls if "ended" in c),
    }


class NativeClient:
    def __init__(
        self, model, tokenizer, output, deadline, model_manifest_sha, plan_sha, adapter=None
    ):
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("all inference parameters must be frozen")
        if adapter is None and hasattr(model, "peft_config"):
            raise ValueError("base-only client cannot carry an adapter")
        if adapter is not None and (
            not hasattr(model, "peft_config")
            or model.active_adapters != ["textcraft_action"]
            or any(getattr(m, "disable_adapters", False) is True for m in model.modules())
        ):
            raise ValueError("explicit active textcraft_action adapter required")
        self.adapter = adapter
        self.model, self.tokenizer, self.output = model, tokenizer, Path(output)
        self.deadline, self.model_manifest_sha, self.plan_sha = (
            deadline,
            model_manifest_sha,
            plan_sha,
        )
        self.returned = self.failed = 0

    def ids(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def call(self, spec):
        import torch
        from transformers import GenerationConfig

        ids = self.ids(spec["prompt"])
        request = {
            **spec,
            "input_token_ids": ids,
            "model": str(BASE),
            "model_manifest_sha256": self.model_manifest_sha,
            "adapter_enabled": self.adapter is not None,
            "adapter_sha256": self.adapter["sha256"] if self.adapter else None,
            "context_limit": 8192,
            "truncation": False,
            "sampling": {
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": spec["cap"],
                "do_sample": True,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "max_time": 90.0,
            },
        }
        if self.adapter:
            request.update(
                adapter_path=self.adapter["path"],
                adapter_commit_sha256=self.adapter["commit_sha256"],
            )
        row = {
            "call_id": spec["call_id"],
            "request": request,
            "request_digest": probe.runtime.digest(request),
            "plan_sha256": self.plan_sha,
            "condition": spec.get("condition", spec["policy"]),
            "role": spec["role"],
            "node_id": spec["node_id"],
            "parent_node_id": spec["parent_node_id"],
            "depth": spec["depth"],
            "episode_id": spec["episode_id"],
            "available": False,
            "text": None,
            "input_token_ids": ids,
            "usage": {"prompt_tokens": len(ids)},
            "started": time.time(),
        }
        save(self.output / "starts" / (spec["call_id"] + ".json"), row)
        try:
            remaining = min(90.0, self.deadline - time.time() - 1)
            if remaining <= 0 or not 1 <= spec["cap"] <= 256 or len(ids) + spec["cap"] > 8192:
                raise ValueError("native context/cap/deadline differs from validated request")
            if spec["role"] != ("root" if spec["depth"] == 0 else "child"):
                raise ValueError("node role/depth mismatch")
            torch.manual_seed(spec["seed"])
            if str(self.model.device).startswith("cuda"):
                torch.cuda.manual_seed_all(spec["seed"])
            config = GenerationConfig(
                do_sample=True,
                temperature=0.5,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                max_new_tokens=spec["cap"],
                max_time=remaining,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            )
            tensor = torch.tensor([ids], device=self.model.device)
            row["effective_max_time"] = remaining
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    generation_config=config,
                )
            sequence = generated[0].detach().cpu().tolist()
            emitted = sequence[len(ids) :]
            if sequence[: len(ids)] != ids or not 1 <= len(emitted) <= spec["cap"]:
                raise ValueError("native continuation/prefix/output cap mismatch")
            row.update(
                available=True,
                output_token_ids=emitted,
                text=self.tokenizer.decode(
                    emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
                usage={"prompt_tokens": len(ids), "completion_tokens": len(emitted)},
                finish_reason="eos"
                if emitted[-1] == self.tokenizer.eos_token_id
                else "length_or_time",
            )
            self.returned += 1
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.failed += 1
        row["ended"] = time.time()
        save(self.output / "calls" / (spec["call_id"] + ".json"), row)
        probe.campaign.snapshot(
            self.output / "STATUS.json",
            {
                "state": "native_calls",
                "returned": self.returned,
                "failed": self.failed,
                "last_call": spec["call_id"],
                "updated": time.time(),
            },
        )
        if self.returned == 1 and row["ended"] - row["started"] > 90:
            raise TimeoutError("first real scientific response exceeded90seconds")
        return row


def episode(task, job, client, world, output, deadline, budget=None, profile="original"):
    profile = job.get("prompt_profile", profile)
    budget = budget or bridge.Budget()
    root = bridge.Frame(
        world,
        dict(task["misc"]["initial_inventory"]),
        task["misc"]["target_items"],
        budget,
        max_depth=0 if job["policy"] == "flat" else 2,
    )
    call_ids, nodes, errors = [], [], Counter()
    next_node = 0
    failure = None
    started = time.time()

    def execute(frame, parent_node_id, context, goal):
        nonlocal next_node
        node_id = f"n{next_node}"
        next_node += 1
        history, local_calls = [], []
        status, error = "unfinished", None
        begin = time.time()
        try:
            while not frame.finished:
                if STOP or time.time() >= deadline - 2:
                    raise TimeoutError("owner cap/interruption; incomplete episode unknown")
                try:
                    cap = budget.reserve()
                except bridge.BudgetExceeded:
                    status = (
                        "global_call_cap"
                        if budget.calls >= budget.max_calls
                        else "global_token_cap"
                    )
                    break
                prompt = render_prompt(frame, history, context=context, goal=goal, profile=profile)
                if len(client.ids(prompt)) + cap > 8192:
                    status = "context_cap"
                    break
                index = budget.calls
                identity = f"{job['episode_id']}-c{index:03d}"
                spec = {
                    "call_id": identity,
                    "prompt": prompt,
                    "policy": job["policy"],
                    "role": "root" if frame.depth == 0 else "child",
                    "depth": frame.depth,
                    "node_id": node_id,
                    "parent_node_id": parent_node_id,
                    "episode_id": job["episode_id"],
                    "task_id": task["id"],
                    "global_call_index": index,
                    "cap": cap,
                    "seed": job["seed"]
                    + int(hashlib.sha256(task["id"].encode()).hexdigest()[:8], 16)
                    + 1000 * index,
                }
                if "condition" in job:
                    spec["condition"] = job["condition"]
                    spec["prompt_profile"] = profile
                call_ids.append(identity)
                local_calls.append(identity)
                call = client.call(spec)
                if not call["available"]:
                    budget.charge(0)  # attempted call; emitted tokens unknown, owner halts below
                    raise RuntimeError("unavailable native response; no retry/reward substitution")
                budget.charge(len(call["output_token_ids"]))
                try:
                    action = bridge.parse_action(call["text"])
                except ValueError as exc:
                    errors["invalid_schema"] += 1
                    history.append(
                        {
                            "response": call["text"],
                            "feedback": f"Rejected action: {exc}. Inventory unchanged.",
                        }
                    )
                    continue
                try:
                    response = frame.apply(action)
                except ValueError as exc:
                    errors["rejected_action"] += 1
                    history.append(
                        {
                            "action": action,
                            "feedback": f"Rejected action: {exc}. Inventory unchanged.",
                        }
                    )
                    continue
                if action["action"] == "delegate":
                    child = execute(response, node_id, action["context"], None)
                    feedback = {
                        "child_status": child["status"],
                        "child_message": child["finish_message"],
                        "inventory_after_return": dict(frame.inventory),
                    }
                else:
                    feedback = response
                    if isinstance(response, str) and response.startswith("Error:"):
                        errors["native_action_error"] += 1
                history.append({"action": action, "feedback": feedback})
            if frame.finished:
                status = "finished"
        except Exception as exc:
            status, error = "external_error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            score, native_details = frame.score() if error is None else (None, None)
            result = {
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "depth": frame.depth,
                "targets": frame.targets,
                "initial_inventory": frame.initial_inventory,
                "final_inventory": dict(frame.inventory),
                "status": status,
                "error": error,
                "finish_message": frame.message,
                "native_score": score,
                "native_details": native_details,
                "call_ids": local_calls,
                "public_history": history,
                "started": begin,
                "ended": time.time(),
            }
            nodes.append(result)
            save(output / "nodes" / f"{job['episode_id']}-{node_id}.json", result)
        return result

    result = None
    try:
        result = execute(root, None, "", task["goal"])
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    record = {
        **job,
        "observed": failure is None,
        "status": result["status"] if result else "external_error",
        "native_score": result["native_score"] if result else None,
        "failure": failure,
        "global_calls": len(call_ids),
        "global_output_tokens": budget.output_tokens,
        "output_tokens_are_lower_bound": failure is not None,
        "call_ids": call_ids,
        "node_ids": [n["node_id"] for n in nodes],
        "node_count": len(nodes),
        "max_depth_reached": max((n["depth"] for n in nodes), default=0),
        "errors": dict(errors),
        "root_initial_inventory": root.initial_inventory,
        "final_inventory": dict(root.inventory),
        "started": started,
        "ended": time.time(),
    }
    save(output / "episodes" / (job["episode_id"] + ".json"), record)
    return record


def prepare(prepared, output, hours, profile="original", persist=True):
    prepared, output = prepared.resolve(), output.resolve()
    if profile not in ("original", "instruction_control"):
        raise ValueError("unknown TextCraft profile")
    if profile == "instruction_control" and hours > 0.75:
        raise ValueError("instruction control capped at45minutes")
    policies = ("flat",) if profile == "instruction_control" else ("flat", "recursive")
    if not 0 < hours <= 1 or inputs.sha(prepared / "MANIFEST.json") != INPUT_SHA:
        raise ValueError("one-hour cap and exact043manifest required")
    manifest = json.loads((prepared / "MANIFEST.json").read_text())
    if manifest["model"] != str(BASE) or manifest["model_manifest_sha256"] != inputs.sha(
        BASE / "local-research-manifest.json"
    ):
        raise ValueError("model differs from CPU-qualified043 token audit")
    for relative, digest in manifest["sha256"].items():
        if inputs.sha(ROOT / relative) != digest:
            raise ValueError("immutable043source/input changed: " + relative)
    if (
        inputs.sha(Path(bridge.__file__))
        != manifest["sha256"]["source-043-textcraft-bridge/textcraft_bridge.py"]
    ):
        raise ValueError("collector bridge differs from qualified043")
    if (
        not manifest["ready"]
        or not manifest["all_native_gold_success"]
        or manifest["episode_seeds"] != list(SEEDS)
    ):
        raise ValueError("frozen CPU qualification/seeds differ")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    if [t["id"] for t in tasks] != manifest["task_ids"] or len(tasks) != 8:
        raise ValueError("exact8 frozen tasks required")
    jobs = [
        {
            "episode_id": f"t{i:02d}-r{repeat}-{policy}",
            "task_id": task["id"],
            "repeat": repeat,
            "seed": seed,
            "policy": policy,
        }
        for i, task in enumerate(tasks)
        for repeat, seed in enumerate(SEEDS)
        for policy in policies
    ]
    plan = {
        "schema": "textcraft-native-recursion-screen-v1",
        "prepared": str(prepared),
        "manifest_sha256": INPUT_SHA,
        "tasks_sha256": inputs.sha(prepared / "tasks.jsonl"),
        "jobs": jobs,
        "planned_episodes": len(jobs),
        "planned_per_policy": 16,
        "parent_tasks": 8,
        "seeds": list(SEEDS),
        "model": str(BASE),
        "model_manifest_sha256": inputs.sha(BASE / "local-research-manifest.json"),
        "adapters": None,
        "sampling": {"temperature": 0.5, "top_p": 1.0, "top_k": 0},
        "max_global_calls": 96,
        "max_global_output_tokens": 8192,
        "max_new_tokens": 256,
        "input_plus_output_limit": 8192,
        "truncation": False,
        "max_agent_depth": {p: 0 if p == "flat" else 2 for p in policies},
        "root_depth": 0,
        "max_native_calls": 96 * len(jobs),
        "budget_seconds": hours * 3600,
        "optimizer": None,
        "seed_rule": "repeat_seed + int(sha256(task_id)[:8],16) +1000*global_call_index",
        "prompt_difference": "Same instructions/state schema; max_agent_depth0 versus2. "
        "Delegation then creates different task goals/context histories, "
        "not matched internal calls.",
        "child_return": "Actual finish message/status/inventory only; no score or free LLM summary",
        "trusted_source": bridge.trusted_provenance(),
        "source_sha256": {
            str(p.resolve()): inputs.sha(p)
            for p in (
                Path(__file__),
                Path(bridge.__file__),
                Path(inputs.__file__),
                Path(probe.__file__),
                Path(probe.runtime.__file__),
                Path(probe.campaign.__file__),
            )
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        },
        "caveat": "Shared-world compositional transfer/interface qualification, "
        "not RAO reproduction; "
        "eight tasks, two seeds are not independent parents.",
    }
    if profile == "instruction_control":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            BASE, local_files_only=True, trust_remote_code=False
        )
        initial_lengths = [
            len(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": bridge.initial_prompt(task, "flat") + INSTRUCTION_REMINDER,
                        }
                    ],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            for task in tasks
        ]
        if max(initial_lengths) + 256 > 8192:
            raise ValueError("instruction-control initial context exceeds8192")
        plan.update(
            schema="textcraft-native-instruction-control-v1",
            profile=profile,
            instruction_reminder=INSTRUCTION_REMINDER,
            prompt_difference="Original flat public prompt plus fixed trailing reminder only; "
            "same full history, parser, inventory, native tools and sampling. "
            "Post-hoc qualification; original three missing flat outcomes remain unknown.",
            baseline_output=str(ROOT / "textcraft-pilot-001"),
            baseline_plan_sha256=inputs.sha(ROOT / "textcraft-pilot-001/PLAN.json"),
            baseline_analysis_sha256=inputs.sha(ROOT / "analysis-textcraft-pilot-001.json"),
            initial_token_audit={
                "prompt_tokens": initial_lengths,
                "max_prompt_plus_cap": max(initial_lengths) + 256,
            },
        )
    if not persist:
        return plan, tasks
    if (output / "PLAN.json").exists():
        if json.loads((output / "PLAN.json").read_text()) != plan:
            raise ValueError("immutable PLAN changed")
    else:
        save(output / "PLAN.json", plan)
    return plan, tasks


def summarize(output, plan):
    rows = {p.stem: json.loads(p.read_text()) for p in (output / "episodes").glob("*.json")}
    calls = [json.loads(p.read_text()) for p in (output / "calls").glob("*.json")]
    if set(rows) - {j["episode_id"] for j in plan["jobs"]}:
        raise ValueError("unplanned episode")
    groups = {}
    for policy in sorted({j.get("condition", j["policy"]) for j in plan["jobs"]}):
        selected = [r for r in rows.values() if r.get("condition", r["policy"]) == policy]
        groups[policy] = {
            "planned": 16,
            "recorded": len(selected),
            "observed": sum(r["observed"] for r in selected),
            "missing_or_unknown": 16 - sum(r["observed"] for r in selected),
            "won": sum(r["observed"] and r["native_score"] == 1 for r in selected),
            "status_counts": dict(Counter(r["status"] for r in selected)),
            "max_depth_counts": dict(Counter(r["max_depth_reached"] for r in selected)),
            "errors": dict(sum((Counter(r["errors"]) for r in selected), Counter())),
            "physical_cost": cost([c for c in calls if c["condition"] == policy]),
        }
    return {
        "planned_episodes": len(plan["jobs"]),
        "recorded_episodes": len(rows),
        "groups": groups,
        "all_slots_recorded": len(rows) == len(plan["jobs"]),
        "physical_cost": cost(calls),
        "unresolved_starts": sorted(
            p.stem
            for p in (output / "starts").glob("*.json")
            if p.stem not in {c["call_id"] for c in calls}
        ),
    }


def run(args, prepared_run=None, adapter=None):
    global STOP
    STOP = False
    output = args.output.resolve()
    profile = getattr(args, "profile", "original")
    plan, tasks = prepared_run or prepare(args.prepared, output, args.hours, profile=profile)
    if args.prepare_only:
        print(json.dumps({"planned_episodes": len(plan["jobs"]), "GPU_loaded": False}))
        return
    if list(output.glob("OWNER-*.json")) or any(
        (output / d).exists() for d in ("calls", "episodes", "nodes")
    ):
        raise ValueError("existing scientific attempt; no silent retry/resume")
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + plan["budget_seconds"], lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient allocation margin")
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    failure, model, client = None, None, None
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
        global STOP
        STOP = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            BASE, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, adapter["path"], adapter_name="textcraft_action", is_trainable=False
            )
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        save(
            output / "LOAD.json",
            {
                "model": str(BASE),
                "adapter": adapter,
                "optimizer_created": False,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "gpu": torch.cuda.get_device_name(),
                "cuda": torch.version.cuda,
            },
        )
        client = NativeClient(
            model,
            tokenizer,
            output,
            deadline,
            plan["model_manifest_sha256"],
            inputs.sha(output / "PLAN.json"),
            adapter=adapter,
        )
        world, lookup = bridge.load_world(), {t["id"]: t for t in tasks}
        for i, job in enumerate(plan["jobs"]):
            if STOP or time.time() >= deadline - 5 or (output / "STOP").exists():
                STOP = True
                break
            record = episode(
                lookup[job["task_id"]], job, client, world, output, deadline, profile=profile
            )
            if not record["observed"]:
                raise RuntimeError(record["failure"])
            if (i + 1) % 4 == 0:
                probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del client, model
        gc.collect()
        torch.cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        save(
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
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--profile", choices=("original", "instruction_control"), default="original"
    )
    run(parser.parse_args())
