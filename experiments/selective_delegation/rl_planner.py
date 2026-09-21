"""Bounded fresh on-policy planner RLOO with frozen isolated-execution helpers."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import math
import os
import random
import signal
import sys
import time
import uuid
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import eval_planner as evaluation
import probe
import train_planner

SEED = 2026092108
TEMPERATURE = 0.8
DENOMINATOR = 64
STOP = False
HELPER_MODES = ("base", "format_reminder", "trained_helper")


def parent_schedule(training, updates, schedule="repeated"):
    maximum = 16 if schedule == "consecutive" else 4
    if schedule not in ("repeated", "consecutive") or not 1 <= updates <= maximum:
        raise ValueError("unknown bounded parent schedule")
    ordered = sorted(training, key=lambda case: case["id"])
    if (
        len(ordered) != 256
        or len({c["id"] for c in ordered}) != 256
        or any(c["split"] != "train" for c in ordered)
    ):
        raise ValueError("expected256 unique training parents")
    random.Random(SEED).shuffle(ordered)
    return [
        ordered[16 * i : 16 * (i + 1)] if schedule == "consecutive" else ordered[:16]
        for i in range(updates)
    ]


def continuation_schedule(training):
    """Fixed global updates1..24; only17..24 execute in the new output."""
    prefix = parent_schedule(training, 16, "consecutive")
    second = sorted(training, key=lambda case: case["id"])
    random.Random(SEED + 1).shuffle(second)
    return prefix + [second[16 * i : 16 * (i + 1)] for i in range(8)]


def validate_run_bounds(updates, hours, schedule, continuation):
    if continuation:
        if updates != 24 or not 0 < hours <= 1.5 or schedule != "consecutive":
            raise ValueError(
                "continuation requires --updates24, consecutive schedule and <=1.5 hours"
            )
    elif not 1 <= updates <= 16 or not 0 < hours <= 3:
        raise ValueError("bounded run requires 1..16 updates and <=3 hours")


def restore_optimizer_rng(parameters, checkpoint, expected_step):
    """Restore committed Adam moments and all saved RNGs, never fresh-Adam warmstart."""
    import torch

    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.0)
    saved = torch.load(
        checkpoint / "optimizer.pt", map_location=parameters[0].device, weights_only=True
    )
    actual_groups = [{k: v for k, v in g.items() if k != "params"} for g in saved["param_groups"]]
    expected_groups = [
        {k: v for k, v in g.items() if k != "params"}
        for g in optimizer.state_dict()["param_groups"]
    ]
    if actual_groups != expected_groups:
        raise ValueError("saved optimizer hyperparameters differ")
    if len(saved["state"]) != len(parameters) or any(
        float(state["step"]) != expected_step for state in saved["state"].values()
    ):
        raise ValueError("saved optimizer step/parameter inventory differs")
    optimizer.load_state_dict(saved)
    rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    return optimizer


def build_helper_contract(mode="base", adapter=None):
    if mode not in HELPER_MODES:
        raise ValueError("unknown helper contract")
    if mode != "trained_helper" and adapter is not None:
        raise ValueError("helper adapter only belongs to trained_helper contract")
    contract = {
        "mode": mode,
        "model": str(evaluation.planner.BASE),
        "base_manifest_sha256": probe.campaign.sha(
            evaluation.planner.BASE / "local-research-manifest.json"
        ),
    }
    if mode == "format_reminder":
        import eval_helper

        contract.update(
            format_reminder=eval_helper.FORMAT_REMINDER,
            reminder_source=str(Path(eval_helper.__file__).resolve()),
            reminder_source_sha256=probe.campaign.sha(Path(eval_helper.__file__)),
        )
    elif mode == "trained_helper":
        if adapter is None:
            raise ValueError("trained_helper requires a committed helper adapter")
        adapter = Path(adapter).resolve()
        binding = evaluation.adapter_identity(adapter)
        state = json.loads((adapter / "STATE.json").read_text())
        training_path = adapter.parent / "PLAN.json"
        training = json.loads(training_path.read_text())
        if training.get("role") != "helper" or training.get("model") != contract["model"]:
            raise ValueError("helper adapter training role/base identity differs")
        if training.get("model_manifest_sha256") != contract["base_manifest_sha256"]:
            raise ValueError("helper training base manifest differs")
        if state["step"] != 36 or state.get("epoch") != 1:
            raise ValueError("helper contract requires completed one-epoch checkpoint36")
        contract.update(
            adapter=str(adapter),
            adapter_binding=binding,
            training_plan_sha256=probe.campaign.sha(training_path),
            weights_frozen=True,
            separate_model=True,
        )
    return contract


def rollout_helper_prompt(case, question, contract):
    if contract["mode"] not in HELPER_MODES:
        raise ValueError("unknown helper contract")
    return evaluation.isolated_helper_prompt(case, question) + (
        contract["format_reminder"] if contract["mode"] == "format_reminder" else ""
    )


def component_identity(plan):
    return probe.runtime.digest(
        {key: plan[key] for key in ("helper_contract", "parent_schedule", "case_ids_by_update")}
    )


def validate_component_resume(state, plan):
    if state.get("component_identity") != component_identity(plan):
        raise ValueError("checkpoint component/helper/schedule identity differs")
    if "continuation" in plan and state.get("continuation_identity") != probe.runtime.digest(
        plan["continuation"]
    ):
        raise ValueError("checkpoint continuation ancestry differs")


def freeze_helper_model(helper, root):
    root_ids = {id(p) for p in root.parameters()}
    if helper is root or any(id(p) in root_ids for p in helper.parameters()):
        raise ValueError("helper must be a separate model without shared parameter objects")
    helper.eval()
    for parameter in helper.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    helper.gradient_checkpointing_disable()
    helper.config.use_cache = True


def assert_frozen_helper(helper, optimizer_parameters):
    if helper is None:
        return
    optimizer_ids = {id(p) for p in optimizer_parameters}
    if any(
        p.requires_grad or p.grad is not None or id(p) in optimizer_ids for p in helper.parameters()
    ):
        raise ValueError("helper must remain frozen and outside root optimizer")


def model_for_role(root, helper, role, helper_mode):
    if role not in ("root", "helper", "final") or helper_mode not in HELPER_MODES:
        raise ValueError("unknown model role/helper contract")
    if role == "helper" and helper_mode == "trained_helper":
        if helper is None:
            raise ValueError("trained helper model missing")
        return helper, True, "helper"
    return root, role == "root", "root"


class MissingGroup(RuntimeError):
    """A generation exception makes the group missing; never turn it into reward zero."""

    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = record


def rloo(rewards):
    if len(rewards) != 4 or any(r not in (0, 1) for r in rewards):
        raise ValueError("RLOO requires four exact binary rewards")
    return [r - (sum(rewards) - r) / 3 for r in rewards]


def token_logps(logits, targets):
    import torch

    values = torch.log_softmax(logits.float() / TEMPERATURE, dim=-1)
    return values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def policy_loss(logps, advantage):
    return -float(advantage) * logps.sum() / DENOMINATOR


def planner_parameters(model):
    import torch

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in n or p.dtype != torch.float32 for n, p in named):
        raise ValueError("only FP32 LoRA parameters may train")
    return [p for _, p in named]


def seed_schedule(update, parent_index, candidate):
    base = SEED + update * 100000 + parent_index * 1000
    return {"root": base + candidate, "downstream": base + 100}


def batch_diagnostics(groups):
    if len(groups) != 16 or any(len(group) != 4 for group in groups):
        raise ValueError("complete fixed 16-by-4 batch required")
    rows = []
    for group in groups:
        rewards = [r["reward"] for r in group]
        mean = sum(rewards) / 4
        distinct = {json.dumps(r["plan"], sort_keys=True) for r in group if r["plan_valid"]}
        valid_rewards = {r["reward"] for r in group if r["plan_valid"]}
        scored_rewards = {r["reward"] for r in group if r.get("status") == "scored"}
        rows.append(
            {
                "rewards": rewards,
                "advantages": rloo(rewards),
                "valid_plans": sum(r["plan_valid"] for r in group),
                "distinct_valid_plans": len(distinct),
                "mean_reward": mean,
                "reward_variance": sum((r - mean) ** 2 for r in rewards) / 4,
                "mixed_valid_plan_rewards": len(valid_rewards) == 2,
                "root_protocol_only_variation": 0 < sum(rewards) < 4 and len(valid_rewards) < 2,
                "fully_scored_content_variation": len(scored_rewards) == 2,
                "valid_plan_variation_includes_downstream_failures": len(valid_rewards) == 2
                and any(r["plan_valid"] and r.get("status") != "scored" for r in group),
                "qualifies": len(distinct) >= 2 and len(valid_rewards) == 2,
            }
        )
    qualifying = sum(row["qualifies"] for row in rows)
    return {
        "episodes": 64,
        "groups": rows,
        "qualifying_groups": qualifying,
        "admitted": qualifying >= 2,
        "mean_reward": sum(r["mean_reward"] for r in rows) / 16,
    }


def guard(deadline, reserve=0):
    if STOP or time.time() >= deadline - reserve:
        raise TimeoutError("bounded planner RL deadline or stop")


def root_logps(model, record):
    import torch

    prefix, emitted = record["input_token_ids"], record["output_token_ids"]
    if not prefix or not emitted:
        raise ValueError("empty root prompt or emitted trajectory")
    ids = torch.tensor([prefix + emitted[:-1]], device=model.device)
    target = torch.tensor([emitted], device=model.device)
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(emitted)).logits
    return token_logps(logits, target)


class Client:
    """No external server, no retries, no cached rollouts across policy updates."""

    def __init__(
        self,
        model,
        tokenizer,
        output,
        deadline,
        adapter_sha,
        *,
        helper_contract=None,
        helper_model=None,
    ):
        self.model, self.tokenizer = model, tokenizer
        self.output, self.deadline, self.adapter_sha = output, deadline, adapter_sha
        self.returned = 0
        self.helper_contract = helper_contract or {"mode": "base"}
        self.helper_model = helper_model

    def call(self, identity, prompt, role, seed, cap):
        import torch
        from transformers import GenerationConfig

        selected, enabled, instance = model_for_role(
            self.model, self.helper_model, role, self.helper_contract["mode"]
        )
        is_root = role == "root"
        adapter_sha = (
            self.adapter_sha
            if is_root
            else self.helper_contract["adapter_binding"]["adapter_model.safetensors"]
            if enabled
            else None
        )
        temperature = TEMPERATURE if is_root else 0.5
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        request = {
            "prompt": prompt,
            "input_token_ids": ids,
            "role": role,
            "model": str(evaluation.planner.BASE),
            "adapter_enabled": enabled,
            "adapter_sha256": adapter_sha,
            "model_instance": instance,
            "helper_contract": self.helper_contract,
            "seed": seed,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": cap,
            "max_time": 90.0,
        }
        path = self.output / "calls" / (identity + ".json")
        if path.exists():
            raise MissingGroup("refusing to regenerate or reuse an existing rollout call")
        record = {
            "call_id": identity,
            "request": request,
            "request_digest": probe.runtime.digest(request),
            "input_token_ids": ids,
            "role": role,
            "adapter_enabled": enabled,
            "adapter_sha256": adapter_sha,
            "model": str(evaluation.planner.BASE),
            "model_instance": instance,
            "available": False,
            "text": None,
            "usage": {"prompt_tokens": len(ids)},
            "started": time.time(),
        }
        probe.runtime.save(self.output / "starts" / (identity + ".json"), record)
        failure = None
        try:
            guard(self.deadline, 1)
            if len(ids) + cap > 8192:
                raise ValueError("context limit; no truncation")
            selected.eval()
            selected.gradient_checkpointing_disable()
            selected.config.use_cache = True
            torch.manual_seed(seed)
            if str(selected.device).startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            tensor = torch.tensor([ids], device=selected.device)
            config = GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                max_new_tokens=cap,
                max_time=min(90.0, self.deadline - time.time() - 1),
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=is_root,
            )
            with nullcontext() if enabled else selected.disable_adapter(), torch.no_grad():
                generated = selected.generate(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    generation_config=config,
                )
            actual = generated.sequences[0].detach().cpu().tolist()
            if actual[: len(ids)] != ids or len(actual) <= len(ids):
                raise ValueError("native generation prefix/continuation mismatch")
            emitted = actual[len(ids) :]
            record.update(
                output_token_ids=emitted,
                usage={"prompt_tokens": len(ids), "completion_tokens": len(emitted)},
                text=self.tokenizer.decode(
                    emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
            )
            eos = emitted[-1] == self.tokenizer.eos_token_id
            if not eos and len(emitted) < cap:
                raise TimeoutError("generation stopped before EOS or token cap")
            if is_root:
                record["generation_logps"] = [
                    float(torch.log_softmax(score.float(), -1)[0, token])
                    for score, token in zip(generated.scores, emitted, strict=True)
                ]
            record.update(available=True, terminated=eos, finish_reason="eos" if eos else "length")
            self.returned += 1
            del generated
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            record["error"] = failure
        record["ended"] = time.time()
        probe.runtime.save(path, record)
        print(
            json.dumps(
                {
                    "call_id": identity,
                    "available": record["available"],
                    "role": role,
                    "usage": record["usage"],
                    "error": failure,
                }
            ),
            flush=True,
        )
        if failure:
            raise MissingGroup(failure, record)
        return record


def rollout(client, case, update, parent_index, candidate):
    key = f"u{update:02d}-{case['id']}-c{candidate}"
    seeds = seed_schedule(update, parent_index, candidate)
    records = []
    row = {
        "episode_id": key,
        "case_id": case["id"],
        "update": update,
        "candidate": candidate,
        "seeds": seeds,
        "reward": 0,
        "plan_valid": False,
        "status": "root",
        "available": True,
    }
    missing = None
    try:
        root = client.call(
            key + "-root", evaluation.planner_prompt(case), "root", seeds["root"], 128
        )
        records.append(root)
        row["status"] = "invalid_plan"
        plan = evaluation.parse_plan(root["text"])
        row.update(plan=plan, plan_valid=True)
        answers, trace = [], []
        cap = 384 // len(plan["subquestions"])
        for index, question in enumerate(plan["subquestions"]):
            row["status"] = "invalid_dependency"
            resolved = evaluation.bind_question(question, answers)
            row["status"] = "helper"
            helper = client.call(
                key + f"-helper-{index + 1}",
                rollout_helper_prompt(case, resolved, client.helper_contract),
                "helper",
                seeds["downstream"] + index + 1,
                cap,
            )
            records.append(helper)
            row["status"] = "invalid_helper"
            answer = evaluation.parse_helper_answer(helper["text"])
            answers.append(answer)
            trace.append(
                {
                    "step": index + 1,
                    "question": question,
                    "resolved_question": resolved,
                    "answer": answer,
                }
            )
        row["helper_trace"] = trace
        report = {"execution": "isolated", "steps": trace}
        row["status"] = "final"
        final = client.call(
            key + "-final",
            evaluation.final_prompt(case, plan, report),
            "final",
            seeds["downstream"] + 2,
            128,
        )
        records.append(final)
        score = probe.grade(final["text"], case)
        row.update(
            score=score,
            reward=int(score["correct"]),
            status="scored" if score["valid"] else "invalid_final",
        )
    except MissingGroup as exc:
        if exc.record is not None:
            records.append(exc.record)
        row.update(available=False, reward=None, status="missing_generation", error=str(exc))
        missing = exc
    except (ValueError, TypeError) as exc:
        # Returned invalid plans, dependencies, helpers or finals have reward zero.
        row["error"] = f"{type(exc).__name__}: {exc}"
    row.update(call_ids=[r["call_id"] for r in records], cost=evaluation.cost(records))
    probe.runtime.save(client.output / "episodes" / (key + ".json"), row)
    if missing:
        raise missing
    return row, records[0]


def checkpoint_valid(path):
    commit = json.loads((path / "COMMIT.json").read_text())
    for name, digest in commit["files"].items():
        if probe.campaign.sha(path / name) != digest:
            raise ValueError("checkpoint changed: " + name)
    state = json.loads((path / "STATE.json").read_text())
    if commit["step"] != state["step"]:
        raise ValueError("checkpoint COMMIT/STATE step differs")
    return state


def validate_continuation(checkpoint, plan):
    """Validate the completed ancestor before giving its state a new schedule identity."""
    checkpoint = Path(checkpoint).resolve()
    ancestor = checkpoint.parent
    old = json.loads((ancestor / "PLAN.json").read_text())
    state = checkpoint_valid(checkpoint)
    commit = json.loads((checkpoint / "COMMIT.json").read_text())
    required = {
        "STATE.json",
        "optimizer.pt",
        "rng.pt",
        "adapter_model.safetensors",
        "adapter_config.json",
    }
    owners = {p.stem.removeprefix("OWNER-") for p in ancestor.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in ancestor.glob("TERMINAL-*.json")}
    terminal_rows = [json.loads(p.read_text()) for p in ancestor.glob("TERMINAL-*.json")]
    if (
        checkpoint.name != "checkpoint-0016"
        or state["step"] != 16
        or state.get("cursor") != 0
        or state.get("next_update") != 17
        or old.get("updates") != 16
        or old.get("parent_schedule") != "consecutive"
        or plan["updates"] != 24
        or old.get("helper_contract", {}).get("mode") != "trained_helper"
        or not required <= commit["files"].keys()
        or not owners
        or owners != terminals
        or not any(
            t.get("state") == "completed_updates"
            and t.get("optimizer_steps") == 16
            and t.get("failure") is None
            for t in terminal_rows
        )
        or any(
            p.name > checkpoint.name
            for p in ancestor.glob("checkpoint-*")
            if (p / "COMMIT.json").exists()
        )
    ):
        raise ValueError("continuation requires completed ancestor checkpoint16")
    validate_component_resume(state, old)
    if (
        len(plan["case_ids_by_update"]) != 24
        or old["case_ids_by_update"] != plan["case_ids_by_update"][:16]
        or state["case_ids"] != old["case_ids_by_update"][15]
    ):
        raise ValueError("ancestor schedule prefix differs")
    for key in (
        "schema",
        "parent_schedule",
        "helper_contract",
        "cases_sha256",
        "adapter",
        "adapter_binding",
        "model",
        "base_manifest_sha256",
        "seed",
        "parents_per_update",
        "candidates_per_parent",
        "denominator",
        "root_temperature",
        "helper_final_temperature",
        "learning_rate",
        "weight_decay",
        "clip",
        "caps",
        "objective",
        "admission",
        "frozen_helpers",
    ):
        if old.get(key) != plan.get(key):
            raise ValueError("continuation contract differs: " + key)
    if state.get("helper_contract") != old["helper_contract"]:
        raise ValueError("ancestor STATE helper contract differs")
    for package in ("torch", "transformers", "peft"):
        if old["environment"][package] != plan["environment"][package]:
            raise ValueError("continuation library version differs: " + package)
    for path, digest in old["dependencies"].items():
        if probe.campaign.sha(Path(path)) != digest:
            raise ValueError("ancestor sealed source differs: " + path)
    return {
        "ancestor_checkpoint": str(checkpoint),
        "ancestor_plan_sha256": probe.campaign.sha(ancestor / "PLAN.json"),
        "ancestor_state_sha256": probe.campaign.sha(checkpoint / "STATE.json"),
        "ancestor_commit_sha256": probe.campaign.sha(checkpoint / "COMMIT.json"),
        "checkpoint_files_sha256": commit["files"],
        "ancestor_source_sha256": old["dependencies"],
        "ancestor_terminal_sha256": {
            str(p): probe.campaign.sha(p) for p in sorted(ancestor.glob("TERMINAL-*.json"))
        },
        "starting_step": 16,
        "additional_updates": 8,
        "second_shuffle_seed": SEED + 1,
        "schedule_prefix_sha256": probe.runtime.digest(old["case_ids_by_update"]),
        "parent_occurrences": dict(Counter(p for b in plan["case_ids_by_update"] for p in b)),
        "new_parent_occurrences": dict(
            Counter(p for b in plan["case_ids_by_update"][16:] for p in b)
        ),
        "maximum_new_rollouts": 512,
        "maximum_new_calls": 5120,
    }, state


def select_restore(output, plan, resume, ancestor_state=None):
    committed = sorted(p for p in output.glob("checkpoint-*") if (p / "COMMIT.json").exists())
    if committed:
        if not resume:
            raise ValueError("explicit resume required for existing checkpoints")
        restored = committed[-1]
        state = checkpoint_valid(restored)
        validate_component_resume(state, plan)
        minimum = 17 if "continuation" in plan else 1
        if not minimum <= state["step"] <= plan["updates"]:
            raise ValueError("local checkpoint outside planned update boundaries")
        if "continuation" in plan and (
            restored.name != f"checkpoint-{state['step']:04d}"
            or state.get("cursor") != 0
            or state.get("next_update") != state["step"] + 1
            or state.get("case_ids") != plan["case_ids_by_update"][state["step"] - 1]
        ):
            raise ValueError("local continuation checkpoint cursor/schedule differs")
        return restored, state
    if "continuation" in plan:
        if ancestor_state is None or ancestor_state["step"] != 16:
            raise ValueError("validated ancestor state required")
        return Path(plan["continuation"]["ancestor_checkpoint"]), dict(ancestor_state)
    return None, {"step": 0, "cursor": 0}


def run(args):
    global STOP
    STOP = False
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    continuation = getattr(args, "continue_from", None)
    validate_run_bounds(
        args.updates, args.hours, getattr(args, "parent_schedule", "repeated"), bool(continuation)
    )
    output, adapter = args.output.resolve(), args.adapter.resolve()
    if continuation and output.is_relative_to(continuation.resolve().parent):
        raise ValueError("continuation output must be outside immutable ancestor")
    helper_contract = build_helper_contract(
        getattr(args, "helper_contract", "base"), getattr(args, "helper_adapter", None)
    )
    schedule_kind = getattr(args, "parent_schedule", "repeated")
    binding = evaluation.adapter_identity(adapter)
    if checkpoint_valid(adapter)["step"] != 48:
        raise ValueError("warmstart must be SFT checkpoint48")
    with args.cases.open() as stream:
        training = sorted(
            (c for c in map(json.loads, stream) if c["split"] == "train"), key=lambda c: c["id"]
        )
    if len(training) != 256:
        raise ValueError("expected the entire frozen256 training parent pool")
    scheduled_cases = (
        continuation_schedule(training)
        if continuation
        else parent_schedule(training, args.updates, schedule_kind)
    )
    manifest = evaluation.planner.BASE / "local-research-manifest.json"
    plan = {
        "schema": "fresh-planner-rloo-v1",
        "case_ids": list(dict.fromkeys(c["id"] for batch in scheduled_cases for c in batch)),
        "case_ids_by_update": [[c["id"] for c in batch] for batch in scheduled_cases],
        "parent_schedule": schedule_kind,
        "helper_contract": helper_contract,
        "cases_sha256": probe.campaign.sha(args.cases),
        "adapter": str(adapter),
        "adapter_binding": binding,
        "model": str(evaluation.planner.BASE),
        "base_manifest_sha256": probe.campaign.sha(manifest),
        "seed": SEED,
        "updates": args.updates,
        "budget_seconds": args.hours * 3600,
        "parents_per_update": 16,
        "candidates_per_parent": 4,
        "denominator": 64,
        "root_temperature": 0.8,
        "helper_final_temperature": 0.5,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "clip": 1.0,
        "caps": {"root": 128, "helper_total": 384, "final": 128},
        "objective": "-sum(leave-other-three-out reward advantage * sum root logp(T=.8))/64",
        "admission": "at least2/16 groups with >=2 distinct valid plans and mixed exact-EM "
        "among valid plans; invalid plans remain reward zero in all64 loss trajectories",
        "frozen_helpers": "Separate frozen helper adapter; final uses root instance with adapter "
        "disabled"
        if helper_contract["mode"] == "trained_helper"
        else "adapter disabled for every helper/final; base parameters frozen",
        "dependencies": {
            str(p): probe.campaign.sha(p)
            for p in (
                Path(__file__),
                Path(evaluation.__file__),
                Path(evaluation.planner.__file__),
                Path(train_planner.__file__),
                Path(probe.__file__),
                probe.MUSIQUE / "metrics/answer.py",
            )
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
    }
    ancestor_state = None
    if continuation:
        plan["continuation"], ancestor_state = validate_continuation(continuation, plan)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if not args.resume or json.loads((output / "PLAN.json").read_text()) != plan:
            raise ValueError("explicit resume and unchanged plan required")
    else:
        probe.runtime.save(output / "PLAN.json", plan)
    restored, state = select_restore(output, plan, args.resume, ancestor_state)
    if (output / f"batch-{state['step'] + 1:04d}").exists():
        raise ValueError("uncommitted rollout batch exists; no implicit retries or regeneration")
    spent = sum(
        json.loads(p.read_text())["elapsed_seconds"] for p in output.glob("TERMINAL-*.json")
    )
    owners = {p.stem.removeprefix("OWNER-") for p in output.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in output.glob("TERMINAL-*.json")}
    if owners != terminals:
        raise ValueError("previous owner unresolved")
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + max(0, args.hours * 3600 - spent), lease - 600)
    guard(deadline, 180)
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    model = optimizer = helper_model = None
    failure, status = None, "starting"
    try:
        probe.runtime.save(
            output / f"OWNER-{invocation}.json",
            {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "started": started,
                "deadline": deadline,
                "allocation_end": lease,
            },
        )

        def stop(*_):
            global STOP
            STOP = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        tokenizer = AutoTokenizer.from_pretrained(
            plan["model"], local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base, restored or adapter, is_trainable=True, autocast_adapter_dtype=True
        )
        parameters = planner_parameters(model)
        if len(parameters) != 504:
            raise ValueError("expected504 rank8 Qwen LoRA tensors")
        if helper_contract["mode"] == "trained_helper":
            helper_base = AutoModelForCausalLM.from_pretrained(
                plan["model"],
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map={"": "cuda:0"},
            )
            helper_model = PeftModel.from_pretrained(
                helper_base,
                helper_contract["adapter"],
                is_trainable=False,
                autocast_adapter_dtype=True,
            )
            freeze_helper_model(helper_model, model)
            assert_frozen_helper(helper_model, parameters)
        if restored:
            optimizer = restore_optimizer_rng(parameters, restored, state["step"])
        model.enable_input_require_grads()
        probe.runtime.save(
            output / f"LOAD-{invocation}.json",
            {
                "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
                "trainable_parameters": sum(p.numel() for p in parameters),
                "all_non_lora_frozen": all(
                    not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n
                ),
                "helpers_adapter_disabled": helper_model is None,
                "helper_weights_frozen": True,
                "helper_contract": helper_contract,
                "helper_trainable_parameters": sum(
                    p.numel() for p in helper_model.parameters() if p.requires_grad
                )
                if helper_model
                else 0,
                "base_manifest_sha256": plan["base_manifest_sha256"],
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                **(
                    {
                        "restored_checkpoint": str(restored),
                        "restored_global_step": state["step"],
                        "optimizer_and_rng_restored": True,
                        "continuation": plan["continuation"],
                    }
                    if continuation
                    else {}
                ),
            },
        )
        for update in range(state["step"] + 1, args.updates + 1):
            cases = scheduled_cases[update - 1]
            guard(deadline, 180)
            if (output / "STOP").exists():
                status = "stopped_at_update_boundary"
                break
            batch_dir = output / f"batch-{update:04d}"
            batch_dir.mkdir(exist_ok=False)
            policy_path = restored or adapter
            client = Client(
                model,
                tokenizer,
                batch_dir,
                deadline,
                probe.campaign.sha(policy_path / "adapter_model.safetensors"),
                helper_contract=helper_contract,
                helper_model=helper_model,
            )
            groups, roots = [], []
            for index, case in enumerate(cases):
                group = []
                for candidate in range(4):
                    row, root = rollout(client, case, update, index, candidate)
                    group.append(row)
                    roots.append(root)
                    probe.campaign.snapshot(
                        output / "STATUS.json",
                        {
                            "state": "collecting",
                            "update": update,
                            "episodes": index * 4 + candidate + 1,
                            "returned_calls": client.returned,
                            "updated": time.time(),
                        },
                    )
                groups.append(group)
            diagnostics = batch_diagnostics(groups)
            probe.runtime.save(batch_dir / "BATCH.json", diagnostics)
            if not diagnostics["admitted"]:
                status = "admission_failed_no_update"
                break
            guard(deadline, 180)
            if [id(p) for p in planner_parameters(model)] != [id(p) for p in parameters]:
                raise ValueError("root trainability changed during rollout")
            assert_frozen_helper(helper_model, parameters)
            model.eval()
            model.config.use_cache = False
            model.gradient_checkpointing_disable()
            before = []
            with torch.no_grad():
                for record in roots:
                    guard(deadline, 120)
                    before.append(root_logps(model, record).detach().cpu().flatten().tolist())
            probe.runtime.save(
                batch_dir / "BEFORE_LOGPS.json",
                {
                    "call_ids": [r["call_id"] for r in roots],
                    "logps": before,
                    "generation_full_forward_max_abs_difference": max(
                        abs(a - b)
                        for r, values in zip(roots, before, strict=True)
                        for a, b in zip(r["generation_logps"], values, strict=True)
                    ),
                },
            )
            if optimizer is None:
                optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.0)
            initial = [p.detach().cpu().clone() for p in parameters]
            optimizer.zero_grad(set_to_none=True)
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            advantages = [a for group in diagnostics["groups"] for a in group["advantages"]]
            objective, replay_gap = 0.0, 0.0
            for index, (record, advantage) in enumerate(zip(roots, advantages, strict=True)):
                guard(deadline, 90)
                if advantage == 0:
                    continue
                values = root_logps(model, record)
                if not torch.isfinite(values).all():
                    raise ValueError("nonfinite root logprobs")
                gap = max(
                    abs(a - b)
                    for a, b in zip(
                        values.detach().cpu().flatten().tolist(), before[index], strict=True
                    )
                )
                replay_gap = max(replay_gap, gap)
                # Record finite BF16 train/eval kernel discrepancies; no unqualified tiny
                # tolerance gate that would prevent the exploratory update.
                loss = policy_loss(values, advantage)
                objective += float(loss.detach())
                loss.backward()
            assert_frozen_helper(helper_model, parameters)
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("zero or nonfinite admitted gradient")
            optimizer.step()
            delta = math.sqrt(
                sum(
                    float((p.detach().cpu() - old).double().square().sum())
                    for p, old in zip(parameters, initial, strict=True)
                )
            )
            if not math.isfinite(delta) or delta <= 0:
                raise ValueError("adapter failed finite nonzero update")
            state = {
                "step": update,
                "cursor": 0,
                "next_update": update + 1,
                "case_ids": [c["id"] for c in cases],
                "component_identity": component_identity(plan),
                "helper_contract": helper_contract,
                "batch_sha256": probe.campaign.sha(batch_dir / "BATCH.json"),
                "objective": objective,
                "gradient_norm": norm,
                "adapter_l2_delta": delta,
                "train_eval_replay_max_abs_difference": replay_gap,
                "mean_reward": diagnostics["mean_reward"],
                "qualifying_groups": diagnostics["qualifying_groups"],
                **(
                    {"continuation_identity": probe.runtime.digest(plan["continuation"])}
                    if continuation
                    else {}
                ),
            }
            restored = train_planner.save_checkpoint(model, optimizer, output, state)
            model.eval()
            model.gradient_checkpointing_disable()
            after = []
            with torch.no_grad():
                for record in roots:
                    guard(deadline, 5)
                    after.append(root_logps(model, record).detach().cpu().flatten().tolist())
            probe.runtime.save(
                batch_dir / "AFTER_LOGPS.json",
                {
                    "call_ids": [r["call_id"] for r in roots],
                    "logps": after,
                    "adapter_sha256": probe.campaign.sha(restored / "adapter_model.safetensors"),
                },
            )
            probe.campaign.snapshot(
                output / "STATUS.json",
                {
                    "state": "updated",
                    **state,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "updated": time.time(),
                },
            )
            status = "completed_updates"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        status = "missing_group_halt" if isinstance(exc, MissingGroup) else "failed_or_capped"
        raise
    finally:
        try:
            if model is not None:
                del model
            if helper_model is not None:
                del helper_model
            gc.collect()
            torch.cuda.empty_cache()
            calls = [json.loads(p.read_text()) for p in output.glob("batch-*/calls/*.json")]
            starts = [json.loads(p.read_text()) for p in output.glob("batch-*/starts/*.json")]
            known = {r["call_id"] for r in calls}
            receipt = {
                "state": status,
                "optimizer_steps": state["step"],
                "failure": failure,
                "physical_cost": evaluation.cost(calls),
                "planned_max_rollouts": (args.updates - (16 if continuation else 0)) * 64,
                **(
                    {
                        "starting_global_step": 16,
                        "additional_optimizer_steps": state["step"] - 16,
                        "planned_max_calls": 5120,
                        "new_committed_parent_occurrences": dict(
                            Counter(
                                p for b in plan["case_ids_by_update"][16 : state["step"]] for p in b
                            )
                        ),
                    }
                    if continuation
                    else {}
                ),
                "unresolved_started_attempts": sum(s["call_id"] not in known for s in starts),
                "elapsed_seconds": time.time() - started,
                "ended": time.time(),
                "deadline": deadline,
            }
            probe.runtime.save(output / f"TERMINAL-{invocation}.json", receipt)
            probe.campaign.snapshot(output / "STATUS.json", receipt)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--continue-from",
        type=Path,
        help="fork completed RL checkpoint16; requires updates24/consecutive/hours<=1.5",
    )
    parser.add_argument("--helper-contract", choices=HELPER_MODES, default="base")
    parser.add_argument("--helper-adapter", type=Path)
    parser.add_argument(
        "--parent-schedule", choices=("repeated", "consecutive"), default="repeated"
    )
    run(parser.parse_args())
