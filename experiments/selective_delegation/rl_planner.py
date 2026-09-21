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
from contextlib import nullcontext
from pathlib import Path

import eval_planner as evaluation
import probe
import train_planner

SEED = 2026092108
TEMPERATURE = 0.8
DENOMINATOR = 64
STOP = False


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

    def __init__(self, model, tokenizer, output, deadline, adapter_sha):
        self.model, self.tokenizer = model, tokenizer
        self.output, self.deadline, self.adapter_sha = output, deadline, adapter_sha
        self.returned = 0

    def call(self, identity, prompt, role, seed, cap):
        import torch
        from transformers import GenerationConfig

        enabled = role == "root"
        temperature = TEMPERATURE if enabled else 0.5
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
            "adapter_sha256": self.adapter_sha if enabled else None,
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
            self.model.eval()
            self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            tensor = torch.tensor([ids], device=self.model.device)
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
                output_scores=enabled,
            )
            with nullcontext() if enabled else self.model.disable_adapter(), torch.no_grad():
                generated = self.model.generate(
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
            if enabled:
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
                evaluation.isolated_helper_prompt(case, resolved),
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
    return json.loads((path / "STATE.json").read_text())


def run(args):
    global STOP
    STOP = False
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not 1 <= args.updates <= 4 or not 0 < args.hours <= 3:
        raise ValueError("bounded run requires 1..4 updates and <=3 hours")
    output, adapter = args.output.resolve(), args.adapter.resolve()
    binding = evaluation.adapter_identity(adapter)
    if checkpoint_valid(adapter)["step"] != 48:
        raise ValueError("warmstart must be SFT checkpoint48")
    with args.cases.open() as stream:
        training = sorted(
            (c for c in map(json.loads, stream) if c["split"] == "train"), key=lambda c: c["id"]
        )
    if len(training) != 256:
        raise ValueError("expected the entire frozen256 training parent pool")
    random.Random(SEED).shuffle(training)
    cases = training[:16]
    manifest = evaluation.planner.BASE / "local-research-manifest.json"
    plan = {
        "schema": "fresh-planner-rloo-v1",
        "case_ids": [c["id"] for c in cases],
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
        "frozen_helpers": "adapter disabled for every helper/final; base parameters frozen",
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
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if not args.resume or json.loads((output / "PLAN.json").read_text()) != plan:
            raise ValueError("explicit resume and unchanged plan required")
    else:
        probe.runtime.save(output / "PLAN.json", plan)
    committed = sorted(p for p in output.glob("checkpoint-*") if (p / "COMMIT.json").exists())
    restored = committed[-1] if args.resume and committed else None
    state = checkpoint_valid(restored) if restored else {"step": 0, "cursor": 0}
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
    model = optimizer = None
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
        if restored:
            optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.0)
            optimizer.load_state_dict(
                torch.load(restored / "optimizer.pt", map_location="cuda:0", weights_only=True)
            )
            rng = torch.load(restored / "rng.pt", map_location="cpu", weights_only=True)
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
        model.enable_input_require_grads()
        probe.runtime.save(
            output / f"LOAD-{invocation}.json",
            {
                "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
                "trainable_parameters": sum(p.numel() for p in parameters),
                "all_non_lora_frozen": all(
                    not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n
                ),
                "helpers_adapter_disabled": True,
                "base_manifest_sha256": plan["base_manifest_sha256"],
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        for update in range(state["step"] + 1, args.updates + 1):
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
                "case_ids": plan["case_ids"],
                "batch_sha256": probe.campaign.sha(batch_dir / "BATCH.json"),
                "objective": objective,
                "gradient_norm": norm,
                "adapter_l2_delta": delta,
                "train_eval_replay_max_abs_difference": replay_gap,
                "mean_reward": diagnostics["mean_reward"],
                "qualifying_groups": diagnostics["qualifying_groups"],
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
                "planned_max_rollouts": args.updates * 64,
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
    run(parser.parse_args())
