"""Finite paired-sufficiency RLOO and matched-dose SFT; explicit sampled/update cursors."""

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import random
import signal
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import eval_sufficiency
import rl_planner as rl
import sufficiency_probe as baseline
import train_planner as recipe
import train_sufficiency as sft

probe = rl.probe
BASE = baseline.native.evaluation.planner.BASE
SEED = 2026092194
STOP = False


def pair_reward(positive, predictions, available=True):
    if not available:
        raise ValueError("unavailable pair is unknown, never reward zero")
    return int(baseline.group_score(positive, predictions)["em"])


def pair_loss(model, records, advantage):
    if len(records) != 2:
        raise ValueError("two separate public responses per paired candidate required")
    return -float(advantage) * sum(rl.root_logps(model, r).sum() for r in records) / 64


def marginal_credit(positive, negative):
    if len(positive) != 4 or len(negative) != 4:
        raise ValueError("four marginal successes per side required")
    pmean, nmean = sum(positive) / 4, sum(negative) / 4
    return [
        (
            nmean * (p - (sum(positive) - p) / 3),
            pmean * (n - (sum(negative) - n) / 3),
        )
        for p, n in zip(positive, negative, strict=True)
    ]


def training_reward(pair, reward_objective="product"):
    if reward_objective == "product":
        return pair["reward"]
    if reward_objective == "additive":
        return (pair["positive_success"] + pair["negative_success"]) / 2
    raise ValueError("unknown reward objective")


def leave_one_out(rewards):
    if len(rewards) != 4:
        raise ValueError("four rewards per parent required")
    return [value - (sum(rewards) - value) / 3 for value in rewards]


def response_advantages(groups, estimator="diagonal", reward_objective="product"):
    if estimator not in ("diagonal", "pairing_mean"):
        raise ValueError("unknown estimator")
    if reward_objective not in ("product", "additive"):
        raise ValueError("unknown reward objective")
    if estimator == "pairing_mean" and reward_objective != "product":
        raise ValueError("pairing_mean requires product reward")
    result = []
    for group in groups:
        if len(group) != 4:
            raise ValueError("four paired candidates per parent required")
        if estimator == "diagonal":
            rewards = [training_reward(pair, reward_objective) for pair in group]
            result.append(
                [
                    (value, value)
                    for value in (
                        rl.rloo(rewards)
                        if reward_objective == "product"
                        else leave_one_out(rewards)
                    )
                ]
            )
        else:
            result.append(
                marginal_credit(
                    [pair["positive_success"] for pair in group],
                    [pair["negative_success"] for pair in group],
                )
            )
    return result


def advance(state, advantages):
    if len(advantages) != 16:
        raise ValueError("complete16-parent batch required")
    if all(
        len(group) == 4 and all(isinstance(value, (int, float)) for value in group)
        for group in advantages
    ):
        advantages = [[(value, value) for value in rl.rloo(group)] for group in advantages]
    effective = sum(any(value != 0 for pair in group for value in pair) for group in advantages)
    return {
        **state,
        "sample_cursor": state["sample_cursor"] + 1,
        "step": state["step"] + int(effective > 0),
        "zero_streak": 0 if effective else state["zero_streak"] + 1,
        "effective_groups": effective,
    }


def finished(state):
    return state["sample_cursor"] >= 8 or state["zero_streak"] >= 4


def check(deadline, reserve=0):
    if STOP or time.time() >= deadline - reserve:
        raise TimeoutError("bounded owner stopped/deadline; incomplete work is unknown")


def paired_cases(cases, parents):
    lookup = {}
    for parent in parents:
        pair = sorted(
            [c for c in cases if c["parent_id"] == parent], key=lambda c: not c["answerable"]
        )
        if len(pair) != 2 or [c["answerable"] for c in pair] != [True, False]:
            raise ValueError("complete official TRAIN pair required")
        lookup[parent] = pair
    return lookup


def control_examples(cases, parents):
    pairs = paired_cases(cases, parents)
    return [
        {
            "id": f"{parent}-k{k}-v{v}",
            "parent_id": parent,
            "split": "train",
            "prompt": baseline.prompt(case),
            "target": json.dumps(
                {
                    "answerable": case["answerable"],
                    "answer": case["answer"] if case["answerable"] else "",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
        for parent in parents
        for k in range(4)
        for v, case in enumerate(pairs[parent])
    ]


class NativeClient:
    def __init__(self, model, tokenizer, output, deadline, adapter):
        self.model, self.tokenizer, self.output = model, tokenizer, Path(output)
        self.deadline, self.adapter = deadline, adapter
        self.returned, self.failed = 0, 0

    def call(self, identity, case, seed):
        import torch
        from transformers import GenerationConfig

        text = baseline.prompt(case)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        request = {
            "prompt": text,
            "input_token_ids": ids,
            "seed": seed,
            "role": "answer_sufficiency",
            "model": str(BASE),
            "adapter_enabled": True,
            "adapter": self.adapter,
            "sampling": {
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "max_new_tokens": 128,
                "max_time": 90.0,
            },
            "context_limit": 8192,
        }
        row = {
            "call_id": identity,
            "request": request,
            "request_digest": probe.runtime.digest(request),
            "input_token_ids": ids,
            "available": False,
            "text": None,
            "usage": {"prompt_tokens": len(ids)},
            "started": time.time(),
        }
        probe.runtime.save(self.output / "starts" / (identity + ".json"), row)
        try:
            check(self.deadline, 1)
            if len(ids) + 128 > 8192:
                raise ValueError("context limit; no truncation")
            self.model.eval()
            self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
            torch.manual_seed(seed)
            if str(self.model.device).startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            tensor = torch.tensor([ids], device=self.model.device)
            config = GenerationConfig(
                do_sample=True,
                temperature=0.8,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                max_new_tokens=128,
                max_time=min(90.0, self.deadline - time.time() - 1),
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    generation_config=config,
                )
            values = generated.sequences[0].detach().cpu().tolist()
            emitted = values[len(ids) :]
            if (
                values[: len(ids)] != ids
                or not 1 <= len(emitted) <= 128
                or len(generated.scores) != len(emitted)
            ):
                raise ValueError("native generation prefix/emitted score mismatch")
            logps = [
                float(torch.log_softmax(score.float(), dim=-1)[0, token])
                for score, token in zip(generated.scores, emitted, strict=True)
            ]
            row.update(
                available=True,
                output_token_ids=emitted,
                generation_logps=logps,
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
            self.failed += 1
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["ended"] = time.time()
        probe.runtime.save(self.output / "calls" / (identity + ".json"), row)
        probe.campaign.snapshot(
            self.output / "STATUS.json",
            {"returned": self.returned, "failed": self.failed, "updated": time.time()},
        )
        if not row["available"]:
            raise RuntimeError("native generation unavailable; preserved, no retry")
        if self.returned == 1 and row["ended"] - row["started"] > 90:
            raise RuntimeError("first real scientific response exceeded90seconds")
        return row


def collect(
    model,
    tokenizer,
    directory,
    deadline,
    cases,
    parents,
    cursor,
    adapter,
    reward_objective="product",
):
    client = NativeClient(model, tokenizer, directory, deadline, adapter)
    pairs, groups, rewards = paired_cases(cases, parents), [], []
    for index, parent in enumerate(parents):
        group, values = [], []
        for candidate in range(4):
            records, predictions = [], []
            for variant, case in enumerate(pairs[parent]):
                identity = f"p{index:02d}-k{candidate}-v{variant}"
                seed = SEED + cursor * 100000 + index * 1000 + candidate * 10 + variant
                record = client.call(identity, case, seed)
                records.append(record)
                try:
                    predictions.append(baseline.parse_output(record["text"]))
                except (ValueError, TypeError):
                    predictions.append(None)
            reward = pair_reward(pairs[parent][0], predictions)
            positive_success = pair_reward(
                pairs[parent][0], [predictions[0], {"answerable": False, "answer": ""}]
            )
            negative_success = int(predictions[1] is not None and not predictions[1]["answerable"])
            if reward != positive_success * negative_success:
                raise ValueError("saved marginal successes differ from official paired reward")
            item = {
                "parent_id": parent,
                "candidate": candidate,
                "reward": reward,
                "positive_success": positive_success,
                "negative_success": negative_success,
                "all_valid": all(p is not None for p in predictions),
                "predictions": predictions,
                "call_ids": [r["call_id"] for r in records],
            }
            item["training_reward"] = training_reward(item, reward_objective)
            probe.runtime.save(directory / "pairs" / f"p{index:02d}-k{candidate}.json", item)
            group.append({**item, "records": records})
            values.append(reward)
        groups.append(group)
        rewards.append(values)
    return groups, rewards


def optimize_rl(
    model, optimizer, groups, deadline, estimator="diagonal", reward_objective="product"
):
    import torch

    credits = response_advantages(groups, estimator, reward_objective)
    if not any(value for group in credits for pair in group for value in pair):
        return {"optimizer_called": False}
    parameters = rl.planner_parameters(model)
    before = [p.detach().cpu().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    total, audits = 0.0, []
    for group, group_credits in zip(groups, credits, strict=True):
        for pair, pair_credits in zip(group, group_credits, strict=True):
            for record, advantage in zip(pair["records"], pair_credits, strict=True):
                if advantage == 0:
                    continue
                check(deadline, 90)
                values = rl.root_logps(model, record)
                loss = -float(advantage) * values.sum() / 64
                loss.backward()
                total += float(loss.detach())
                audits.append(
                    {
                        "call_id": record["call_id"],
                        "advantage": advantage,
                        "before": values.detach().cpu().flatten().tolist(),
                        "generation": record["generation_logps"],
                        "record": record,
                    }
                )
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    if not norm > 0:
        raise ValueError("nonzero reward advantages yielded zero/nonfinite gradient")
    optimizer.step()
    delta = (
        sum(
            float((p.detach().cpu() - old).double().square().sum())
            for p, old in zip(parameters, before, strict=True)
        )
        ** 0.5
    )
    model.eval()
    model.gradient_checkpointing_disable()
    with torch.no_grad():
        for audit in audits:
            check(deadline, 30)
            audit["after"] = rl.root_logps(model, audit.pop("record")).cpu().flatten().tolist()
    return {
        "loss": total,
        "gradient_norm": norm,
        "parameter_delta_l2": delta,
        "credited_responses": len(audits),
        "likelihoods": audits,
        "generation_replay_max_abs_gap": max(
            abs(a - b)
            for row in audits
            for a, b in zip(row["before"], row["generation"], strict=True)
        ),
    }


def optimize_sft(model, optimizer, rows, deadline):
    import torch

    denominator = sum(len(r["target_ids"]) for r in rows)
    parameters = rl.planner_parameters(model)
    before = [p.detach().cpu().clone() for p in parameters]
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for row in rows:
        check(deadline, 90)
        ids = torch.tensor([row["input_ids"]], device=model.device)
        targets = torch.tensor([row["target_ids"]], device=model.device)
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=targets.shape[1]).logits
        loss = recipe.target_loss(logits, targets) / denominator
        loss.backward()
        loss_sum += float(loss.detach())
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    optimizer.step()
    return {
        "loss": loss_sum,
        "gradient_norm": norm,
        "examples": len(rows),
        "target_tokens": denominator,
        "parameter_delta_l2": sum(
            float((p.detach().cpu() - old).double().square().sum())
            for p, old in zip(parameters, before, strict=True)
        )
        ** 0.5,
    }


def save_boundary(model, optimizer, output, state):
    directory = output / "boundaries" / f"sample-{state['sample_cursor']:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = recipe.save_checkpoint(model, optimizer, directory, state)
    probe.runtime.save(
        directory / "BOUNDARY.json",
        {
            "checkpoint": str(checkpoint),
            "state": state,
            "commit_sha256": probe.campaign.sha(checkpoint / "COMMIT.json"),
        },
    )
    return checkpoint


def validate_boundary_state(state, previous=None):
    step, cursor, zero = (state[k] for k in ("step", "sample_cursor", "zero_streak"))
    if not (0 <= step <= cursor <= 8 and 0 <= zero <= min(cursor, 4)):
        raise ValueError("invalid finite boundary counters")
    if previous is None:
        if (step, cursor, zero) != (0, 0, 0):
            raise ValueError("initial boundary must precede all sampled/optimizer steps")
    else:
        delta = step - previous["step"]
        expected_zero = previous["zero_streak"] + 1 if delta == 0 else 0
        if (
            cursor != previous["sample_cursor"] + 1
            or delta not in (0, 1)
            or zero != expected_zero
            or finished(previous)
        ):
            raise ValueError("invalid sampled/optimizer boundary transition")


def boundary_inventory(output):
    result = []
    for directory in sorted((output / "boundaries").glob("sample-*")):
        if not (directory / "BOUNDARY.json").exists():
            raise ValueError("uncommitted sampling boundary; no implicit recovery")
        receipt = json.loads((directory / "BOUNDARY.json").read_text())
        checkpoint = Path(receipt["checkpoint"])
        if probe.campaign.sha(checkpoint / "COMMIT.json") != receipt["commit_sha256"]:
            raise ValueError("boundary commit changed")
        state = json.loads((checkpoint / "STATE.json").read_text())
        commit = json.loads((checkpoint / "COMMIT.json").read_text())
        if (
            commit["step"] != state["step"]
            or probe.campaign.sha(checkpoint / "STATE.json") != commit["files"]["STATE.json"]
        ):
            raise ValueError("committed STATE differs")
        if state != receipt["state"] or state["sample_cursor"] != len(result):
            raise ValueError("boundary state/order mismatch")
        validate_boundary_state(state, result[-1]["state"] if result else None)
        result.append(receipt)
    return result


def prepare(args):
    prepared, output, adapter = (
        args.prepared.resolve(),
        args.output.resolve(),
        args.adapter.resolve(),
    )
    estimator = getattr(args, "estimator", "diagonal")
    reward_objective = getattr(args, "reward_objective", "product")
    if estimator not in ("diagonal", "pairing_mean"):
        raise ValueError("unknown estimator")
    if reward_objective not in ("product", "additive"):
        raise ValueError("unknown reward objective")
    if args.mode != "rl" and estimator != "diagonal":
        raise ValueError("pairing_mean applies only to RL")
    if args.mode != "rl" and reward_objective != "product":
        raise ValueError("non-product reward applies only to RL")
    if estimator == "pairing_mean" and reward_objective != "product":
        raise ValueError("pairing_mean requires product reward")
    cap = 0.75 if args.mode == "rl" else 0.5
    if not 0 < args.hours <= cap:
        raise ValueError("fixed45minute RL/30minute SFT cumulative cap")
    manifest_path, cases_path = prepared / "MANIFEST.json", prepared / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != "paired-sufficiency-rl-input-proposal-v1"
        or manifest["seed"] != 2026092193
        or not manifest["ready_without_truncation"]
        or probe.campaign.sha(cases_path) != manifest["cases_sha256"]
    ):
        raise ValueError("immutable128TRAIN proposal required")
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    parents, blocks = manifest["selected_parents"], manifest["fixed_parent_blocks"]
    if (
        len(cases) != 256
        or len(set(parents)) != 128
        or sum(blocks, []) != parents
        or len(blocks) != 8
        or any(len(b) != 16 for b in blocks)
        or any(c["split"] != "train" for c in cases)
    ):
        raise ValueError("fixed8x16 TRAIN schedule required")
    paired_cases(cases, parents)
    identity = eval_sufficiency.adapter_identity(adapter, "joint")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config["r"] != 8 or config["lora_dropout"] != 0:
        raise ValueError("rank8 zero-dropout warmstart required")
    plan = {
        "schema": "paired-sufficiency-finite-training-v1",
        "mode": args.mode,
        "estimator": estimator,
        "reward_objective": reward_objective,
        "cases": str(cases_path),
        "cases_sha256": manifest["cases_sha256"],
        "input_manifest_sha256": probe.campaign.sha(manifest_path),
        "parent_blocks": blocks,
        "model": str(BASE),
        "warmstart": identity,
        "base_manifest_sha256": probe.campaign.sha(BASE / "local-research-manifest.json"),
        "official_metric_sha256": {
            str(path): probe.campaign.sha(path)
            for path in sorted((baseline.panel.OFFICIAL / "metrics").glob("*.py"))
        },
        "environment_lock_sha256": probe.campaign.sha(
            Path(sys.executable).parent.parent.parent / "uv.lock"
        ),
        "seed": SEED,
        "learning_rate": 2e-5,
        "fresh_optimizer": True,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "microbatch": 1,
        "candidate_pairs_per_parent": 4,
        "pair_denominator": 64,
        "max_sampled_blocks": 8,
        "max_calls": 1024 if args.mode == "rl" else 0,
        "max_consecutive_zero_blocks": 4,
        "budget_seconds": args.hours * 3600,
        "sampling_temperature": 0.8,
        "sampling_top_p": 1.0,
        "sampling_top_k": 0,
        "max_new_tokens": 128,
        "max_context": 8192,
        "objective": (
            "official paired EM product reward diagonal RLOO, both response token-logp sums /64"
            if estimator == "diagonal" and reward_objective == "product"
            else "official paired marginal-success additive reward diagonal RLOO, "
            "both response token-logp sums /64"
            if estimator == "diagonal"
            else "official paired EM product reward pairing-mean RLOO expectation, "
            "response coefficients /64"
        )
        if args.mode == "rl"
        else "paired gold JSON+EOS masked SFT, target-token mean; "
        "same RL updated/skipped blocks,4copies/variant, dose not FLOP matched",
        "source_sha256": {
            str(Path(m.__file__).resolve()): probe.campaign.sha(Path(m.__file__))
            for m in (rl, baseline, recipe, sft, eval_sufficiency)
        },
        "runner_sha256": probe.campaign.sha(Path(__file__)),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
    }
    control = []
    if args.mode == "sft_control":
        if args.rl_output is None:
            raise ValueError("fixed completed RL inventory required")
        rl_output = args.rl_output.resolve()
        owners = list(rl_output.glob("OWNER-*.json"))
        if not owners or any(
            not p.with_name(p.name.replace("OWNER-", "TERMINAL-")).exists() for p in owners
        ):
            raise ValueError("RL owner unresolved")
        import psutil

        for path in owners:
            owner = json.loads(path.read_text())
            try:
                process = psutil.Process(owner["pid"])
                if (
                    abs(process.create_time() - owner["create_time"]) < 0.01
                    and process.status() != psutil.STATUS_ZOMBIE
                ):
                    raise ValueError("RL owner still active")
            except psutil.NoSuchProcess:
                pass
        if any(json.loads(p.read_text()).get("failure") for p in rl_output.glob("TERMINAL-*.json")):
            raise ValueError("failed RL owner requires explicit review, not matched control")
        source_plan = json.loads((rl_output / "PLAN.json").read_text())
        if (
            source_plan["mode"] != "rl"
            or source_plan["warmstart"] != identity
            or source_plan["cases_sha256"] != plan["cases_sha256"]
        ):
            raise ValueError("control initializer/TRAIN inventory differs")
        control = boundary_inventory(rl_output)
        if not control:
            raise ValueError("no committed RL endpoint")
        plan.update(
            rl_output=str(rl_output),
            rl_plan_sha256=probe.campaign.sha(rl_output / "PLAN.json"),
            matched_rl_boundaries=control,
            planned_actual_updates=control[-1]["state"]["step"],
        )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if json.loads((output / "PLAN.json").read_text()) != plan:
            raise ValueError("immutable PLAN changed")
    else:
        probe.runtime.save(output / "PLAN.json", plan)
    return plan, cases, control


def run(args):
    global STOP
    STOP = False
    plan, cases, control = prepare(args)
    output = args.output.resolve()
    if args.prepare_only:
        print(json.dumps({"mode": args.mode, "max_sampled_blocks": 8, "GPU_loaded": False}))
        return
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    owners = {p.name.replace("OWNER-", "") for p in output.glob("OWNER-*.json")}
    terminals = {p.name.replace("TERMINAL-", "") for p in output.glob("TERMINAL-*.json")}
    if owners != terminals or (owners and not args.resume):
        raise ValueError("unresolved owner or explicit resume required")
    boundaries = boundary_inventory(output)
    if boundaries and not args.resume:
        raise ValueError("explicit boundary resume required")
    state = (
        boundaries[-1]["state"]
        if boundaries
        else {
            "step": 0,
            "sample_cursor": 0,
            "zero_streak": 0,
            "effective_groups": 0,
            "plan_sha256": probe.campaign.sha(output / "PLAN.json"),
        }
    )
    if state["plan_sha256"] != probe.campaign.sha(output / "PLAN.json"):
        raise ValueError("checkpoint PLAN identity differs")
    expected_batches = {f"sample-{i:04d}" for i in range(1, state["sample_cursor"] + 1)}
    if {p.name for p in (output / "batches").glob("sample-*")} - expected_batches:
        raise ValueError("uncommitted partial batch exists; no replay/regeneration")
    spent = sum(
        json.loads(p.read_text())["elapsed_seconds"] for p in output.glob("TERMINAL-*.json")
    )
    if spent >= plan["budget_seconds"]:
        raise ValueError("cumulative owner budget exhausted")
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + plan["budget_seconds"] - spent, lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient allocation margin")
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    model = optimizer = None
    failure, stop_reason = None, None
    checkpoint = Path(boundaries[-1]["checkpoint"]) if boundaries else args.adapter.resolve()
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
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        tokenizer = AutoTokenizer.from_pretrained(
            BASE, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base, checkpoint, is_trainable=True, autocast_adapter_dtype=True
        )
        model.enable_input_require_grads()
        parameters = rl.planner_parameters(model)
        optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.0)
        if boundaries:
            receipt = json.loads((checkpoint / "COMMIT.json").read_text())
            for name, sha in receipt["files"].items():
                if probe.campaign.sha(checkpoint / name) != sha:
                    raise ValueError("committed checkpoint changed")
            optimizer.load_state_dict(
                torch.load(checkpoint / "optimizer.pt", map_location="cuda:0", weights_only=True)
            )
            rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            checkpoint = save_boundary(model, optimizer, output, state)
        probe.runtime.save(
            output / f"LOAD-{invocation}.json",
            {
                "trainable_parameters": sum(p.numel() for p in parameters),
                "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
                "fresh_optimizer": not boundaries,
                "checkpoint": str(checkpoint),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        for cursor in range(state["sample_cursor"] + 1, 9):
            if finished(state):
                stop_reason = (
                    "four_zero_blocks" if state["zero_streak"] >= 4 else "eight_sampled_blocks"
                )
                break
            if args.mode == "sft_control" and cursor >= len(control):
                stop_reason = "matched_rl_endpoint"
                break
            if STOP or time.time() >= deadline - 180 or (output / "STOP").exists():
                stop_reason = "signal_or_time_cap"
                break
            directory = output / "batches" / f"sample-{cursor:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            parents = plan["parent_blocks"][cursor - 1]
            if args.mode == "rl":
                groups, rewards = collect(
                    model,
                    tokenizer,
                    directory,
                    deadline,
                    cases,
                    parents,
                    cursor,
                    {
                        "checkpoint": str(checkpoint),
                        "adapter_sha256": probe.campaign.sha(
                            checkpoint / "adapter_model.safetensors"
                        ),
                        "optimizer_step": state["step"],
                        "sample_cursor": state["sample_cursor"],
                    },
                    plan["reward_objective"],
                )
                training_rewards = [[pair["training_reward"] for pair in group] for group in groups]
                credits = response_advantages(groups, plan["estimator"], plan["reward_objective"])
                next_state = advance(state, credits)
                diagnostics = {
                    "rewards": rewards,
                    "parents": parents,
                    "valid_paired_candidates": sum(p["all_valid"] for g in groups for p in g),
                    "fully_valid_mixed_groups": sum(
                        len({p["reward"] for p in g if p["all_valid"]}) == 2 for g in groups
                    ),
                    "estimator": plan["estimator"],
                    "reward_objective": plan["reward_objective"],
                    "training_rewards": training_rewards,
                    "effective_groups": next_state["effective_groups"],
                    "response_advantages": credits,
                    "reward_success_count_histogram": dict(
                        Counter(sum(group) for group in rewards)
                    ),
                    "training_reward_sum_histogram": dict(
                        Counter(sum(group) for group in training_rewards)
                    ),
                    "absolute_advantage_mass": sum(
                        abs(value) for group in rewards for value in rl.rloo(group)
                    ),
                    "absolute_response_advantage_mass": sum(
                        abs(value) for group in credits for pair in group for value in pair
                    ),
                    "distinct_valid_pair_predictions": [
                        len(
                            {
                                json.dumps(p["predictions"], sort_keys=True)
                                for p in group
                                if p["all_valid"]
                            }
                        )
                        for group in groups
                    ],
                    "updated": next_state["step"] > state["step"],
                }
                probe.runtime.save(directory / "ROLLOUT.json", diagnostics)
                statistics = optimize_rl(
                    model, optimizer, groups, deadline, plan["estimator"], plan["reward_objective"]
                )
            else:
                source_state = control[cursor]["state"]
                next_state = {**source_state, "plan_sha256": state["plan_sha256"]}
                updated = source_state["step"] > state["step"]
                rows = sft.tokenize_rows(control_examples(cases, parents), tokenizer)
                statistics = (
                    optimize_sft(model, optimizer, rows, deadline)
                    if updated
                    else {"optimizer_called": False}
                )
                diagnostics = {
                    "parents": parents,
                    "updated": updated,
                    "source_rl_boundary": control[cursor],
                }
            probe.runtime.save(directory / "UPDATE.json", {**diagnostics, **statistics})
            checkpoint = save_boundary(model, optimizer, output, next_state)
            state = next_state
            probe.campaign.snapshot(
                output / "STATUS.json",
                {"state": "boundary_committed", **state, "updated": time.time()},
            )
        stop_reason = stop_reason or (
            "four_zero_blocks" if state["zero_streak"] >= 4 else "sample_cap_or_matched_endpoint"
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        calls = [
            json.loads(p.read_text()) for p in (output / "batches").glob("sample-*/calls/*.json")
        ]
        pairs = [
            json.loads(p.read_text()) for p in (output / "batches").glob("sample-*/pairs/*.json")
        ]
        probe.campaign.snapshot(
            output / "SUMMARY.json",
            {
                "mode": args.mode,
                "estimator": plan["estimator"],
                "reward_objective": plan["reward_objective"],
                "max_planned_calls": plan["max_calls"],
                "committed_sampled_blocks": state["sample_cursor"],
                "actual_optimizer_steps": state["step"],
                "recorded_pairs": len(pairs),
                "valid_pairs": sum(p["all_valid"] for p in pairs),
                "paired_reward_sum": sum(p["reward"] for p in pairs),
                "physical_cost": baseline.native.evaluation.cost(calls),
                "native_service_seconds": sum(c["ended"] - c["started"] for c in calls),
                "stop_reason": stop_reason,
                "failure": failure,
                "endpoint": str(checkpoint),
                "matched_control_complete": state["sample_cursor"] == len(control) - 1
                if args.mode == "sft_control"
                else None,
            },
        )
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                **state,
                "failure": failure,
                "stopped": STOP,
                "stop_reason": stop_reason,
                "endpoint": str(checkpoint),
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
                "endpoint_selection": "last committed boundary, never held score",
            },
        )
        probe.campaign.snapshot(
            output / "STATUS.json",
            {
                "state": "failed" if failure else "terminal",
                **state,
                "failure": failure,
                "stop_reason": stop_reason,
                "updated": time.time(),
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rl", "sft_control"), required=True)
    parser.add_argument("--estimator", choices=("diagonal", "pairing_mean"), default="diagonal")
    parser.add_argument("--reward-objective", choices=("product", "additive"), default="product")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rl-output", type=Path)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
