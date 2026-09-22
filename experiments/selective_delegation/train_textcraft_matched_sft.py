"""Prospective whole-row token/update-matched extra SFT; no launch authority."""

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import random
import signal
import time
import uuid
from pathlib import Path

import psutil
import textcraft_trajectory_loss as credit_math
import train_planner as checkpoint
import train_textcraft_sft as recipe

p = recipe.probe
ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
SEED = 2026092243
ROWS_SHA = "dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a"
READINESS_SHA = "e96a7ac7e9ded7695217378cd1668607c68cc5c0bbe244d40a88842b4c691444"
WARM_SHA = "f029d36967eb1cef30e102fe82e52f22f90d5eacdaf91e82937b60218ba5b00f"
STOP = False


def read(path):
    return json.loads(path.read_text())


def credited_budget(episodes, calls, update):
    credits = credit_math.batch_credits(episodes, calls)
    active = [item for item in credits if item.advantage != 0]
    tokens = sum(len(calls[item.call_id]["output_token_ids"]) for item in active)
    if (
        update.get("optimizer_called") is not bool(active)
        or update.get("nonzero_action_calls") != len(active)
        or (active and update.get("train_eval_replay_token_count") != tokens)
    ):
        raise ValueError("committed UPDATE differs from native nonzero credit")
    return tokens


def endpoint_steps(terminal, summary):
    if (
        terminal.get("complete") is not True
        or terminal.get("endpoint_usable") is not True
        or terminal.get("failure")
        or terminal.get("stopped")
        or summary.get("complete") is not True
        or summary.get("failure")
    ):
        raise ValueError("failed/capped/unusable RL endpoint is not a partial-dose control")
    counts = [
        row[key]
        for row in (terminal, summary)
        for key in ("actual_optimizer_steps", "committed_optimizer_steps")
    ]
    if len(set(counts)) != 1 or type(counts[0]) is not int or not 0 <= counts[0] <= 2:
        raise ValueError("inconsistent actual committed update dose")
    return counts[0]


def row_schedule(examples, order, budgets):
    if sorted(order) != list(range(len(examples))) or not examples:
        raise ValueError("exact fixed permutation required")
    cursor, result = 0, []
    for budget in budgets:
        if type(budget) is not int or budget <= 0:
            raise ValueError("positive credited-token update budget required")
        selected, tokens = [], 0
        while tokens < budget:
            index = order[cursor % len(order)]
            size = len(examples[index]["target_ids"])
            if not size:
                raise ValueError("empty teacher target")
            selected.append(index)
            tokens += size
            cursor += 1
        result.append(
            dict(
                row_indices=selected,
                row_ids=[examples[i]["id"] for i in selected],
                credited_rl_tokens=budget,
                actual_target_tokens=tokens,
                overshoot_tokens=tokens - budget,
                cyclic_cursor=cursor,
            )
        )
    return result


def teacher_loss(model, row, denominator):
    import torch

    targets = torch.tensor([row["target_ids"]], device=model.device)
    logits = model(
        input_ids=torch.tensor([row["input_ids"]], device=model.device),
        logits_to_keep=len(row["target_ids"]),
        use_cache=False,
    ).logits
    return checkpoint.target_loss(logits, targets) / denominator


def teacher_inputs():
    from transformers import AutoTokenizer

    source = ROOT / "textcraft-public-discovery-prototype-001/rows.jsonl"
    native_plan = ROOT / "textcraft-train-readiness-001/PLAN.json"
    if p.campaign.sha(source) != ROWS_SHA or p.campaign.sha(native_plan) != READINESS_SHA:
        raise ValueError("fixed public055/063 inventory changed")
    ids = {job["task_id"] for job in read(native_plan)["jobs"]}
    rows = sorted(
        [row for row in map(json.loads, source.read_text().splitlines()) if row["task_id"] in ids],
        key=lambda row: (row["task_id"], row["step"]),
    )
    if len(ids) != 8 or len(rows) != 86 or any(".train." not in row["task_id"] for row in rows):
        raise ValueError("exact eight TRAIN tasks /86 public rows required")
    for row in rows:
        recipe.strict_action(row["target"])
    tokenizer = AutoTokenizer.from_pretrained(
        p.campaign.MODELS["4b"], local_files_only=True, trust_remote_code=False
    )
    examples = recipe.tokenize_rows(rows, tokenizer)
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    frozen = dict(
        schema="textcraft-extra-sft-fixed-order-v1",
        seed=SEED,
        rows_sha256=ROWS_SHA,
        readiness_plan_sha256=READINESS_SHA,
        task_ids=sorted(ids),
        row_ids=[e["id"] for e in examples],
        cycle_ids=[examples[i]["id"] for i in order],
        order=order,
        target_lengths=[len(e["target_ids"]) for e in examples],
        target_tokens=sum(len(e["target_ids"]) for e in examples),
        prompt_tokens=sum(len(e["input_ids"]) - len(e["target_ids"]) + 1 for e in examples),
        max_target=max(len(e["target_ids"]) for e in examples),
        model_manifest_sha256=p.campaign.sha(
            Path(p.campaign.MODELS["4b"]) / "local-research-manifest.json"
        ),
    )
    return frozen, examples


def rl_dose(output):
    plan = read(output / "PLAN.json")
    expected_tasks = {
        job["task_id"] for job in read(ROOT / "textcraft-train-readiness-001/PLAN.json")["jobs"]
    }
    if (
        plan.get("schema") != "textcraft-terminal-rloo-two-update-v1"
        or plan["warm"]["sha256"] != WARM_SHA
        or plan["first_batch"]["plan_sha256"] != READINESS_SHA
        or plan["learning_rate"] != 2e-5
        or plan["temperature"] != 0.5
    ):
        raise ValueError("exact conditional065 public056 RL contract required")
    skipped = output / "CONDITIONAL-SKIP.json"
    if skipped.exists():
        skip = read(skipped)
        if skip.get("actual_optimizer_steps") != 0 or skip.get("GPU_loaded") is not False:
            raise ValueError("invalid zero-training skip")
        return plan, [], {str(skipped): p.campaign.sha(skipped)}
    owners = list(output.glob("OWNER-*.json"))
    if len(owners) != 1:
        raise ValueError("one released RL owner required")
    owner = read(owners[0])
    try:
        process = psutil.Process(owner["pid"])
        if (
            abs(process.create_time() - owner["create_time"]) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            raise ValueError("RL owner still active")
    except psutil.NoSuchProcess:
        pass
    terminal_path = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
    terminal, summary = read(terminal_path), read(output / "SUMMARY.json")
    steps = endpoint_steps(terminal, summary)
    pins = {
        str(path): p.campaign.sha(path)
        for path in (output / "PLAN.json", owners[0], terminal_path, output / "SUMMARY.json")
    }
    dose, previous = [], 0
    cursor = terminal["committed_sampled_batches"]
    if cursor != summary["committed_sampled_batches"] or not 0 <= cursor <= 4:
        raise ValueError("committed sample count differs")
    endpoint = None
    for sample in range(1, cursor + 1):
        boundary_path = output / "boundaries" / f"sample-{sample:04d}" / "BOUNDARY.json"
        boundary = read(boundary_path)
        endpoint = Path(boundary["checkpoint"])
        if endpoint.resolve().parent != boundary_path.parent.resolve():
            raise ValueError("checkpoint outside committed sampled boundary")
        commit_path, state_path = endpoint / "COMMIT.json", endpoint / "STATE.json"
        commit, state = read(commit_path), read(state_path)
        if (
            p.campaign.sha(commit_path) != boundary["commit_sha256"]
            or p.campaign.sha(state_path) != commit["files"]["STATE.json"]
            or state != boundary["state"]
            or state["sample_cursor"] != sample
            or state["plan_sha256"] != p.campaign.sha(output / "PLAN.json")
        ):
            raise ValueError("committed sampled state identity differs")
        batch_path = output / "batches" / f"sample-{sample:04d}" / "BATCH.json"
        if p.campaign.sha(batch_path) != state["batch_sha256"]:
            raise ValueError("BATCH changed after committed update")
        batch = read(batch_path)
        if {episode["task_id"] for episode in batch["episodes"]} != expected_tasks:
            raise ValueError("RL dose does not use the same eight frozen TRAIN tasks")
        calls = {}
        for filename, digest in batch["native_receipt_sha256"].items():
            path = Path(filename)
            if path.parent.name == "calls":
                if p.campaign.sha(path) != digest:
                    raise ValueError("native call changed")
                calls[path.stem] = read(path)
        tokens = credited_budget(batch["episodes"], calls, state["update"])
        updated = tokens > 0
        if state["step"] != previous + int(updated):
            raise ValueError("committed update counter not actual optimizer calls")
        if updated:
            marker = read(batch_path.parent / "OPTIMIZER-STEP-STARTED.json")
            replay = read(batch_path.parent / "TRAIN-EVAL-REPLAY.json")
            if (
                marker["previous_step"] != previous
                or replay["passed"] is not True
                or replay["token_count"] != tokens
            ):
                raise ValueError("actual optimizer/replay token receipt differs")
            dose.append(tokens)
        previous = state["step"]
        for path in (boundary_path, commit_path, state_path, batch_path):
            pins[str(path)] = p.campaign.sha(path)
    if previous != steps or len(dose) != steps or str(endpoint) != terminal["endpoint"]:
        raise ValueError("terminal endpoint/dose differs from committed boundaries")
    return plan, dose, pins


def run(args):
    global STOP
    frozen, examples = teacher_inputs()
    if args.freeze_only:
        p.runtime.save(args.frozen, {**frozen, "frozen_utc_epoch": time.time()})
        return
    actual = read(args.frozen)
    if {key: actual[key] for key in frozen} != frozen:
        raise ValueError("prospective row order/token inventory changed")
    if not args.rl_output or not args.output or not 0 < args.hours <= 0.5:
        raise ValueError("RL output, new control output and at most30minutes required")
    rl_plan, budgets, pins = rl_dose(args.rl_output.resolve())
    owners = list(args.rl_output.glob("OWNER-*.json"))
    if owners and actual["frozen_utc_epoch"] >= min(read(x)["started"] for x in owners):
        raise ValueError("teacher ordering must be frozen before RL training")
    schedule = row_schedule(examples, frozen["order"], budgets)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(
        schema="textcraft-token-update-matched-extra-sft-v1",
        warm=rl_plan["warm"],
        rl_output=str(args.rl_output.resolve()),
        rl_receipt_sha256=pins,
        frozen_sha256=p.campaign.sha(args.frozen),
        seed=SEED,
        schedule=schedule,
        planned_updates=len(schedule),
        learning_rate=2e-5,
        weight_decay=0,
        gradient_clip=1,
        lora=dict(rank=8, alpha=16, dropout=0),
        microbatch=1,
        environment={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "safetensors")
        },
        temperature=1,
        loss="sum gold target CE / actual selected target tokens",
        budget_seconds=args.hours * 3600,
        source_sha256={
            str(path): p.campaign.sha(path)
            for path in (
                Path(__file__),
                Path(recipe.__file__),
                Path(checkpoint.__file__),
                Path(credit_math.__file__),
            )
        },
        caveat="Update/target-token matched with whole-row overshoot, not matched "
        "states, information, prompt compute or FLOPs; SFT T1 vs RL T0.5.",
    )
    if (output / "PLAN.json").exists():
        if read(output / "PLAN.json") != plan:
            raise ValueError("prepared immutable control PLAN differs")
    else:
        p.runtime.save(output / "PLAN.json", plan)
    if not schedule:
        p.runtime.save(
            output / "SKIPPED.json", dict(reason="zero actual RL steps", GPU_loaded=False)
        )
        return
    if args.prepare_only:
        return
    if list(output.glob("OWNER-*.json")):
        raise ValueError("no implicit retry/resume or partial-dose continuation")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("main must explicitly assign one GPU")
    started = time.time()
    deadline = min(started + plan["budget_seconds"], int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    lock = (p.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = uuid.uuid4().hex[:12]
    p.runtime.save(
        output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
        ),
    )
    model = optimizer = endpoint = None
    failure, complete, step = None, False, 0

    def guard():
        if STOP or time.time() >= deadline - 60:
            raise TimeoutError("control incomplete; no partial endpoint substitution")

    def stop(*_):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        guard()
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.set_num_threads(4)
        warm = Path(plan["warm"]["path"])
        if p.campaign.sha(warm / "adapter_model.safetensors") != WARM_SHA:
            raise ValueError("public056 warm adapter changed")
        base = AutoModelForCausalLM.from_pretrained(
            p.campaign.MODELS["4b"],
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(
            base, warm, is_trainable=True, autocast_adapter_dtype=True
        )
        config = model.peft_config["default"]
        if (config.r, config.lora_alpha, config.lora_dropout) != (8, 16, 0):
            raise ValueError("fixed LoRA configuration changed")
        params = [value for name, value in model.named_parameters() if value.requires_grad]
        if (
            not params
            or any(value.dtype != torch.float32 for value in params)
            or any(
                "lora_" not in name
                for name, value in model.named_parameters()
                if value.requires_grad
            )
        ):
            raise ValueError("FP32 LoRA-only optimizer required")
        optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0)
        model.train()
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        endpoint = checkpoint.save_checkpoint(model, optimizer, output, dict(step=0))
        for update in schedule:
            guard()
            optimizer.zero_grad(set_to_none=True)
            nll, prompt_tokens = 0.0, 0
            before = [value.detach().cpu().clone() for value in params]
            for index in update["row_indices"]:
                guard()
                row = examples[index]
                loss = teacher_loss(model, row, update["actual_target_tokens"])
                loss.backward()
                nll += float(loss.detach()) * update["actual_target_tokens"]
                prompt_tokens += len(row["input_ids"]) - len(row["target_ids"]) + 1
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
            guard()
            optimizer.step()
            step += 1
            delta = (
                sum(
                    float((value.detach().cpu() - old).double().square().sum())
                    for value, old in zip(params, before, strict=True)
                )
                ** 0.5
            )
            state = dict(
                step=step,
                planned_updates=len(schedule),
                plan_sha256=p.campaign.sha(output / "PLAN.json"),
            )
            endpoint = checkpoint.save_checkpoint(model, optimizer, output, state)
            p.runtime.save(
                output / "updates" / f"{step:04d}.json",
                dict(
                    **update,
                    step=step,
                    target_nll=nll / update["actual_target_tokens"],
                    prompt_tokens=prompt_tokens,
                    gradient_norm=norm,
                    adapter_l2_delta=delta,
                    checkpoint=str(endpoint),
                ),
            )
        complete = step == len(schedule)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        p.runtime.save(
            output / f"TERMINAL-{owner}.json",
            dict(
                complete=complete,
                failure=failure,
                actual_optimizer_steps=step,
                endpoint=str(endpoint) if endpoint else None,
                endpoint_usable=complete and not failure,
                ended=time.time(),
                elapsed_seconds=time.time() - started,
            ),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--rl-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
