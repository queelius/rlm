"""One fixed epoch of TextCraft public flat-action SFT; requires a later accepted launch."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import importlib.metadata
import json
import os
import random
import signal
import sys
import time
import uuid
from pathlib import Path

import probe
from train_planner import epoch_order, save_checkpoint, target_loss

SEED, ROWS, TASKS, CONTEXT, TARGET_CAP = 2026092208, 366, 32, 8192, 256
STOP = False
ACTIVE_FAILURE = None


def batch_sizes(count: int, start: int = 0) -> list[int]:
    return [min(16, count - cursor) for cursor in range(start, count, 16)]


def committed_checkpoint(output: Path, resume: bool) -> tuple[Path | None, dict | None]:
    snapshots = sorted(
        path for path in output.glob("checkpoint-*") if (path / "COMMIT.json").exists()
    )
    restored = snapshots[-1] if resume and snapshots else None
    if restored is None:
        return None, None
    receipt = json.loads((restored / "COMMIT.json").read_text())
    for name, digest in receipt["files"].items():
        if probe.campaign.sha(restored / name) != digest:
            raise ValueError("checkpoint checksum mismatch")
    return restored, json.loads((restored / "STATE.json").read_text())


def completed_epoch(state: dict | None) -> bool:
    return state is not None and state.get("epoch") == 1 and state.get("cursor") == 0


def parameter_l1_delta(parameters, initial_parameters) -> float:
    return sum(
        (parameter.detach() - initial).abs().sum().item()
        for parameter, initial in zip(parameters, initial_parameters, strict=True)
    )


def strict_action(target: str) -> None:
    value = json.loads(target)
    if not isinstance(value, dict) or not isinstance(value.get("action"), str):
        raise ValueError("strict public action JSON required")
    action = value["action"]
    if action == "get_info":
        valid = set(value) == {"action", "items"} and isinstance(value["items"], list)
        valid = valid and all(isinstance(item, str) for item in value["items"])
    elif action == "craft":
        valid = set(value) == {"action", "ingredients", "target_item", "output_count"}
        valid = valid and isinstance(value["ingredients"], dict)
        valid = valid and all(
            isinstance(key, str) and type(count) is int
            for key, count in value["ingredients"].items()
        )
        valid = (
            valid and isinstance(value["target_item"], str) and type(value["output_count"]) is int
        )
    elif action == "finish":
        valid = set(value) == {"action", "message"} and isinstance(value["message"], str)
    else:
        valid = False
    if not valid:
        raise ValueError("strict public action JSON required")


def validate_rows(rows: list[dict], manifest: dict) -> int:
    if len(rows) != ROWS or manifest.get("rows") != ROWS:
        raise ValueError("exactly366 public action rows required")
    if manifest.get("eligible_task_count") != TASKS:
        raise ValueError("exactly32 native-successful TRAIN tasks required")
    identities = {(row.get("task_id"), row.get("step")) for row in rows}
    if len(identities) != ROWS or len({task for task, _ in identities}) != TASKS:
        raise ValueError("unique rows from exactly32 tasks required")
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not row["prompt"].startswith(
            "You control a crafting inventory."
        ):
            raise ValueError("exact public crafting prompt required")
        strict_action(row.get("target", ""))
    return TASKS


def tokenize_rows(rows: list[dict], tokenizer) -> list[dict]:
    examples = []
    for row in rows:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        target = tokenizer.encode(row["target"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if (
            tokenizer.eos_token_id is None
            or len(prefix) + len(target) > CONTEXT
            or len(target) > TARGET_CAP
        ):
            raise ValueError("context/target cap exceeded; never truncate")
        examples.append(
            {
                "id": f"{row['task_id']}:{row['step']}",
                "input_ids": prefix + target[:-1],
                "target_ids": target,
            }
        )
    return examples


def check(deadline: float, reserve: float = 0) -> None:
    if STOP or time.time() >= deadline - reserve:
        raise TimeoutError("bounded owner stopped/deadline; incomplete work is unknown")


def run(args) -> None:
    global ACTIVE_FAILURE, STOP
    ACTIVE_FAILURE, STOP = None, False
    if args.epochs != 1 or args.learning_rate != 1e-4 or not 0 < args.hours <= 0.5:
        raise ValueError("fixed one epoch, LR1e-4, and at most30 minutes")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prepared, output = args.prepared.resolve(), args.output.resolve()
    rows_path, manifest_path = prepared / "rows.jsonl", prepared / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if probe.campaign.sha(rows_path) != manifest.get("rows_sha256"):
        raise ValueError("prepared public rows checksum differs")
    rows = [json.loads(line) for line in rows_path.open()]
    validate_rows(rows, manifest)
    tokenizer = AutoTokenizer.from_pretrained(probe.campaign.MODELS["4b"], local_files_only=True)
    examples = tokenize_rows(rows, tokenizer)
    if batch_sizes(len(examples)) != [16] * 22 + [14]:
        raise ValueError("expected23 updates with final14 rows")
    base_manifest = Path(probe.campaign.MODELS["4b"]) / "local-research-manifest.json"
    if manifest.get("model") != probe.campaign.MODELS["4b"] or manifest.get(
        "model_manifest_sha256"
    ) != probe.campaign.sha(base_manifest):
        raise ValueError("prepared model identity differs")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "textcraft-public-flat-action-sft-v1",
        "prepared": str(prepared),
        "prepared_manifest_sha256": probe.campaign.sha(manifest_path),
        "rows_sha256": probe.campaign.sha(rows_path),
        "model": probe.campaign.MODELS["4b"],
        "tasks": TASKS,
        "rows": ROWS,
        "epochs": 1,
        "seed": SEED,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "effective_batch": 16,
        "microbatch": 1,
        "planned_updates": 23,
        "last_update_rows": 14,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0},
        "dtype": {"base": "bfloat16", "lora": "float32"},
        "max_context": CONTEXT,
        "max_target": TARGET_CAP,
        "target_only_json_eos": True,
        "checkpoint_steps": list(range(24)),
        "fixed_endpoint": "checkpoint-0023; not selected on validation",
        "budget_seconds": args.hours * 3600,
        "source_sha256": probe.campaign.sha(Path(__file__)),
        "training_recipe_sha256": probe.campaign.sha(Path(target_loss.__code__.co_filename)),
        "base_manifest_sha256": probe.campaign.sha(base_manifest),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{
                name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
            },
        },
    }
    plan_path = output / "PLAN.json"
    if plan_path.exists():
        if not args.resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("resume requires exact unchanged plan")
    else:
        probe.runtime.save(plan_path, plan)
    if args.prepare_only:
        return
    restored, restored_state = committed_checkpoint(output, args.resume)
    if completed_epoch(restored_state):
        return
    owners = {path.stem.removeprefix("OWNER-") for path in output.glob("OWNER-*.json")}
    terminals = {path.stem.removeprefix("TERMINAL-") for path in output.glob("TERMINAL-*.json")}
    if owners != terminals:
        raise ValueError("unresolved prior owner")
    spent = sum(
        json.loads(path.read_text())["elapsed_seconds"] for path in output.glob("TERMINAL-*.json")
    )
    if spent >= args.hours * 3600:
        raise ValueError("cumulative cap exhausted")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("no usable GPU; this CPU preparation does not launch training")
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    started = time.time()
    deadline = min(started + args.hours * 3600 - spent, lease - 600)
    if deadline < time.time() + 180:
        raise RuntimeError("insufficient allocation time")
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation = uuid.uuid4().hex[:12]
    import psutil

    probe.runtime.save(
        output / f"OWNER-{invocation}.json",
        {
            "pid": os.getpid(),
            "create_time": psutil.Process().create_time(),
            "started": started,
            "deadline": deadline,
            "source": str(Path(__file__).resolve()),
            "source_sha256": plan["source_sha256"],
        },
    )
    state = restored_state or {"step": 0, "epoch": 0, "cursor": 0, "training_seconds": 0.0}
    terminal_written = False

    def finish() -> None:
        nonlocal terminal_written
        if terminal_written:
            return
        terminal_written = True
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                **state,
                "failure": ACTIVE_FAILURE,
                "elapsed_seconds": time.time() - started,
                "ended": time.time(),
                "endpoint": "complete" if state["epoch"] else "capped_or_failed",
                "complete": bool(state["epoch"]),
                "stopped": STOP,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    atexit.register(finish)
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base = AutoModelForCausalLM.from_pretrained(
        probe.campaign.MODELS["4b"],
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = (
        PeftModel.from_pretrained(base, restored, is_trainable=True, autocast_adapter_dtype=True)
        if restored
        else get_peft_model(base, config, autocast_adapter_dtype=True)
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("fresh FP32 LoRA parameters required")
    if any(
        "lora_" not in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ):
        raise ValueError("base weights must remain frozen")
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
    if restored:
        optimizer.load_state_dict(
            torch.load(restored / "optimizer.pt", map_location="cuda:0", weights_only=True)
        )
        rng = torch.load(restored / "rng.pt", map_location="cpu", weights_only=True)
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
    else:
        save_checkpoint(model, optimizer, output, state)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    initial = [parameter.detach().clone() for parameter in parameters]
    for cursor in range(state["cursor"], ROWS, 16):
        check(deadline, 90)
        batch = [examples[index] for index in epoch_order(ROWS, SEED, 0)[cursor : cursor + 16]]
        denominator = sum(len(row["target_ids"]) for row in batch)
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        nll = 0.0
        for row in batch:
            ids = torch.tensor([row["input_ids"]], device="cuda:0")
            targets = torch.tensor([row["target_ids"]], device="cuda:0")
            loss = target_loss(
                model(input_ids=ids, logits_to_keep=targets.shape[1]).logits, targets
            )
            (loss / denominator).backward()
            nll += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - step_started
        state = {
            "step": state["step"] + 1,
            "epoch": int(cursor + len(batch) == ROWS),
            "cursor": 0 if cursor + len(batch) == ROWS else cursor + len(batch),
            "training_seconds": state["training_seconds"] + elapsed,
        }
        probe.runtime.save(
            output / "steps" / f"{state['step']:04d}.json",
            {
                **state,
                "rows": len(batch),
                "target_tokens": denominator,
                "target_nll": nll / denominator,
                "gradient_norm": float(norm),
                "seconds": elapsed,
            },
        )
        if state["step"] == 1:
            delta = parameter_l1_delta(parameters, initial)
            if not delta > 0:
                raise ValueError("first optimizer step had zero adapter parameter delta")
            probe.runtime.save(
                output / "FIRST-UPDATE.json",
                {"target_nll": nll / denominator, "parameter_l1_delta": delta},
            )
        save_checkpoint(model, optimizer, output, state)
    finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    try:
        run(parser.parse_args())
    except Exception as exc:
        ACTIVE_FAILURE = f"{type(exc).__name__}: {exc}"
        raise
