"""One-epoch public indexed-action SFT; execution requires a later accepted launcher."""

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

SEED, EXAMPLES, CONTEXT, TARGET_CAP = 2026092200, 524, 8192, 128
ACTIVE_FAILURE = None


def batch_sizes(count: int) -> list[int]:
    return [min(16, count - start) for start in range(0, count, 16)]


def committed_checkpoint(output: Path, resume: bool) -> tuple[Path | None, dict | None]:
    """Return the newest checksum-verified checkpoint, never a merely present directory."""
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
    state = json.loads((restored / "STATE.json").read_text())
    return restored, state


def completed_epoch(state: dict | None) -> bool:
    return state is not None and state.get("epoch") >= 1 and state.get("cursor") == 0


def parameter_l1_delta(parameters, initial_parameters) -> float:
    return sum(
        (parameter.detach() - initial).abs().sum().item()
        for parameter, initial in zip(parameters, initial_parameters, strict=True)
    )


def validate_rows(rows: list[dict], manifest: dict) -> int:
    if len(rows) != EXAMPLES or manifest.get("examples") != EXAMPLES:
        raise ValueError("exactly524 prepared successful-game examples required")
    if manifest.get("successful_games_only") is not True:
        raise ValueError("explicit success-filtered TRAIN provenance required")
    if len({row["id"] for row in rows}) != EXAMPLES:
        raise ValueError("unique action examples required")
    for row in rows:
        target = json.loads(row["target_json"])
        if set(target) != {"action_index"} or type(target["action_index"]) is not int:
            raise ValueError("target must be strict integer action-index JSON")
        if not isinstance(row["prompt"], str) or "public_context" not in row["prompt"]:
            raise ValueError("exact public flat prompt required")
    return EXAMPLES


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
        target = tokenizer.encode(row["target_json"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if len(prefix) + len(target) > CONTEXT or len(target) > TARGET_CAP:
            raise ValueError("target/context exceeds fixed cap; never truncate")
        examples.append({"id": row["id"], "input_ids": prefix + target[:-1], "target_ids": target})
    return examples


def run(args) -> None:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = None
    if args.epochs != 1 or args.learning_rate != 1e-4 or not 0 < args.hours <= 0.5:
        raise ValueError("fixed one epoch, LR1e-4, and at most30 minutes")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prepared, output = args.prepared.resolve(), args.output.resolve()
    rows_path, manifest_path = prepared / "examples.jsonl", prepared / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if probe.campaign.sha(rows_path) != manifest["examples_sha256"]:
        raise ValueError("prepared action data checksum differs")
    rows = [json.loads(line) for line in rows_path.open()]
    validate_rows(rows, manifest)
    tokenizer = AutoTokenizer.from_pretrained(probe.campaign.MODELS["4b"], local_files_only=True)
    examples = tokenize_rows(rows, tokenizer)
    if batch_sizes(len(examples)) != [16] * 32 + [12]:
        raise ValueError("expected 33 effective16 updates with final12")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "alfworld-public-action-sft-v1",
        "prepared": str(prepared),
        "prepared_manifest_sha256": probe.campaign.sha(manifest_path),
        "examples_sha256": probe.campaign.sha(rows_path),
        "examples": EXAMPLES,
        "epochs": 1,
        "seed": SEED,
        "learning_rate": 1e-4,
        "effective_batch": 16,
        "microbatch": 1,
        "planned_updates": 33,
        "last_update_examples": 12,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0},
        "dtype": {"base": "bfloat16", "lora": "float32"},
        "max_context": CONTEXT,
        "target_only_json_eos": True,
        "budget_seconds": args.hours * 3600,
        "checkpoints": [0, 8, 16, 24, 32, 33],
        "source_sha256": probe.campaign.sha(Path(__file__)),
        "base_manifest_sha256": probe.campaign.sha(
            Path(probe.campaign.MODELS["4b"]) / "local-research-manifest.json"
        ),
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
    # A completed one-epoch checkpoint is terminal.  In particular, never repeat its epoch
    # just because its cursor is represented as zero at the next-epoch boundary.
    if completed_epoch(restored_state):
        return
    owners = {p.stem.removeprefix("OWNER-") for p in output.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in output.glob("TERMINAL-*.json")}
    if owners != terminals:
        raise ValueError("unresolved prior owner")
    spent = sum(
        json.loads(path.read_text())["elapsed_seconds"] for path in output.glob("TERMINAL-*.json")
    )
    if spent >= args.hours * 3600:
        raise ValueError("cumulative cap exhausted")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("no usable GPU; source041 preparation does not launch CPU training")
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    started_owner = time.time()
    deadline = min(started_owner + args.hours * 3600 - spent, lease - 600)
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
            "started": started_owner,
            "deadline": deadline,
            "source": str(Path(__file__).resolve()),
            "source_sha256": plan["source_sha256"],
        },
    )
    stopping = False
    state = {"step": 0, "epoch": 0, "cursor": 0, "training_seconds": 0.0}
    terminal_written = False

    def finish():
        nonlocal terminal_written
        if terminal_written:
            return
        terminal_written = True
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                **state,
                "failure": ACTIVE_FAILURE,
                "elapsed_seconds": time.time() - started_owner,
                "ended": time.time(),
                "endpoint": "complete" if state["epoch"] >= 1 else "capped_or_failed",
                "complete": state["epoch"] >= 1,
                "stopping": stopping,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    atexit.register(finish)

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
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
    model = (
        PeftModel.from_pretrained(base, restored, is_trainable=True, autocast_adapter_dtype=True)
        if restored
        else get_peft_model(
            base,
            LoraConfig(
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
            ),
            autocast_adapter_dtype=True,
        )
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("fresh FP32 LoRA parameters required")
    if any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
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
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    state = restored_state or {"step": 0, "epoch": 0, "cursor": 0, "training_seconds": 0.0}
    if not restored:
        save_checkpoint(model, optimizer, output, state)
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    for cursor in range(state["cursor"], EXAMPLES, 16):
        if stopping or time.time() >= deadline - 90:
            break
        batch = [examples[index] for index in epoch_order(EXAMPLES, SEED, 0)[cursor : cursor + 16]]
        denom = sum(len(row["target_ids"]) for row in batch)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        nll = 0.0
        for row in batch:
            ids = torch.tensor([row["input_ids"]], device="cuda:0")
            targets = torch.tensor([row["target_ids"]], device="cuda:0")
            loss = target_loss(
                model(input_ids=ids, logits_to_keep=targets.shape[1]).logits, targets
            )
            (loss / denom).backward()
            nll += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        state = {
            "step": state["step"] + 1,
            "epoch": int(cursor + len(batch) == EXAMPLES),
            "cursor": 0 if cursor + len(batch) == EXAMPLES else cursor + len(batch),
            "training_seconds": state["training_seconds"] + elapsed,
        }
        probe.runtime.save(
            output / "steps" / f"{state['step']:04d}.json",
            {
                **state,
                "examples": len(batch),
                "target_tokens": denom,
                "gradient_norm": float(norm),
                "target_nll": nll / denom,
                "seconds": elapsed,
            },
        )
        if state["step"] == 1:
            delta = parameter_l1_delta(parameters, initial_parameters)
            if not delta > 0:
                raise ValueError("first optimizer step had zero adapter parameter delta")
            probe.runtime.save(
                output / "FIRST-UPDATE.json",
                {"target_nll": nll / denom, "parameter_l1_delta": delta},
            )
        if state["step"] in plan["checkpoints"]:
            save_checkpoint(model, optimizer, output, state)
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
