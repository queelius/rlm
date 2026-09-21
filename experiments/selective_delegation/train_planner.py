"""Bounded question-only planner SFT; helpers remain frozen in later evaluation."""

from __future__ import annotations

import argparse
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

STOP = False


def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def target_loss(logits, targets):
    import torch.nn.functional as functional

    return functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
    )


def tokenize_rows(rows, tokenizer):
    result = []
    for row in rows:
        if row["split"] != "train":
            raise ValueError("only training parents may enter optimizer")
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
        if len(prefix) > 512:
            raise ValueError("training prompt exceeds 512 tokens")
        if len(target) > 256:
            raise ValueError("training target exceeds 256 tokens")
        if len(prefix) + len(target) > 2048:
            raise ValueError("training context exceeds declared cap; never truncate")
        # Last T input positions predict the T target tokens, starting at prefix[-1].
        result.append(
            {
                "id": row["id"],
                "input_ids": prefix + target[:-1],
                "target_ids": target,
                "prompt_tokens": len(prefix),
            }
        )
    return result


def save_checkpoint(model, optimizer, output, state):
    import torch

    destination = output / f"checkpoint-{state['step']:04d}"
    if destination.exists():
        if not (destination / "COMMIT.json").exists():
            raise ValueError("uncommitted checkpoint occupies destination")
        return destination
    temporary = output / f".checkpoint-{state['step']:04d}-{uuid.uuid4().hex[:8]}"
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(
        {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        temporary / "rng.pt",
    )
    probe.runtime.save(temporary / "STATE.json", state)
    hashes = {p.name: probe.campaign.sha(p) for p in temporary.iterdir() if p.is_file()}
    probe.runtime.save(temporary / "COMMIT.json", {"files": hashes, "step": state["step"]})
    temporary.rename(destination)
    return destination


def run(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = args.prepared.resolve()
    examples_path = prepared / "examples.jsonl"
    rows = [json.loads(line) for line in examples_path.open()]
    if not rows:
        raise ValueError("no training examples")
    tokenizer = AutoTokenizer.from_pretrained(probe.campaign.MODELS["4b"], local_files_only=True)
    examples = tokenize_rows(rows, tokenizer)
    training_project = Path(
        "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training"
    )
    plan = {
        "question": "Can supervised reference questions improve a matched generated planner?",
        "objective": "SFT of question-list tokens only; no helper/final answer training",
        "model": probe.campaign.MODELS["4b"],
        "examples_sha256": probe.campaign.sha(examples_path),
        "preparation_sha256": probe.campaign.sha(prepared / "MANIFEST.json"),
        "source_sha256": probe.campaign.sha(Path(__file__)),
        "probe_dependency_sha256": probe.campaign.sha(Path(probe.__file__)),
        "parents": len(examples),
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": 16,
        "microbatch_size": 1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "max_context": 2048,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
        "environment_lock_sha256": probe.campaign.sha(training_project / "uv.lock"),
        "training_order": [
            [examples[i]["id"] for i in epoch_order(len(examples), args.seed, epoch)]
            for epoch in range(args.epochs)
        ],
    }
    if (output / "PLAN.json").exists():
        if not args.resume or json.loads((output / "PLAN.json").read_text()) != plan:
            raise ValueError("resume requires unchanged plan and explicit --resume")
    else:
        probe.runtime.save(output / "PLAN.json", plan)
    snapshots = sorted(p for p in output.glob("checkpoint-*") if (p / "COMMIT.json").exists())
    restored = snapshots[-1] if args.resume and snapshots else None
    state = {"step": 0, "epoch": 0, "cursor": 0, "training_seconds": 0.0}
    if restored:
        receipt = json.loads((restored / "COMMIT.json").read_text())
        for name, digest in receipt["files"].items():
            if probe.campaign.sha(restored / name) != digest:
                raise ValueError("checkpoint file checksum mismatch")
        state = json.loads((restored / "STATE.json").read_text())
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600, lease - 600)
    invocation = uuid.uuid4().hex[:12]
    import psutil

    probe.runtime.save(
        output / f"OWNER-{invocation}.json",
        {
            "pid": os.getpid(),
            "create_time": psutil.Process().create_time(),
            "started": time.time(),
            "deadline": deadline,
            "allocation_end": lease,
        },
    )

    def stop(*_):
        global STOP
        STOP = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    model = optimizer = None
    failure = None
    try:
        if deadline < time.time() + 180 or not torch.cuda.is_available():
            raise RuntimeError("no usable GPU allocation time")
        torch.set_num_threads(4)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        base = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
        )
        if restored:
            model = PeftModel.from_pretrained(
                base, restored, is_trainable=True, autocast_adapter_dtype=True
            )
        else:
            model = get_peft_model(
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
        parameters = [p for p in model.parameters() if p.requires_grad]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not names or any("lora_" not in n for n in names):
            raise ValueError("only LoRA parameters may train")
        if any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("LoRA parameters must be FP32")
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
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
        probe.runtime.save(
            output / f"LOAD-{invocation}.json",
            {
                "trainable_parameters": sum(p.numel() for p in parameters),
                "names": names,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "prompt_tokens": [x["prompt_tokens"] for x in examples],
                "target_tokens": [len(x["target_ids"]) for x in examples],
            },
        )
        save_checkpoint(model, optimizer, output, state)
        for epoch in range(state["epoch"], args.epochs):
            order = epoch_order(len(examples), args.seed, epoch)
            for cursor in range(state["cursor"], len(order), 16):
                if STOP or time.time() >= deadline - 90:
                    break
                batch = [examples[i] for i in order[cursor : cursor + 16]]
                denominator = sum(len(x["target_ids"]) for x in batch)
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                nll = 0.0
                for row in batch:
                    ids = torch.tensor([row["input_ids"]], device="cuda:0")
                    target = torch.tensor([row["target_ids"]], device="cuda:0")
                    logits = model(
                        input_ids=ids, use_cache=False, logits_to_keep=target.shape[1]
                    ).logits
                    loss = target_loss(logits, target)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite target loss")
                    (loss / denominator).backward()
                    nll += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                epoch_done = cursor + len(batch) == len(order)
                state = {
                    "step": state["step"] + 1,
                    "epoch": epoch + 1 if epoch_done else epoch,
                    "cursor": 0 if epoch_done else cursor + len(batch),
                    "training_seconds": state["training_seconds"] + elapsed,
                }
                record = {
                    **state,
                    "nll": nll / denominator,
                    "target_tokens": denominator,
                    "gradient_norm": float(norm),
                    "seconds": elapsed,
                    "parents": [r["id"] for r in batch],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                # A resume may replay work after the last checkpoint. Preserve each attempt.
                probe.runtime.save(
                    output / "steps" / f"{invocation}-{state['step']:04d}.json", record
                )
                probe.campaign.snapshot(output / "STATUS.json", {"state": "training", **record})
                print(json.dumps(record), flush=True)
                if state["step"] % 8 == 0 or epoch_done:
                    save_checkpoint(model, optimizer, output, state)
            if STOP or time.time() >= deadline - 90:
                break
        save_checkpoint(model, optimizer, output, state)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        probe.campaign.snapshot(
            output / "STATUS.json",
            {
                "state": "failed" if failure else "finished_or_capped",
                **state,
                "failure": failure,
                "updated": time.time(),
            },
        )
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                **state,
                "failure": failure,
                "stopping": STOP,
                "ended": time.time(),
                "complete": state["epoch"] >= args.epochs,
                "deadline": deadline,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
