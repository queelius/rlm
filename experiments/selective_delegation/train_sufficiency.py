"""Fixed one-epoch sufficiency SFT arms; audited planner-training mechanics reused."""

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
from collections import Counter, defaultdict
from pathlib import Path

import probe
import sufficiency_probe as sufficiency
import train_planner as recipe

STOP = False
SEED = 2026092189
epoch_order, target_loss, save_checkpoint = (
    recipe.epoch_order,
    recipe.target_loss,
    recipe.save_checkpoint,
)


def resolve_options(args):
    args.role, args.epochs, args.seed = "sufficiency", 1, SEED
    if not 0 < args.hours <= 0.5 or args.learning_rate != 1e-4:
        raise ValueError("fixed LR1e-4 and at most30 minutes per arm")


def validate_input_role(rows, manifest, role):
    if role != "sufficiency" or manifest.get("role") != role:
        raise ValueError("sufficiency role required")
    if manifest.get("arm") not in ("joint", "positive_only"):
        raise ValueError("explicit paired training arm required")
    if (
        len(rows) != 512
        or manifest.get("examples") != 512
        or manifest.get("training_parents") != 256
    ):
        raise ValueError("exactly512 examples required")
    if len({r["id"] for r in rows}) != 512 or any(r["split"] != "train" for r in rows):
        raise ValueError("unique train-only examples required")
    parents = Counter(r["parent_id"] for r in rows)
    if (
        len(parents) != 256
        or set(parents.values()) != {2}
        or set(parents) != set(manifest["selected_parents"])
    ):
        raise ValueError("exactly256 parents with two slots required")
    groups = defaultdict(list)
    for row in rows:
        value = sufficiency.parse_output(row["target"])
        if (value["answerable"] and not value["answer"].strip()) or (
            not value["answerable"] and value["answer"] != ""
        ):
            raise ValueError("exact supported answer or empty negative target required")
        if not row["prompt"].startswith(sufficiency.INSTRUCTION):
            raise ValueError("exact sufficiency prompt instruction required")
        groups[row["parent_id"]].append((row["prompt"], row["target"], value["answerable"]))
    for group in groups.values():
        if manifest["arm"] == "joint" and {r[2] for r in group} != {True, False}:
            raise ValueError("joint requires both official labels")
        if manifest["arm"] == "positive_only" and (group[0] != group[1] or not group[0][2]):
            raise ValueError("positive-only requires two identical supported prompt/targets")
    return 256


def tokenize_rows(rows, tokenizer, *, role="sufficiency"):
    if role != "sufficiency":
        raise ValueError("sufficiency role required")
    examples = []
    for row in rows:
        if row["split"] != "train":
            raise ValueError("train rows only")
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
        if len(prefix) + len(target) > 8192 or len(target) > 128:
            raise ValueError("context8192/target128 exceeded; never truncate")
        examples.append(
            dict(
                id=row["id"],
                parent_id=row["parent_id"],
                input_ids=prefix + target[:-1],
                target_ids=target,
                prompt_tokens=len(prefix),
            )
        )
    return examples


def run(args):
    global STOP
    STOP = False
    resolve_options(args)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = args.prepared.resolve()
    examples_path = prepared / "examples.jsonl"
    manifest = json.loads((prepared / "MANIFEST.json").read_text())
    if probe.campaign.sha(examples_path) != manifest["examples_sha256"]:
        raise ValueError("prepared examples checksum differs")
    rows = [json.loads(line) for line in examples_path.open()]
    if not rows:
        raise ValueError("no training examples")
    parents = validate_input_role(rows, manifest, args.role)
    base_manifest = Path(probe.campaign.MODELS["4b"]) / "local-research-manifest.json"
    if args.role == "sufficiency":
        if manifest["model"] != probe.campaign.MODELS["4b"] or manifest[
            "model_manifest_sha256"
        ] != probe.campaign.sha(base_manifest):
            raise ValueError("sufficiency preparation model identity differs")
        for name, digest in {"examples.jsonl": manifest["examples_sha256"]}.items():
            if probe.campaign.sha(prepared / name) != digest:
                raise ValueError("sealed sufficiency preparation artifact changed: " + name)
    tokenizer = AutoTokenizer.from_pretrained(probe.campaign.MODELS["4b"], local_files_only=True)
    examples = tokenize_rows(rows, tokenizer, role=args.role)
    training_project = Path(
        "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training"
    )
    plan = {
        "question": "Can joint answer/sufficiency training outperform positive-only SFT?",
        "objective": "Masked target JSON+EOS SFT",
        "model": probe.campaign.MODELS["4b"],
        "examples_sha256": probe.campaign.sha(examples_path),
        "preparation_sha256": probe.campaign.sha(prepared / "MANIFEST.json"),
        "source_sha256": probe.campaign.sha(Path(__file__)),
        "training_recipe_sha256": probe.campaign.sha(Path(recipe.__file__)),
        "sufficiency_prompt_sha256": probe.campaign.sha(Path(sufficiency.__file__)),
        "comparison_arm": manifest["arm"],
        "probe_dependency_sha256": probe.campaign.sha(Path(probe.__file__)),
        "parents": parents,
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
    if args.role == "sufficiency":
        plan.update(
            role="sufficiency",
            examples=len(examples),
            question="Does joint answer/sufficiency SFT improve paired scoring "
            "without suppressing supported answers?",
            objective="SFT of official TRAIN answerable+answer JSON and EOS; "
            "prompt positions masked.",
            max_context=8192,
            max_target=128,
            planned_updates=32,
            budget_seconds=args.hours * 3600,
            model_manifest_sha256=probe.campaign.sha(base_manifest),
            supervision=manifest["target"],
            initialization="Fresh rank8 LoRA on released base for each arm; no warm-start adapter",
        )
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
    spent = 0.0
    if args.role == "sufficiency":
        owners = {p.stem.removeprefix("OWNER-") for p in output.glob("OWNER-*.json")}
        terminals = {p.stem.removeprefix("TERMINAL-") for p in output.glob("TERMINAL-*.json")}
        if owners != terminals:
            raise ValueError("previous sufficiency-training owner unresolved")
        spent = sum(
            json.loads(p.read_text())["elapsed_seconds"] for p in output.glob("TERMINAL-*.json")
        )
        if spent >= args.hours * 3600:
            raise ValueError("cumulative sufficiency-training budget exhausted")
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    started_owner = time.time()
    deadline = min(started_owner + max(0.0, args.hours * 3600 - spent), lease - 600)
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
        if (
            deadline < time.time() + 180
            or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
        ):
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
                    if args.role == "sufficiency" and (STOP or time.time() >= deadline - 30):
                        raise TimeoutError(
                            "sufficiency training cap during accumulation; no partial step"
                        )
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
                if args.role == "sufficiency":
                    record["example_ids"] = record["parents"]
                    record["parents"] = [r["parent_id"] for r in batch]
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
                "elapsed_seconds": time.time() - started_owner,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
