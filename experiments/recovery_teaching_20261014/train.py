"""Fixed one-update recovery/ordinary teaching; preserve the September29 data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import sys
import time
import uuid
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
DATA = ROOT / "textcraft-recovery-data-20260929-001"
MANIFEST_SHA = "64f53f5c972bb7bd1fdea7887396e7d57c0341841a6d2f0c69c7804915301f34"
LIBRARY = Path(
    "/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/"
    "experiments/selective_delegation/rl_resume_20260928"
)
OUTPUT = Path("/project/alex_phd/runs/rlm-teaching-goal-20261014/recovery-teaching-001")
SEED = 202609290201


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def native():
    path = LIBRARY / "probe_common.py"
    spec = importlib.util.spec_from_file_location("recovery_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    import torch
    from transformers import AutoTokenizer

    if args.arm not in ("recovery", "ordinary") or sha(DATA / "MANIFEST.json") != MANIFEST_SHA:
        raise ValueError("frozen recovery/ordinary input manifest required")
    manifest = read(DATA / "MANIFEST.json")
    rows_path = DATA / f"{args.arm}-rows.jsonl"
    info = manifest["arms"][args.arm]
    if sha(rows_path) != info["rows_sha256"]:
        raise ValueError("selected teaching rows changed")
    for path, expected in manifest["tokenization"]["assets_sha256"].items():
        if sha(path) != expected:
            raise ValueError("frozen tokenizer asset changed")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if len(rows) != 32 or [r["row_id"] for r in rows] != info["row_ids"]:
        raise ValueError("exact32 frozen selected rows required")
    if any(
        r["split"] != "train" or r["dataset_group"] != "A" or r["task_id"] not in info["task_ids"]
        for r in rows
    ):
        raise ValueError("A-only training identities required")
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenization"]["path"], local_files_only=True, trust_remote_code=False
    )
    examples = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target_ids = tokenizer.encode(row["target"], add_special_tokens=False)
        target_ids.append(tokenizer.eos_token_id)
        if prompt_ids + target_ids != row["input_ids"] or target_ids != row["target_ids"]:
            raise ValueError("saved request/target token encoding differs")
        examples.append(example(row))
    if sum(len(r["target_ids"]) for r in examples) != 823:
        raise ValueError("fixed target-token dose changed")
    p = native()
    binding = p.adapter_binding(Path(manifest["future_training_start"]["path"]))
    if binding != manifest["future_training_start"]:
        raise ValueError("fixed public-discovery warm actor changed")
    plan = dict(
        schema="recovery-teaching-one-step-20261014-v1",
        arm=args.arm,
        data=str(DATA),
        data_manifest_sha256=MANIFEST_SHA,
        rows_sha256=sha(rows_path),
        rows=32,
        task_ids=info["task_ids"],
        row_ids=info["row_ids"],
        supervised_tokens=823,
        prompt_tokens=sum(r["prompt_tokens"] for r in rows),
        adapter=binding,
        model=manifest["tokenization"]["path"],
        model_manifest_sha256=manifest["tokenization"]["model_manifest_sha256"],
        seed=SEED,
        optimizer="fresh_AdamW",
        learning_rate=2e-5,
        weight_decay=0,
        clip_grad_norm=1.0,
        base_dtype="float16",
        lora_dtype="float32",
        maximum_new_optimizer_updates=1,
        budget_seconds=1200,
        objective="T1 sum target JSON+EOS NLL /823; saved full sequence shifted once",
        source_sha256=p.source_pins(Path(__file__), "raw"),
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__),
        evaluation="Ordinary A task starts, not saved error prefixes; fixed seeds202609290210/211",
        caveat="24 labels identical,8 differ; prompt doses/FLOPs/history not matched. "
        "One supervised update, not RL. B excluded from construction and selection.",
    )
    if sha(Path(plan["model"]) / "local-research-manifest.json") != plan["model_manifest_sha256"]:
        raise ValueError("model manifest changed")
    path = args.output / "PLAN.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("immutable teaching plan differs")
    else:
        p.c.save(path, plan)
    return plan, examples, p


def example(row: dict) -> dict:
    """Convert full unshifted saved SFT rows to next-token replay inputs once."""
    ids, targets = row["input_ids"], row["target_ids"]
    n = row["prompt_tokens"]
    if (
        n <= 0
        or not targets
        or len(ids) != n + len(targets)
        or ids[n:] != targets
        or row["target_tokens"] != len(targets)
        or row["labels"] != [-100] * n + targets
    ):
        raise ValueError("saved causal token boundaries or target mask differ")
    return {"input_ids": ids[:-1], "target_ids": targets}


def objective(model, row: dict, denominator: int):
    import torch
    import torch.nn.functional as functional

    ids = torch.tensor([row["input_ids"]], device=model.device)
    targets = torch.tensor(row["target_ids"], device=model.device)
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(targets)).logits[0]
    return functional.cross_entropy(logits.float(), targets, reduction="sum") / denominator


def one_update(model, optimizer, examples, guard, checkpoint=None):
    """One complete token-normalized Adam update; checkpoint before post-fit checks."""
    import torch

    denominator = sum(len(row["target_ids"]) for row in examples)
    if not denominator or optimizer.state:
        raise ValueError("nonempty teaching set and fresh optimizer required")
    parameters = [v for v in model.parameters() if v.requires_grad]
    old = [v.detach().cpu().clone() for v in parameters]
    model.eval()
    before = []
    with torch.no_grad():
        for row in examples:
            guard()
            before.append(float(objective(model, row, denominator)))
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    during, maximum_gap = 0.0, 0.0
    for row, expected in zip(examples, before, strict=True):
        guard()
        loss = objective(model, row, denominator)
        if not bool(torch.isfinite(loss)) or not math.isfinite(expected):
            raise ValueError("nonfinite teaching loss")
        value = float(loss.detach())
        maximum_gap = max(maximum_gap, abs(value - expected))
        during += value
        loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    if not norm > 0 or maximum_gap > 0.002:
        raise ValueError("zero gradient or train/eval replay discrepancy")
    guard()
    optimizer.step()
    if {int(s["step"]) for s in optimizer.state.values()} != {1}:
        raise ValueError("Adam did not perform exactly one update")
    delta = math.sqrt(
        sum(
            float((p.detach().cpu() - q).double().square().sum())
            for p, q in zip(parameters, old, strict=True)
        )
    )
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("nonfinite or zero parameter change")
    result = dict(
        rows=len(examples),
        target_tokens=denominator,
        actual_new_optimizer_steps=1,
        loss_before=sum(before),
        training_loss=during,
        gradient_norm=norm,
        adapter_l2_delta=delta,
        maximum_weighted_row_replay_gap=maximum_gap,
    )
    if checkpoint is not None:
        checkpoint(result)
    model.eval()
    after = []
    with torch.no_grad():
        for row in examples:
            guard()
            value = float(objective(model, row, denominator))
            if not math.isfinite(value):
                raise ValueError("nonfinite post-update loss")
            after.append(value)
    result.update(loss_after=sum(after), row_losses_before=before, row_losses_after=after)
    return result


def run(args):
    plan, examples, p = prepare(args)
    if args.prepare_only:
        print(
            json.dumps(
                dict(
                    prepared=True,
                    GPU_loaded=False,
                    rows=len(examples),
                    target_tokens=823,
                    output=str(args.output),
                )
            )
        )
        return
    if args.output.resolve().parent != OUTPUT or list(args.output.glob("OWNER-*.json")):
        raise ValueError("new scoped attempt required; no implicit retry")
    import fcntl

    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one allocated GPU required")
    started = time.time()
    deadline = min(started + 1200, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    if deadline < started + 180:
        raise ValueError("insufficient allocation margin")
    lock = (p.c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = uuid.uuid4().hex[:12]
    p.c.save(
        args.output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            source=str(Path(__file__).resolve()),
            source_sha256=plan["source_sha256"],
        ),
    )
    stopped = False
    endpoint = result = failure = None
    model = base = optimizer = parameters = None

    def stop(*_):
        nonlocal stopped
        stopped = True

    def guard():
        if stopped or (args.output / "STOP").exists() or time.time() > deadline - 90:
            raise TimeoutError("bounded recovery teaching stopped; preserve partial attempt")

    def checkpoint(update):
        nonlocal endpoint
        if any(v.grad is not None for n, v in model.named_parameters() if "lora_" not in n):
            raise ValueError("base parameter received gradient")
        state = dict(
            step=1,
            sample_cursor=1,
            zero_streak=0,
            plan_sha256=sha(args.output / "PLAN.json"),
            update=dict(update),
        )
        endpoint = p.rl.save_boundary(model, optimizer, args.output, state)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        torch.set_num_threads(4)
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        base = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(
            base,
            plan["adapter"]["path"],
            adapter_name="textcraft_action",
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
        parameters = p.rl.set_mode(model, training=True)
        optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0)
        p.c.save(
            args.output / "LOAD.json",
            dict(
                gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda,
                base_dtype=str(base.dtype),
                trainable_dtypes=sorted({str(v.dtype) for v in parameters}),
                trainable_parameters=sum(v.numel() for v in parameters),
            ),
        )
        result = one_update(model, optimizer, examples, guard, checkpoint)
        p.c.save(args.output / "UPDATE.json", result)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        actual = sorted({int(s["step"]) for s in optimizer.state.values()}) if optimizer else []
        model = base = optimizer = parameters = None
        gc.collect()
        torch.cuda.empty_cache()
        summary = dict(
            complete=failure is None and endpoint is not None,
            endpoint_usable=failure is None and endpoint is not None,
            endpoint=str(endpoint) if endpoint else None,
            failure=failure,
            actual_optimizer_steps=actual,
            update=result,
            stopped=stopped,
            ended=time.time(),
            elapsed_seconds=time.time() - started,
            deadline=deadline,
        )
        p.c.save(args.output / "SUMMARY.json", summary)
        p.c.save(args.output / f"TERMINAL-{owner}.json", summary)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("recovery", "ordinary"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
