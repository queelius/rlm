"""One extra public-teacher SFT optimizer step from exactly the RL warm checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import signal
import time
import uuid
from pathlib import Path

import fresh_common as f
import train_textcraft_sft as recipe

TEACHER = f.ROOT / "textcraft-public-discovery-prototype-001"
MANIFEST_SHA = "c69ef258a07f4c4f9b45b5bc044880f1cc9aa884e510fe590f3ddf77f19441da"
AUDIT_SHA = "885be2299ad24139709c6b7505ea2d460db81cb0de1ee4534704d60c490272c7"


def objective(model, row, denominator):
    import torch

    ids = torch.tensor([row["input_ids"]], device=model.device)
    targets = torch.tensor([row["target_ids"]], device=model.device)
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=targets.shape[1]).logits
    return f.rl.checkpoints.target_loss(logits, targets) / denominator


def prepare(args):
    if not 0 < args.hours <= 1 or not args.output.resolve().is_relative_to(f.ROOT):
        raise ValueError("campaign output and at most60minutes required")
    if (
        f.sha(TEACHER / "MANIFEST.json") != MANIFEST_SHA
        or f.sha(TEACHER / "PUBLIC-REPLAY-AUDIT.json") != AUDIT_SHA
    ):
        raise ValueError("original public teacher qualification changed")
    manifest = f.read(TEACHER / "MANIFEST.json")
    if f.sha(TEACHER / "rows.jsonl") != manifest["rows_sha256"]:
        raise ValueError("original366 public teacher rows changed")
    rows = [json.loads(line) for line in (TEACHER / "rows.jsonl").read_text().splitlines()]
    recipe.validate_rows(rows, manifest)
    if any(not r["task_id"].startswith("textcraft_synth.train.") for r in rows):
        raise ValueError("SFT control must contain only original TRAIN data")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        f.c.BASE, local_files_only=True, trust_remote_code=False
    )
    examples = recipe.tokenize_rows(rows, tokenizer)
    denominator = sum(len(e["target_ids"]) for e in examples)
    pins = f.source_pins(Path(__file__), "raw")
    pins[str(Path(recipe.__file__).resolve())] = f.sha(Path(recipe.__file__))
    plan = dict(
        schema="textcraft-one-extra-public-sft-20260928-v1",
        split="train",
        teacher=str(TEACHER),
        teacher_manifest_sha256=MANIFEST_SHA,
        rows_sha256=manifest["rows_sha256"],
        teacher_replay_audit_sha256=AUDIT_SHA,
        rows=366,
        tasks=32,
        selected_rows="All original366 in saved order; no outcome selection",
        supervised_tokens=denominator,
        adapter=f.adapter_binding(f.WARM),
        model=str(f.c.BASE),
        model_manifest_sha256=f.sha(f.c.BASE / "local-research-manifest.json"),
        base_dtype="float16",
        lora_dtype="float32",
        learning_rate=2e-5,
        optimizer="fresh_AdamW",
        weight_decay=0,
        clip_grad_norm=1.0,
        objective="Original teacher-target T1 token-mean negative log likelihood, including EOS",
        maximum_new_optimizer_updates=1,
        budget_seconds=args.hours * 3600,
        matched="Warm checkpoint, fresh AdamW, one update, LR2e-5, gradient clip1, numeric types",
        unmatched="Old public366 expert-action data versus new RL trajectories; token count, "
        "prefix distribution, T1 SFT versus T0.5 RL likelihood, and weighting differ. "
        "This is a one-step extra-SFT control, not matched FLOPs/token dose or unbiased RL.",
        source_sha256=pins,
    )
    f.persist(args.output / "PLAN.json", plan)
    return plan, examples


def run(args):
    plan, examples = prepare(args)
    if args.prepare_only:
        print(
            json.dumps(
                dict(
                    prepared=True,
                    GPU_loaded=False,
                    rows=len(examples),
                    supervised_tokens=plan["supervised_tokens"],
                    plan_sha256=f.sha(args.output / "PLAN.json"),
                )
            )
        )
        return
    if list(args.output.glob("OWNER-*.json")):
        raise ValueError("existing SFT attempt; no implicit retry")
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + plan["budget_seconds"], lease - 600)
    f.rl.STOP = False
    f.rl.guard(deadline, 180)
    lock = (f.c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner, started = uuid.uuid4().hex[:12], time.time()
    f.c.save(
        args.output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            allocation_end=lease,
            source=str(Path(__file__).resolve()),
            source_sha256=plan["source_sha256"],
        ),
    )
    model = base = optimizer = parameters = None
    endpoint, failure, complete, actual_steps, update = None, None, False, 0, None

    def stop(*_):
        f.rl.STOP = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("parent must assign exactly one GPU")
        torch.set_num_threads(4)
        torch.manual_seed(202609280500)
        base = AutoModelForCausalLM.from_pretrained(
            f.c.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base,
            str(f.WARM),
            adapter_name="textcraft_action",
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
        parameters = f.rl.set_mode(model, training=True)
        optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0)
        old = [v.detach().cpu().clone() for v in parameters]
        f.rl.set_mode(model, training=False)
        before = []
        with torch.no_grad():
            for row in examples:
                f.rl.guard(deadline, 120)
                before.append(float(objective(model, row, plan["supervised_tokens"])))
        f.c.save(args.output / "BEFORE.json", dict(token_mean_nll=sum(before), row_losses=before))
        f.rl.set_mode(model, training=True)
        optimizer.zero_grad(set_to_none=True)
        during, largest_gap = 0.0, 0.0
        for row, expected in zip(examples, before, strict=True):
            f.rl.guard(deadline, 90)
            loss = objective(model, row, plan["supervised_tokens"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite teacher loss")
            value = float(loss.detach())
            during += value
            largest_gap = max(largest_gap, abs(value - expected))
            loss.backward()
        if any(v.grad is not None for n, v in model.named_parameters() if "lora_" not in n):
            raise ValueError("frozen base received gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        if not norm > 0 or largest_gap > 0.002:
            raise ValueError("SFT gradient or eval/train replay discrepancy")
        f.rl.guard(deadline, 90)
        f.c.save(args.output / "OPTIMIZER-STEP-STARTED.json", dict(previous_step=0))
        optimizer.step()
        actual_steps = 1
        delta = math.sqrt(
            sum(
                float((v.detach().cpu() - a).double().square().sum())
                for v, a in zip(parameters, old, strict=True)
            )
        )
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("nonfinite or zero SFT parameter change")
        update = dict(
            loss_before=sum(before),
            training_loss=during,
            gradient_norm=norm,
            adapter_l2_delta=delta,
            maximum_weighted_row_replay_gap=largest_gap,
        )
        state = dict(
            step=1,
            sample_cursor=1,
            zero_streak=0,
            plan_sha256=f.sha(args.output / "PLAN.json"),
            update=update,
        )
        endpoint = f.rl.save_boundary(model, optimizer, args.output, state)
        f.rl.set_mode(model, training=False)
        after = []
        with torch.no_grad():
            for row in examples:
                f.rl.guard(deadline, 5)
                after.append(float(objective(model, row, plan["supervised_tokens"])))
        f.c.save(args.output / "AFTER.json", dict(token_mean_nll=sum(after), row_losses=after))
        update["loss_after"] = sum(after)
        complete = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        model = base = optimizer = parameters = None
        gc.collect()
        torch.cuda.empty_cache()
        summary = dict(
            complete=complete,
            failure=failure,
            stopped=f.rl.STOP,
            actual_new_optimizer_steps=actual_steps,
            endpoint=str(endpoint) if endpoint else None,
            endpoint_usable=complete and endpoint is not None,
            update=update,
            ended=time.time(),
            elapsed_seconds=time.time() - started,
            deadline=deadline,
        )
        f.c.save(args.output / "SUMMARY.json", summary)
        f.c.save(args.output / f"TERMINAL-{owner}.json", summary)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.75)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
