"""Continue each fixed teacher by two genuine epochs with restored source048 AdamW."""

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
from pathlib import Path

import dose_common as d


def prepare(args):
    import torch
    from transformers import AutoTokenizer

    d.require(
        args.root.resolve().is_relative_to(d.ROOT) and args.hours == 0.5,
        "fixed campaign root and30-minute cumulative cap required",
    )
    original, prior, receipt = d.original(args.teacher)
    recipe = d.recipe()
    prepared = Path(prior["prepared"])
    manifest = d.read(prepared / "MANIFEST.json")
    rows = [json.loads(line) for line in (prepared / "rows.jsonl").read_text().splitlines()]
    recipe.validate_rows(rows, manifest)
    tokenizer = AutoTokenizer.from_pretrained(prior["model"], local_files_only=True)
    examples = recipe.tokenize_rows(rows, tokenizer)
    d.require(
        sum(len(e["target_ids"]) for e in examples) == d.LABEL_TOKENS,
        "frozen per-epoch target dose differs",
    )
    paths = {
        Path(__file__).resolve(),
        Path(d.__file__).resolve(),
        *(d.SOURCE / filename for filename in d.SOURCE_PINS),
        Path(recipe.probe.__file__).resolve(),
        Path(recipe.probe.runtime.__file__).resolve(),
        Path(recipe.probe.campaign.__file__).resolve(),
    }
    plan = dict(
        schema="textcraft-matched-teacher-continuation-20260928-v1",
        teacher=args.teacher,
        seed=d.SEED,
        original=receipt,
        prepared=str(prepared),
        rows_sha256=prior["rows_sha256"],
        prepared_manifest_sha256=prior["prepared_manifest_sha256"],
        model=prior["model"],
        base_manifest_sha256=prior["base_manifest_sha256"],
        tasks=32,
        rows=366,
        original_epochs=1,
        additional_epochs=2,
        cumulative_epochs=3,
        original_updates=23,
        additional_updates=46,
        cumulative_updates=69,
        fixed_readout_updates=[46, 69],
        target_tokens_per_epoch=d.LABEL_TOKENS,
        additional_target_tokens=2 * d.LABEL_TOKENS,
        cumulative_target_tokens=3 * d.LABEL_TOKENS,
        effective_batch=16,
        microbatch=1,
        last_update_rows=14,
        additional_microbatches=732,
        cumulative_microbatches=1098,
        learning_rate=1e-4,
        weight_decay=0.0,
        gradient_clip=1.0,
        dtype=prior["dtype"],
        lora=prior["lora"],
        optimizer="Restore original SFT AdamW including moments and step23; no reset",
        rng="Restore saved Python, torch and CUDA state after model/optimizer loading",
        shuffle="source048 epoch_order(366,2026092208,epoch), new epochs1 and2",
        target_objective="source048 target_loss SUM / minibatch target-token count; T1 JSON+EOS",
        target_only_json_eos=True,
        max_context=8192,
        max_target=256,
        checkpoint_steps=list(range(24, 70)),
        budget_seconds=args.hours * 3600,
        endpoint_selection="Prospectively fixed cumulative46/69; never selected by VAL",
        source_sha256={str(path): d.sha(path) for path in sorted(paths)},
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            torch=torch.__version__,
            **{name: importlib.metadata.version(name) for name in ("transformers", "peft")},
        ),
        caveat="Dose control retains each teacher's original chronological training row order. "
        "Action/label multisets and total dose match, but minibatch target order and prompt "
        "histories do not. This does not solve the visibility/distribution confound.",
    )
    d.require(
        d.sha(Path(prior["model"]) / "local-research-manifest.json")
        == plan["base_manifest_sha256"],
        "base model manifest changed",
    )
    output = args.root / f"train-{args.teacher}"
    path = output / "PLAN.json"
    if path.exists():
        d.require(args.resume and d.read(path) == plan, "resume requires exact unchanged plan")
    else:
        d.save(path, plan)
    return output, plan, examples, original / "checkpoint-0023"


def run(args):
    output, plan, examples, checkpoint = prepare(args)
    if args.prepare_only:
        print(
            {
                "prepared": True,
                "GPU_loaded": False,
                "output": str(output),
                "additional_updates": 46,
                "estimated_seconds": plan["original"]["new_two_epoch_seconds_estimate"],
            },
            flush=True,
        )
        return
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    state = dict(
        plan["original"]["state"],
        cumulative_microbatches=366,
        cumulative_target_tokens=8820,
        continuation_plan_sha256=d.sha(output / "PLAN.json"),
    )
    committed = sorted(p for p in output.glob("checkpoint-*") if (p / "COMMIT.json").exists())
    if committed:
        d.require(args.resume, "existing continuation requires explicit resume")
        checkpoint = committed[-1]
        state, _ = d.verify_checkpoint(checkpoint)
        d.require(
            state["continuation_plan_sha256"] == d.sha(output / "PLAN.json"),
            "resumed checkpoint belongs to different plan",
        )
    d.validate_state(state)
    if state["step"] == 69:
        return
    owners = {p.stem.removeprefix("OWNER-") for p in output.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in output.glob("TERMINAL-*.json")}
    d.require(owners == terminals, "unresolved prior owner")
    spent = sum(d.read(p)["elapsed_seconds"] for p in output.glob("TERMINAL-*.json"))
    d.require(spent < plan["budget_seconds"], "cumulative owner cap exhausted")
    d.require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "parent must assign exactly one GPU",
    )
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    started = time.time()
    deadline = min(started + plan["budget_seconds"] - spent, lease - 600)
    d.require(deadline > started + 180, "insufficient allocation margin")
    recipe = d.recipe()
    lock = (recipe.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open(
        "a"
    )
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = uuid.uuid4().hex[:12]
    d.save(
        output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            source=str(Path(__file__).resolve()),
            source_sha256=plan["source_sha256"],
        ),
    )
    stopped, failure = False, None
    model = base = optimizer = parameters = None

    def stop(*_):
        nonlocal stopped
        stopped = True

    def guard(reserve=90):
        if stopped or (output / "STOP").exists() or time.time() >= deadline - reserve:
            raise TimeoutError("bounded continuation stopped; preserve committed checkpoints")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        torch.set_num_threads(4)
        random.seed(d.SEED)
        torch.manual_seed(d.SEED)
        torch.cuda.manual_seed_all(d.SEED)
        base = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(
            base, checkpoint, is_trainable=True, autocast_adapter_dtype=True
        )
        parameters = [v for v in model.parameters() if v.requires_grad]
        d.require(
            parameters and all(v.dtype == torch.float32 for v in parameters),
            "FP32 trainable LoRA required",
        )
        d.require(
            all("lora_" in n for n, v in model.named_parameters() if v.requires_grad),
            "only LoRA may train",
        )
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
        d.restore_optimizer_rng(optimizer, checkpoint, state["step"], device="cuda:0")
        d.save(
            output / f"LOAD-{owner}.json",
            dict(
                restored=str(checkpoint),
                step=state["step"],
                base_dtype=str(base.dtype),
                lora_dtypes=sorted({str(v.dtype) for v in parameters}),
                gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda,
                optimizer_steps=sorted({int(v["step"]) for v in optimizer.state.values()}),
            ),
        )
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        for epoch, cursor, indices in d.schedule(state):
            guard()
            d.require((epoch, cursor) == (state["epoch"], state["cursor"]), "schedule drift")
            batch = [examples[i] for i in indices]
            denominator = sum(len(row["target_ids"]) for row in batch)
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            nll = 0.0
            for row in batch:
                ids = torch.tensor([row["input_ids"]], device="cuda:0")
                targets = torch.tensor([row["target_ids"]], device="cuda:0")
                loss = recipe.target_loss(
                    model(input_ids=ids, logits_to_keep=targets.shape[1]).logits, targets
                )
                d.require(bool(torch.isfinite(loss)), "nonfinite target loss")
                (loss / denominator).backward()
                nll += float(loss.detach())
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            d.require(math.isfinite(norm) and norm > 0, "zero/nonfinite gradient")
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - step_started
            state = d.advance(state, rows=len(batch), tokens=denominator, seconds=elapsed)
            d.require(
                {int(v["step"]) for v in optimizer.state.values()} == {state["step"]},
                "actual Adam update does not match cumulative step",
            )
            receipt = dict(
                **state,
                rows=len(batch),
                target_tokens=denominator,
                target_nll=nll / denominator,
                gradient_norm=norm,
                seconds=elapsed,
                row_indices=indices,
                shuffle_epoch=epoch,
            )
            d.save(output / "steps" / f"{state['step']:04d}.json", receipt)
            recipe.save_checkpoint(model, optimizer, output, state)
            print(
                {
                    "step": state["step"],
                    "epoch": state["epoch"],
                    "target_nll": nll / denominator,
                    "seconds": elapsed,
                },
                flush=True,
            )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        model = base = optimizer = parameters = None
        gc.collect()
        torch.cuda.empty_cache()
        d.save(
            output / f"TERMINAL-{owner}.json",
            dict(
                **state,
                failure=failure,
                complete=state["step"] == 69 and failure is None,
                stopped=stopped,
                elapsed_seconds=time.time() - started,
                ended=time.time(),
                deadline=deadline,
            ),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=d.OUTPUT)
    parser.add_argument("--teacher", choices=d.ORIGINALS, required=True)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
