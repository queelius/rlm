"""Explicit cp1/Adam/RNG continuation with new FP16 on-policy rollouts only."""

import math
import random
from pathlib import Path

import textcraft_stopped_amendment as amendment

SEED = 2026092253
FP16_PROBE = amendment.ROOT / "textcraft-precision-probe-001"


def fresh_jobs(jobs):
    return {
        str(batch): [dict(j, seed=SEED + batch * 10 + j["repeat"]) for j in jobs]
        for batch in (2, 3)
    }


def restore_rng(checkpoint):
    import torch

    rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])


def restore_optimizer_rng(parameters, checkpoint):
    import torch

    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0)
    saved = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)

    def groups(obj):
        return [{k: v for k, v in g.items() if k != "params"} for g in obj["param_groups"]]

    if (
        groups(saved) != groups(optimizer.state_dict())
        or len(saved["state"]) != len(parameters)
        or any(float(row["step"]) != 1 for row in saved["state"].values())
    ):
        raise ValueError("exact ancestor Adam hyperparameters/step/parameter inventory required")
    optimizer.load_state_dict(saved)
    restore_rng(checkpoint)
    return optimizer


def record_before(directory, before, captured):
    from rl_textcraft_terminal import c

    if set(before) != set(captured):
        raise ValueError("all fresh calls require captured generation scores")
    differences = [
        abs(a - b) for cid in before for a, b in zip(before[cid], captured[cid], strict=True)
    ]
    finite = bool(differences) and all(math.isfinite(x) for x in differences)
    maximum = max(differences) if differences else None
    mean = sum(differences) / len(differences) if differences else None
    c.save(
        directory / "BEFORE_LOGPS.json",
        dict(
            logps=before,
            captured_generation_logps=captured,
            generation_replay_max_abs=maximum,
            generation_replay_mean_abs=mean,
            all_finite=finite,
            token_count=len(differences),
            provenance="New FP16 on-policy generation and unchanged-weight full-forward replay; "
            "saved before numerical threshold rejection",
        ),
    )
    if not finite or maximum > 0.25 or mean > 0.025:
        raise ValueError("generation/replay discrepancy exceeded declared tolerance")


def prepare(args):
    import rl_textcraft_terminal as rl

    c, read, sha = rl.c, rl.audit.read, rl.c.inputs.sha
    if args.root.resolve() != c.ROOT or not 0 < args.hours <= 3:
        raise ValueError("fixed campaign and <=3hours required")
    ancestor, _, pins = amendment.qualify(args.fp16_continuation_amendment, amendment.RL_OUTPUT)
    endpoint = Path(read(args.fp16_continuation_amendment)["endpoint"])
    probe = read(FP16_PROBE / "SUMMARY.json")
    terminals = list(FP16_PROBE.glob("TERMINAL-*.json"))
    if (
        len(terminals) != 1
        or not read(terminals[0])["complete"]
        or not probe["all_finite"]
        or probe["fixed_completed"] != 47
        or probe["fresh_completed"] != 2
    ):
        raise ValueError("completed finite precision probe required, not automatic promotion")
    original = read(Path(ancestor["first_batch"]["output"]) / "PLAN.json")
    probe_calls = amendment.RL_OUTPUT / "batches/sample-0002/rollout/calls"
    longest = max(
        probe_calls.glob("*.json"),
        key=lambda p: len(read(p)["input_token_ids"]) + len(read(p)["output_token_ids"]),
    )
    for p in (FP16_PROBE / "PLAN.json", FP16_PROBE / "SUMMARY.json", terminals[0], longest):
        pins[str(p)] = sha(p)
    plan = dict(ancestor)
    plan.update(
        schema="textcraft-explicit-fp16-one-update-continuation-v1",
        status="PROPOSED_NOT_GPU_ACCEPTED",
        restore_checkpoint=str(endpoint),
        restore_state=read(endpoint / "STATE.json"),
        ancestor_receipt_sha256=pins,
        ancestor_output=str(amendment.RL_OUTPUT),
        ancestor_actual_steps=1,
        maximum_optimizer_updates=2,
        maximum_new_optimizer_updates=1,
        maximum_sampled_batches=3,
        maximum_new_sampled_batches=2,
        seed=SEED,
        fresh_jobs=fresh_jobs(original["jobs"]),
        budget_seconds=args.hours * 3600,
        base_dtype="torch.float16",
        lora_dtype="torch.float32",
        optimizer="restored_AdamW_step1",
        numeric_probe_call=str(longest),
        numeric_probe_not_training_data=True,
        discarded_training_batch=str(amendment.RL_OUTPUT / "batches/sample-0002"),
        discard_reason="BF16 behavior is not reused as FP16 on-policy training",
        first_batch_reused=False,
        maximum_new_scientific_episode_calls=2 * 32 * 96,
        maximum_qualification_calls=0,
        intervention="Only base arithmetic BF16->FP16; actual cp1 FP32LoRA+Adam+RNG retained; "
        "fresh fixed seeds are necessary new on-policy sampling, not recycled BF16sample2.",
        source_sha256={
            str(Path(m.__file__).resolve()): sha(Path(m.__file__).resolve())
            for m in (
                rl,
                rl.reader,
                rl.reader.shared,
                rl.reader.public,
                c,
                c.bridge,
                c.inputs,
                c.probe,
                c.probe.runtime,
                c.probe.campaign,
                rl.audit,
                rl.readiness,
                rl.loss_math,
                rl.checkpoints,
                amendment,
            )
        },
        caveat="One additional TRAIN update at changed training arithmetic, not an RL efficacy "
        "claim. BF16 cp2 evaluation fixed later; no cp1 SFT control yet. "
        "Failed ancestor preserved.",
    )
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__))
    path = args.output / "PLAN.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("immutable continuation PLAN changed")
    else:
        c.save(path, plan)
    return plan


def pre_rollout_probe(model, parameters, plan, output):
    import rl_textcraft_terminal as rl
    import torch

    record = rl.audit.read(Path(plan["numeric_probe_call"]))
    loss = rl.loss_math.replay_action_loss(model, record, 1.0)
    finite_loss = bool(torch.isfinite(loss))
    if finite_loss:
        loss.backward()
    finite_grads = all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)
    has_grad = any(p.grad is not None for p in parameters)
    base_grad = any(p.grad is not None for n, p in model.named_parameters() if "lora_" not in n)
    rl.c.save(
        output / "FP16-BACKWARD-PREFLIGHT.json",
        dict(
            call_id=record["call_id"],
            finite_loss=finite_loss,
            finite_gradients=finite_grads,
            has_gradient=has_grad,
            base_gradient=base_grad,
            optimizer_called=False,
            data_used_for_update=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    model.zero_grad(set_to_none=True)
    restore_rng(Path(plan["restore_checkpoint"]))
    if not finite_loss or not finite_grads or not has_grad or base_grad:
        raise ValueError("FP16 longest-prefix finite backward qualification failed")
