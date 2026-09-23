"""Biased positive-only terminal-credit update on the same saved FP16 cp1 batch."""

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

import eval_textcraft_fp16_endpoint as reference
import rl_textcraft_terminal as rl
import textcraft_fp16_continuation as fp16
import textcraft_lr_replay as signed

c, read, sha = rl.c, rl.audit.read, rl.c.inputs.sha
SOURCE = c.ROOT / "textcraft-terminal-fp16-continuation-002"
DATA = SOURCE / "batches/sample-0002/rollout"
OUTPUT = c.ROOT / "textcraft-positive-credit-replay-001"
READOUT = c.ROOT / "textcraft-fresh-positive-credit-001"
REFERENCE_READOUT = signed.READOUT


def positive_credits(credits):
    """Preserve each original positive advantage exactly; do not renormalize."""
    return [credit for credit in credits if credit.advantage > 0]


def change_lr(optimizer):
    if any(g["lr"] != 2e-5 or g["weight_decay"] != 0 for g in optimizer.param_groups) or {
        int(v["step"]) for v in optimizer.state.values()
    } != {1}:
        raise ValueError("unchanged restored Adam1 required before LR intervention")
    for group in optimizer.param_groups:
        group["lr"] = 1e-4


def valid_terminal(row):
    if (
        not row.get("complete")
        or row.get("failure")
        or row.get("stopped")
        or row.get("actual_optimizer_steps") != 2
    ):
        raise ValueError("successful single replay update, cumulative Adam2 required")


def source_identity():
    terminal, pins = reference.reader.released_terminal(SOURCE)
    valid_terminal(terminal)
    plan = read(SOURCE / "PLAN.json")
    if sha(SOURCE / "PLAN.json") != reference.TRAIN_PLAN_SHA:
        raise ValueError("fixed successful FP16 reference changed")
    checkpoint = Path(plan["restore_checkpoint"])
    commit = read(checkpoint / "COMMIT.json")
    for name, digest in commit["files"].items():
        if sha(checkpoint / name) != digest:
            raise ValueError("ancestor checkpoint differs: " + name)
    state = read(Path(terminal["endpoint"]) / "STATE.json")
    batch = read(SOURCE / "batches/sample-0002/BATCH.json")
    batch_plan = read(DATA / "PLAN.json")
    if state["sample_cursor"] != 2 or state["step"] != 2 or len(batch["episodes"]) != 32:
        raise ValueError("exact successful sample2 required")
    if batch_plan["adapter"]["sha256"] != commit["files"]["adapter_model.safetensors"]:
        raise ValueError("saved behavior not sampled from restored cp1 weights")
    for path, digest in batch["native_receipt_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("native source receipt changed")
    for path in (
        SOURCE / "PLAN.json",
        SOURCE / "SUMMARY.json",
        DATA / "PLAN.json",
        SOURCE / "batches/sample-0002/BATCH.json",
        checkpoint / "COMMIT.json",
    ):
        pins[str(path)] = sha(path)
    for path in sorted((DATA / "generation-logps").glob("*.json")):
        pins[str(path)] = sha(path)
    return dict(
        restore_checkpoint=str(checkpoint),
        source_plan=plan,
        pins=pins,
        native_receipt_sha256=batch["native_receipt_sha256"],
        episodes=32,
        calls=batch["costs"]["calls"],
        sample_cursor=2,
        source_update=state["update"],
        historical_collection_cost=batch["costs"],
    )


def prepare():
    identity = source_identity()
    plan = dict(identity["source_plan"])
    plan.update(
        schema="textcraft-same-batch-positive-credit-v1",
        learning_rate=1e-4,
        reference_learning_rate=1e-4,
        signed_reference_endpoint=signed.endpoint_binding(),
        credit_filter="advantage > 0; original weights and denominator32 preserved",
        expected_positive_credit=dict(episodes=12, calls=399, emitted_tokens=12927),
        expected_signed_credit=dict(calls=819, emitted_tokens=26933),
        budget_seconds=5400,
        saved_batch=str(DATA),
        source_identity=identity,
        new_rollout_episodes=0,
        actual_additional_updates=1,
        maximum_new_scientific_episode_calls=0,
        maximum_new_sampled_batches=0,
        maximum_sampled_batches=2,
        fresh_jobs={},
        optimizer="restored_AdamW_step1_with_explicit_lr1e4",
        intervention="Only negative terminal-credit terms are omitted versus signed1e-4. "
        "Same restored cp1/Adam1/RNG, FP16 base, batch, positive advantage weights, denominator32 "
        "and single step; no resampling, reward changes, renormalization or repeated updates.",
        caveat="Positive-only objective is biased, not unbiased RLOO or SFT matched. "
        "It changes gradient direction, norm and credited token dose; clip1 retained. "
        "Adaptive exposed-panel mechanism test, not fresh replication or proof that shared "
        "useful actions were suppressed. Negative-credit episodes remain in native audit.",
    )
    plan["source_sha256"] = {
        str(Path(m.__file__).resolve()): sha(Path(m.__file__))
        for m in (
            rl,
            signed,
            fp16,
            reference,
            reference.reader,
            reference.comparison,
            rl.audit,
            rl.readiness,
            rl.reader,
            rl.loss_math,
            rl.checkpoints,
            c,
            c.bridge,
            c.inputs,
            c.probe,
            c.probe.runtime,
            c.probe.campaign,
        )
    }
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__))
    path = OUTPUT / "PLAN.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("immutable replay PLAN changed")
    else:
        c.save(path, plan)
    return plan


def train(prepare_only=False):
    plan = prepare()
    if prepare_only:
        print(json.dumps(dict(prepared=True, GPU_loaded=False, saved_calls=909)))
        return
    if list(OUTPUT.glob("OWNER-*.json")):
        raise ValueError("no implicit resume or retry")
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + plan["budget_seconds"], lease - 600)
    rl.guard(deadline, 300)
    lock = (c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner, started = uuid.uuid4().hex[:12], time.time()
    c.save(
        OUTPUT / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            allocation_end=lease,
            source=str(Path(__file__).resolve()),
        ),
    )

    def stop(*_):
        rl.STOP = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    model = optimizer = None
    failure, complete, endpoint, actual = None, False, None, 1
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one assigned GPU required")
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            c.BASE, local_files_only=True, trust_remote_code=False
        )
        original = read(Path(plan["first_batch"]["output"]) / "PLAN.json")
        tasks = {t["id"]: t for t in rl.readiness.runtime_tasks(original)}
        episodes, calls, credits, audits = rl.native_batch(
            DATA, read(DATA / "PLAN.json"), tokenizer, tasks, c.bridge.load_world()
        )
        batch_dir = OUTPUT / "batches/sample-0002"
        c.save(
            batch_dir / "BATCH.json",
            dict(
                data=str(DATA),
                episodes=episodes,
                audits=audits,
                reused=True,
                costs=c.cost(list(calls.values())),
                native_receipt_sha256=plan["source_identity"]["native_receipt_sha256"],
            ),
        )
        base = AutoModelForCausalLM.from_pretrained(
            c.BASE,
            dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base,
            plan["restore_checkpoint"],
            adapter_name="textcraft_action",
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
        parameters = rl.set_mode(model, training=True)
        optimizer = fp16.restore_optimizer_rng(parameters, Path(plan["restore_checkpoint"]))
        change_lr(optimizer)
        fp16.pre_rollout_probe(model, parameters, plan, OUTPUT)
        state = {
            **plan["restore_state"],
            "plan_sha256": sha(OUTPUT / "PLAN.json"),
            "ancestor_commit_sha256": sha(Path(plan["restore_checkpoint"]) / "COMMIT.json"),
        }
        rl.save_boundary(model, optimizer, OUTPUT, state)
        rl.set_mode(model, training=False)
        before, captured = {}, {}
        with torch.no_grad():
            for cid, call in calls.items():
                rl.guard(deadline, 180)
                before[cid] = rl.action_logps(model, call).flatten().cpu().tolist()
                score = read(DATA / "generation-logps" / (cid + ".json"))
                if score["call_sha256"] != sha(DATA / "calls" / (cid + ".json")):
                    raise ValueError("generation-score/call binding mismatch")
                captured[cid] = score["logps"]
        fp16.record_before(batch_dir, before, captured)
        rl.set_mode(model, training=True)
        optimizer.zero_grad(set_to_none=True)
        old = [p.detach().cpu().clone() for p in parameters]
        nonzero = positive_credits(credits)
        c.save(
            batch_dir / "CREDIT-MASK.json",
            dict(
                rule="advantage > 0",
                denominator=32,
                positive_calls=len(nonzero),
                positive_episodes=len({v.episode_id for v in nonzero}),
                positive_tokens=sum(len(calls[v.call_id]["output_token_ids"]) for v in nonzero),
                all_credits=[
                    dict(
                        call_id=v.call_id,
                        episode_id=v.episode_id,
                        advantage=v.advantage,
                        optimized=v.advantage > 0,
                    )
                    for v in credits
                ],
            ),
        )
        value, gaps = 0.0, []
        for credit in nonzero:
            rl.guard(deadline, 120)
            logps = rl.action_logps(model, calls[credit.call_id])
            gaps.extend(
                abs(a - b)
                for a, b in zip(
                    logps.detach().flatten().cpu().tolist(), before[credit.call_id], strict=True
                )
            )
            objective = -credit.advantage * logps.sum() / 32
            if not torch.isfinite(objective):
                raise ValueError("nonfinite trajectory objective")
            value += float(objective.detach())
            objective.backward()
        if any(p.grad is not None for n, p in model.named_parameters() if "lora_" not in n):
            raise ValueError("base received gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        if not norm > 0:
            raise ValueError("no usable gradient")
        rl.guard(deadline, 90)
        rl.checked_optimizer_step(
            optimizer,
            batch_dir,
            previous_step=1,
            maximum=max(gaps),
            mean=sum(gaps) / len(gaps),
            count=len(gaps),
            max_tolerance=0.25,
            mean_tolerance=0.025,
        )
        actual = 2
        delta = math.sqrt(
            sum(
                float((p.detach().cpu() - o).double().square().sum())
                for p, o in zip(parameters, old, strict=True)
            )
        )
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("nonfinite/zero update")
        update = dict(
            optimizer_called=True,
            nonzero_action_calls=len(nonzero),
            objective=value,
            gradient_norm=norm,
            adapter_l2_delta=delta,
            train_eval_replay_max_abs=max(gaps),
            train_eval_replay_mean_abs=sum(gaps) / len(gaps),
            train_eval_replay_token_count=len(gaps),
        )
        state = {
            **rl.advance(state, updated=True),
            "batch_sha256": sha(batch_dir / "BATCH.json"),
            "update": update,
        }
        endpoint = rl.save_boundary(model, optimizer, OUTPUT, state)
        rl.set_mode(model, training=False)
        after = {}
        with torch.no_grad():
            for credit in nonzero:
                rl.guard(deadline, 5)
                after[credit.call_id] = (
                    rl.action_logps(model, calls[credit.call_id]).flatten().cpu().tolist()
                )
        c.save(batch_dir / "AFTER_LOGPS.json", dict(logps=after, endpoint=str(endpoint)))
        complete = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        terminal = dict(
            complete=complete,
            failure=failure,
            stopped=rl.STOP,
            actual_optimizer_steps=actual,
            new_optimizer_steps=actual - 1,
            cumulative_optimizer_steps=actual,
            committed_optimizer_steps=2 if endpoint else 1,
            committed_sampled_batches=2,
            new_sampled_batches=0,
            reused_sampled_batches=1,
            new_physical_calls=0,
            historical_collection_cost=plan["source_identity"]["historical_collection_cost"],
            endpoint=str(endpoint) if endpoint else None,
            endpoint_usable=complete,
            ended=time.time(),
            elapsed_seconds=time.time() - started,
            deadline=deadline,
        )
        c.save(OUTPUT / "SUMMARY.json", terminal)
        c.save(OUTPUT / f"TERMINAL-{owner}.json", terminal)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def endpoint_binding():
    import torch

    terminal, pins = reference.reader.released_terminal(OUTPUT)
    valid_terminal(terminal)
    plan = read(OUTPUT / "PLAN.json")
    adapter = OUTPUT / "boundaries/sample-0002/checkpoint-0002"
    if terminal["endpoint"] != str(adapter):
        raise ValueError("fixed cp2 required")
    commit, state = read(adapter / "COMMIT.json"), read(adapter / "STATE.json")
    for name, digest in commit["files"].items():
        if sha(adapter / name) != digest:
            raise ValueError("committed endpoint changed")
    optimizer = torch.load(adapter / "optimizer.pt", map_location="cpu", weights_only=True)
    if (
        {int(v["step"]) for v in optimizer["state"].values()} != {2}
        or any(g["lr"] != 1e-4 for g in optimizer["param_groups"])
        or state["step"] != 2
        or state["plan_sha256"] != sha(OUTPUT / "PLAN.json")
    ):
        raise ValueError("wrong Adam/dose/PLAN")
    if not read(OUTPUT / "batches/sample-0002/TRAIN-EVAL-REPLAY.json")["passed"]:
        raise ValueError("unqualified update")
    pins[str(OUTPUT / "PLAN.json")] = sha(OUTPUT / "PLAN.json")
    return {
        **rl.binding(adapter),
        "state": state,
        "training_plan_sha256": sha(OUTPUT / "PLAN.json"),
        "training_receipt_sha256": pins,
        "learning_rate": plan["learning_rate"],
    }


def readout(prepare_only=False):
    template, tasks = reference.reader.template_inputs()
    binding = endpoint_binding()
    plan = reference.reader.bound_plan(template, "positive_credit_same_batch", binding, OUTPUT)
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__))
    destination = READOUT / "PLAN.json"
    if destination.exists():
        if read(destination) != plan:
            raise ValueError("immutable readout changed")
    else:
        c.save(destination, plan)
    c.run(
        argparse.Namespace(output=READOUT, prepare_only=prepare_only),
        prepared_run=(plan, tasks),
        adapter=binding,
    )


def analyze(report):
    from transformers import AutoTokenizer

    comparison = reference.comparison
    paths = (REFERENCE_READOUT, READOUT)
    plans = [read(p / "PLAN.json") for p in paths]
    if (
        plans[0]["adapter"] != signed.endpoint_binding()
        or plans[1]["adapter"] != endpoint_binding()
    ):
        raise ValueError("fixed endpoint identities differ")
    comparison.match_slots(plans)
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    arms = [rl.audit.analyze(p, tokenizer, expected_task_count=16) for p in paths]
    for arm in arms:
        arm.pop("paired", None)
        arm.pop("depth_strata", None)
    rows = [
        {
            (v["task_id"], v["repeat"]): v
            for v in (read(p) for p in (directory / "episodes").glob("*.json"))
        }
        for directory in paths
    ]
    effect = comparison.profiles.compare(plans[0]["jobs"], *rows)
    result = dict(
        arms=arms,
        positive_only_minus_signed=effect,
        parents=16,
        repeats=2,
        caveat=read(OUTPUT / "PLAN.json")["caveat"],
        source_sha256=sha(Path(__file__)),
        plans={str(p): sha(p / "PLAN.json") for p in paths},
    )
    c.save(report, result)
    with report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Same-batch positive-only versus signed terminal credit\n\n"
            + result["caveat"]
            + "\n\n```json\n"
            + json.dumps(effect, indent=2)
            + "\n```\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "readout", "analyze"), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.prepare_only)
    elif args.mode == "readout":
        readout(args.prepare_only)
    else:
        if args.report is None:
            parser.error("--report required")
        analyze(args.report)
