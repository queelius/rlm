"""One exact-behavior signed RLOO update; conditional fresh-batch continuation only."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import random
import signal
import time
import uuid
from pathlib import Path

import probe_common as p


def signed_objective(logps, advantage: float):
    return -advantage * logps.sum() / 32


def probability_changes(credits, before: dict, after: dict) -> dict:
    result = {
        "scope": "Sampled action-token changes on the training batch; not KL or task accuracy"
    }
    for label, sign in (("positive", 1), ("negative", -1)):
        values = [
            new - old
            for credit in credits
            if credit.advantage * sign > 0
            for old, new in zip(before[credit.call_id], after[credit.call_id], strict=True)
        ]
        result[label] = dict(
            tokens=len(values),
            mean_sampled_token_logp_delta=sum(values) / len(values) if values else None,
        )
    return result


def prepare(args) -> dict:
    collection = p.read(args.collection / "PLAN.json")
    if (
        collection.get("schema") != "textcraft-assisted-rl-train-collection-20260928-v1"
        or collection["split"] != "train"
        or collection["phase"] != "collect"
        or collection["planned_episodes"] != 32
        or collection["base_dtype"] != "float16"
    ):
        raise ValueError("fresh FP16 TRAIN32 collection contract required")
    if not 0 < args.hours <= 1.5 or not args.output.resolve().is_relative_to(p.ROOT):
        raise ValueError("campaign output and at most90minutes required")
    checkpoint = Path(collection["adapter"]["path"])
    if p.adapter_binding(checkpoint) != collection["adapter"]:
        raise ValueError("behavior checkpoint changed")
    update = collection["update"]
    if update == 1 and checkpoint != p.WARM:
        raise ValueError("first update starts from public discovery cp23")
    if update > 1 and (p.read(checkpoint / "STATE.json").get("step") != update - 1):
        raise ValueError("continuation requires the immediately preceding committed optimizer")
    plan = dict(
        schema="textcraft-assisted-terminal-rloo-20260928-v1",
        split="train",
        execution_mode=collection["execution_mode"],
        update=update,
        collection=str(args.collection.resolve()),
        collection_plan_sha256=p.sha(args.collection / "PLAN.json"),
        adapter=collection["adapter"],
        base_dtype="float16",
        lora_dtype="float32",
        learning_rate=2e-5,
        weight_decay=0,
        clip_grad_norm=1.0,
        optimizer="fresh_AdamW" if update == 1 else "restore_previous_AdamW",
        budget_seconds=args.hours * 3600,
        maximum_new_optimizer_updates=1,
        objective="-sum_i[(r_i - mean(other three same-task rewards)) "
        "* sum_all_emitted_tokens(log pi_T0.5(token|original sampled prefix))]/32",
        baseline="Other three same-task sampled terminal native rewards; no standardization",
        reward="Binary native terminal success; schema/action errors retain trajectory reward",
        on_policy="One optimizer step only from exact collection weights and FP16 arithmetic. "
        "Any next step needs new trajectories under the committed updated weights.",
        training_targets="Original saved output_token_ids only, including EOS and incorrect "
        "ingredients. Executed binder arguments are transitions, never substituted targets.",
        token_weighting="Trajectory token SUM; raw and binder can have unequal realized "
        "length/credited-token dose despite identical96call/8192token per-episode caps.",
        max_replay_gap=0.25,
        mean_replay_gap=0.025,
        source_sha256=p.source_pins(Path(__file__), collection["execution_mode"]),
        checkpoint_policy="Commit adapter, optimizer, RNG and state after this update. "
        "A failed probability diagnostic after commit preserves but does not auto-promote it.",
        continuation_gate="Support at most four prospectively numbered fresh batches. No "
        "automatic continuation: parent reviews mixed groups, success/cost, entropy, and "
        "matched warm/extra-SFT controls. Prefer fresh TRAIN goals if near ceiling.",
        caveat="Exposed TRAIN mechanism pilot; no validation or held-out efficacy claim.",
    )
    path = args.output / "PLAN.json"
    if path.exists():
        if p.read(path) != plan:
            raise ValueError("immutable trainer PLAN differs")
    else:
        p.c.save(path, plan)
    return plan


def load_batch(directory: Path, plan: dict):
    summary = p.read(directory / "SUMMARY.json")
    native = p.read(directory / "NATIVE-AUDIT.json")
    if not summary["complete"] or summary["failure"] or native["observed"] != 32:
        raise ValueError("complete independently native-audited32 required")
    for name, digest in native["receipt_sha256"].items():
        if p.sha(Path(name)) != digest:
            raise ValueError("audited collection receipt changed")
    calls = {f.stem: p.read(f) for f in (directory / "calls").glob("*.json")}
    collected = p.read(directory / "PLAN.json")
    episodes = [
        p.read(directory / "episodes" / (job["episode_id"] + ".json")) for job in collected["jobs"]
    ]
    if any(not e["task_id"].startswith("textcraft_synth.train.") for e in episodes):
        raise ValueError("only official TRAIN identities may enter the optimizer")
    if p.sha(directory / "PLAN.json") != plan["collection_plan_sha256"]:
        raise ValueError("collection PLAN changed")
    for name, digest in collected["source_sha256"].items():
        if p.sha(Path(name)) != digest:
            raise ValueError("collected source changed before gradient replay")
    captured, entropies = {}, []
    for cid in calls:
        record = p.read(directory / "generation-logps" / (cid + ".json"))
        if record["call_sha256"] != p.sha(directory / "calls" / (cid + ".json")):
            raise ValueError("captured score belongs to different sampled tokens")
        captured[cid] = record["logps"]
        entropies.extend(record["token_entropies"])
    credits = p.rl.loss_math.batch_credits(episodes, calls)
    return (
        calls,
        credits,
        captured,
        dict(
            collection_audit_sha256=p.sha(directory / "NATIVE-AUDIT.json"),
            groups=native["groups"],
            successes=native["successes"],
            mixed_groups=native["mixed_groups"],
            physical_cost=native["physical_cost"],
            first_response_diversity=native["first_response_diversity"],
            sampled_token_entropy_mean=sum(entropies) / len(entropies),
            entropy_token_count=len(entropies),
        ),
    )


def run(args):
    plan = prepare(args)
    if args.prepare_only:
        print(
            json.dumps(
                dict(
                    prepared=True,
                    GPU_loaded=False,
                    admission="conditional_complete_collection",
                    output=str(args.output),
                    plan_sha256=p.sha(args.output / "PLAN.json"),
                )
            )
        )
        return
    if list(args.output.glob("OWNER-*.json")):
        raise ValueError("existing training attempt; no silent retry")
    calls, credits, captured, admission = load_batch(args.collection, plan)
    p.c.save(args.output / "ADMISSION.json", admission)
    nonzero = [credit for credit in credits if credit.advantage != 0]
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + plan["budget_seconds"], lease - 600)
    p.rl.STOP = False
    p.rl.guard(deadline, 180)
    lock = (p.c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner, started = uuid.uuid4().hex[:12], time.time()
    p.c.save(
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
    endpoint, failure, complete, actual_steps, diagnostics = None, None, False, 0, None

    def stop(*_):
        p.rl.STOP = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        if not nonzero:
            p.c.save(
                args.output / "CONDITIONAL-SKIP.json",
                dict(
                    reason="No within-task terminal reward variation",
                    GPU_loaded=False,
                    actual_optimizer_steps=0,
                    endpoint=None,
                ),
            )
            complete = True
            return
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("parent must assign exactly one GPU")
        torch.set_num_threads(4)
        random.seed(2026092800 + plan["update"])
        torch.manual_seed(2026092800 + plan["update"])
        base = AutoModelForCausalLM.from_pretrained(
            p.c.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base,
            plan["adapter"]["path"],
            adapter_name="textcraft_action",
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
        parameters = p.rl.set_mode(model, training=True)
        optimizer = torch.optim.AdamW(parameters, lr=plan["learning_rate"], weight_decay=0)
        if plan["update"] > 1:
            checkpoint = Path(plan["adapter"]["path"])
            commit = p.read(checkpoint / "COMMIT.json")
            for name in ("optimizer.pt", "rng.pt", "STATE.json"):
                if p.sha(checkpoint / name) != commit["files"][name]:
                    raise ValueError("previous optimizer boundary changed")
            optimizer.load_state_dict(
                torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
            )
            if {int(v["step"]) for v in optimizer.state.values()} != {plan["update"] - 1}:
                raise ValueError("previous optimizer step mismatch")
            if any(g["lr"] != 2e-5 or g["weight_decay"] != 0 for g in optimizer.param_groups):
                raise ValueError("optimizer continuation hyperparameters changed")
        p.c.save(
            args.output / "LOAD.json",
            dict(
                base_dtype=str(model.get_base_model().dtype),
                gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda,
                lora_dtypes=sorted({str(v.dtype) for v in parameters}),
                optimizer=plan["optimizer"],
                trainable_parameters=sum(v.numel() for v in parameters),
            ),
        )
        p.rl.set_mode(model, training=False)
        before, gaps = {}, []
        with torch.no_grad():
            for cid, call in calls.items():
                p.rl.guard(deadline, 120)
                before[cid] = p.rl.action_logps(model, call).flatten().cpu().tolist()
                gaps.extend(abs(a - b) for a, b in zip(before[cid], captured[cid], strict=True))
        p.c.save(
            args.output / "BEFORE_LOGPS.json",
            dict(
                logps=before,
                captured_generation_logps=captured,
                max_generation_replay_gap=max(gaps),
                mean_generation_replay_gap=sum(gaps) / len(gaps),
            ),
        )
        if (
            not all(math.isfinite(v) for v in gaps)
            or max(gaps) > plan["max_replay_gap"]
            or sum(gaps) / len(gaps) > plan["mean_replay_gap"]
        ):
            raise ValueError("unchanged generation/replay probabilities differ")
        p.rl.set_mode(model, training=True)
        optimizer.zero_grad(set_to_none=True)
        old_parameters = [v.detach().cpu().clone() for v in parameters]
        value, max_gap, gap_sum, count = 0.0, 0.0, 0.0, 0
        for credit in nonzero:
            p.rl.guard(deadline, 90)
            logps = p.rl.action_logps(model, calls[credit.call_id])
            values = logps.detach().flatten().cpu().tolist()
            gaps = [abs(a - b) for a, b in zip(values, before[credit.call_id], strict=True)]
            max_gap, gap_sum, count = (
                max(max_gap, max(gaps)),
                gap_sum + sum(gaps),
                count + len(gaps),
            )
            objective = signed_objective(logps, credit.advantage)
            if not torch.isfinite(objective):
                raise ValueError("nonfinite objective")
            value += float(objective.detach())
            objective.backward()
        if any(v.grad is not None for n, v in model.named_parameters() if "lora_" not in n):
            raise ValueError("frozen base received gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        if not norm > 0:
            raise ValueError("nonzero credits gave no gradient")
        p.rl.guard(deadline, 90)
        p.rl.checked_optimizer_step(
            optimizer,
            args.output,
            previous_step=plan["update"] - 1,
            maximum=max_gap,
            mean=gap_sum / count,
            count=count,
            max_tolerance=plan["max_replay_gap"],
            mean_tolerance=plan["mean_replay_gap"],
        )
        actual_steps = 1
        delta = math.sqrt(
            sum(
                float((v.detach().cpu() - old).double().square().sum())
                for v, old in zip(parameters, old_parameters, strict=True)
            )
        )
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("nonfinite or zero adapter update")
        diagnostics = dict(
            objective_before=value,
            gradient_norm=norm,
            adapter_l2_delta=delta,
            nonzero_calls=len(nonzero),
            total_calls=len(calls),
            credited_tokens=count,
            positive_tokens=sum(
                len(calls[v.call_id]["output_token_ids"]) for v in nonzero if v.advantage > 0
            ),
            negative_tokens=sum(
                len(calls[v.call_id]["output_token_ids"]) for v in nonzero if v.advantage < 0
            ),
            training_eval_max_gap=max_gap,
            training_eval_mean_gap=gap_sum / count,
        )
        state = dict(
            step=plan["update"],
            sample_cursor=plan["update"],
            zero_streak=0,
            plan_sha256=p.sha(args.output / "PLAN.json"),
            update=diagnostics,
        )
        endpoint = p.rl.save_boundary(model, optimizer, args.output, state)
        p.rl.set_mode(model, training=False)
        after = {}
        with torch.no_grad():
            for credit in nonzero:
                p.rl.guard(deadline, 5)
                after[credit.call_id] = (
                    p.rl.action_logps(model, calls[credit.call_id]).flatten().cpu().tolist()
                )
        p.c.save(args.output / "AFTER_LOGPS.json", dict(logps=after, endpoint=str(endpoint)))
        diagnostics.update(
            probability_changes=probability_changes(nonzero, before, after),
            objective_after=-sum(v.advantage * sum(after[v.call_id]) for v in nonzero) / 32,
        )
        p.c.save(args.output / "UPDATE.json", diagnostics)
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
            stopped=p.rl.STOP,
            actual_new_optimizer_steps=actual_steps,
            committed_optimizer_steps=plan["update"] if endpoint else plan["update"] - 1,
            endpoint=str(endpoint) if endpoint else None,
            endpoint_usable=complete and endpoint is not None,
            update=diagnostics,
            admission=admission,
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
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
