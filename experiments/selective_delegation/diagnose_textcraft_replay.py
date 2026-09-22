"""Read-only checkpoint1 numeric diagnostic; no optimizer, tolerance change or policy promotion."""

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import signal
import time
import uuid
from pathlib import Path

import psutil
import rl_textcraft_terminal as rl

c = rl.c
SOURCE = c.ROOT / "textcraft-terminal-rl-002"
COMMIT_SHA = "063785496989e1cc16b46bd701a92b71d6c8c50aac1a5fde8c2ed27b66f518c7"
STOP = False


def cached_logps(model, call):
    """Teacher-force saved emitted IDs through a growing native KV cache, never resample."""
    import torch

    prefix = torch.tensor([call["input_token_ids"]], device=model.device)
    targets = call["output_token_ids"]
    mask = torch.ones_like(prefix)
    past, values = None, []
    for index, target in enumerate(targets):
        inputs = prefix if index == 0 else torch.tensor([[targets[index - 1]]], device=model.device)
        result = model(
            input_ids=inputs,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        values.append(torch.log_softmax(result.logits[0, -1].float() / 0.5, -1)[target].item())
        past = result.past_key_values
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
    return values


def cpu_overlap():
    """Exact-token comparison already available at cp1, no model load."""
    root = SOURCE / "batches/sample-0001"
    after_path = root / "AFTER_LOGPS.json"
    after = rl.audit.read(after_path)
    batch = rl.audit.read(root / "BATCH.json")
    index = {}
    for cid, scores in sorted(after["logps"].items()):
        path = Path(batch["data"]) / "calls" / f"{cid}.json"
        call = rl.audit.read(path)
        key = (tuple(call["input_token_ids"]), tuple(call["output_token_ids"]))
        index.setdefault(key, []).append((cid, scores))
    matches = []
    data = SOURCE / "batches/sample-0002/rollout"
    for path in sorted((data / "calls").glob("*.json")):
        call = rl.audit.read(path)
        key = (tuple(call["input_token_ids"]), tuple(call["output_token_ids"]))
        if key not in index:
            continue
        cid, full = index[key][0]
        captured = rl.audit.read(data / "generation-logps" / path.name)["logps"]
        matches.append(
            dict(
                sample2_call=path.stem,
                sample1_call=cid,
                sample1_exact_matches=[row[0] for row in index[key]],
                captured_vs_after=compare(full, captured, call["output_token_ids"]),
            )
        )
    gaps = [t["absolute_gap"] for r in matches for t in r["captured_vs_after"]["tokens"]]
    return dict(
        source_after_sha256=c.inputs.sha(after_path),
        endpoint=after["endpoint"],
        matched_sample2_calls=len(matches),
        unique_prefix_output_pairs=len({r["sample1_call"] for r in matches}),
        token_count=len(gaps),
        max_abs=max(gaps, default=None),
        mean_abs=sum(gaps) / len(gaps) if gaps else None,
        matches=matches,
        caveat="Exact input AND output IDs, same committed weights; "
        "repeated matches not independent. "
        "This subset does not identify the full sample2 failed threshold.",
    )


def select_calls(calls, episodes):
    last = [row["call_ids"][-1] for row in episodes]
    if len(last) != 32 or len(set(last)) != 32:
        raise ValueError("all32 episode-last calls required")
    remainder = sorted(
        set(calls) - set(last), key=lambda cid: (len(calls[cid]["input_token_ids"]), cid)
    )
    if len(remainder) < 12:
        raise ValueError("insufficient length-stratified calls")
    chosen = []
    for index in range(12):
        group = remainder[len(remainder) * index // 12 : len(remainder) * (index + 1) // 12]
        chosen.append(
            min(group, key=lambda cid: hashlib.sha256(("2026092251:" + cid).encode()).hexdigest())
        )
    return last + chosen


def compare(reference, replay, tokens):
    if len(reference) != len(replay) or len(tokens) != len(reference) or not tokens:
        raise ValueError("exact emitted-token length required")
    rows = [
        dict(index=i, token_id=t, reference=a, replay=b, gap=b - a, absolute_gap=abs(b - a))
        for i, (a, b, t) in enumerate(zip(reference, replay, tokens, strict=True))
    ]
    if not all(math.isfinite(row["gap"]) for row in rows):
        raise ValueError("nonfinite token logprob")
    values = sorted(row["absolute_gap"] for row in rows)
    return dict(
        max_abs=max(values),
        mean_abs=sum(values) / len(values),
        sequence_sum_gap=sum(row["gap"] for row in rows),
        absolute_quantiles={
            str(q): values[int(q * (len(values) - 1))] for q in (0.5, 0.9, 0.95, 0.99)
        },
        tokens=rows,
        top_tokens=sorted(rows, key=lambda row: -row["absolute_gap"])[:10],
    )


def aggregate(full_records):
    gaps = [
        dict(call_id=cid, **row)
        for cid, record in full_records.items()
        for row in record["original_cached_vs_full"]["tokens"]
    ]
    values = sorted(row["absolute_gap"] for row in gaps)
    maximum, mean = max(values), sum(values) / len(values)
    return dict(
        planned_calls=712,
        completed_calls=len(full_records),
        token_count=len(values),
        max_abs=maximum,
        mean_abs=mean,
        threshold_branches=dict(max_exceeds_025=maximum > 0.25, mean_exceeds_0025=mean > 0.025),
        complete=len(full_records) == 712,
        absolute_quantiles={
            str(q): values[int(q * (len(values) - 1))] for q in (0.5, 0.9, 0.95, 0.99)
        },
        top_offending_tokens=sorted(gaps, key=lambda row: -row["absolute_gap"])[:30],
    )


def cached_selection(fixed, full_records):
    remaining = sorted(
        set(full_records) - set(fixed),
        key=lambda cid: (-full_records[cid]["original_cached_vs_full"]["max_abs"], cid),
    )
    return fixed + remaining[:3]


def prepare(output):
    data = SOURCE / "batches/sample-0002/rollout"
    batch_path = SOURCE / "batches/sample-0002/BATCH.json"
    batch = rl.audit.read(batch_path)
    original = rl.audit.read(data / "PLAN.json")
    adapter = Path(original["adapter"]["path"])
    if c.inputs.sha(adapter / "COMMIT.json") != COMMIT_SHA:
        raise ValueError("exact committed first-update checkpoint required")
    commit = rl.audit.read(adapter / "COMMIT.json")
    calls = {path.stem: rl.audit.read(path) for path in (data / "calls").glob("*.json")}
    if len(calls) != 712 or len(batch["episodes"]) != 32:
        raise ValueError("exact completed sample2 inventory required")
    chosen = select_calls(calls, batch["episodes"])
    pins = {
        str(path): c.inputs.sha(path)
        for path in (batch_path, data / "PLAN.json", adapter / "COMMIT.json")
    }
    lengths = []
    for cid in sorted(calls):
        path = data / "calls" / f"{cid}.json"
        capture = data / "generation-logps" / f"{cid}.json"
        scores = rl.audit.read(capture)
        if (
            c.inputs.sha(path) != batch["native_receipt_sha256"][str(path)]
            or scores["call_sha256"] != c.inputs.sha(path)
            or len(scores["logps"]) != len(calls[cid]["output_token_ids"])
            or calls[cid]["request"]["adapter_commit_sha256"] != COMMIT_SHA
        ):
            raise ValueError("saved native call/capture/checkpoint mismatch")
        for source in (path, capture):
            pins[str(source)] = c.inputs.sha(source)
        lengths.append(
            dict(
                call_id=cid,
                input_tokens=len(calls[cid]["input_token_ids"]),
                emitted_tokens=len(calls[cid]["output_token_ids"]),
            )
        )
    plan = dict(
        schema="textcraft-checkpoint1-replay-numerics-v2",
        source=str(SOURCE),
        data=str(data),
        adapter=original["adapter"],
        model=original["model"],
        model_manifest_sha256=original["model_manifest_sha256"],
        selection="All32 last calls in planned episode order +oneSHA-selected call per12 "
        "equal-count input-length rank bins excludinglast; no rewards/gaps used.",
        selected_calls=chosen,
        full_forward_calls=sorted(calls),
        posthoc_cached_rule="Add up to3 worst maximum-gap calls outside fixed44, ties bycallID; "
        "posthoc numerical diagnosis, not performance evaluation.",
        lengths=lengths,
        seed=2026092251,
        budget_seconds=600,
        primary="Full712 full-forward first, persist aggregate threshold branches; "
        "then fixed44+up to3 worst-gap cached paths; two native regenerations optional last.",
        regenerated_calls=chosen[:2],
        checkpoint_adapter_sha256=commit["files"]["adapter_model.safetensors"],
        pins=pins,
        original_tolerances=dict(max_abs=0.25, mean_abs=0.025),
        source_sha256={
            str(Path(m.__file__)): c.inputs.sha(Path(m.__file__)) for m in (rl, c, rl.loss_math)
        },
        diagnostic_sha256=c.inputs.sha(Path(__file__)),
        no_optimizer=True,
        no_environment_actions=True,
        no_tolerance_change=True,
    )
    destination = output / "PLAN.json"
    if destination.exists():
        if rl.audit.read(destination) != plan:
            raise ValueError("immutable diagnostic selection changed")
    else:
        c.save(destination, plan)
    return plan, calls


def run(args):
    global STOP
    plan, calls = prepare(args.output)
    overlap = args.output / "CPU-EXACT-OVERLAP.json"
    if not overlap.exists():
        c.save(overlap, cpu_overlap())
    if args.prepare_only:
        print(
            json.dumps(dict(prepared=True, full_calls=712, fixed_cached_calls=44, GPU_loaded=False))
        )
        return
    if list(args.output.glob("OWNER-*.json")):
        raise ValueError("no implicit diagnostic retry")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("main must assign exactly one GPU")
    started = time.time()
    deadline = min(started + 600, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    if deadline < started + 180:
        raise ValueError("insufficient lease margin")
    lock = (c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = uuid.uuid4().hex[:12]
    c.save(
        args.output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            source=str(Path(__file__).resolve()),
        ),
    )
    model = client = None
    failure, records, full_records, selected = None, [], {}, []

    def guard(margin=100):
        if STOP or time.time() > deadline - margin:
            raise TimeoutError("bounded diagnostic incomplete; preserve recorded comparisons")

    def stop(*_):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        guard()
        torch.set_num_threads(4)
        adapter = Path(plan["adapter"]["path"])
        if c.inputs.sha(adapter / "adapter_model.safetensors") != plan["checkpoint_adapter_sha256"]:
            raise ValueError("checkpoint1 weights changed")
        tokenizer = AutoTokenizer.from_pretrained(
            c.BASE, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            c.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base,
            adapter,
            adapter_name="textcraft_action",
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
        # Match original actor's input-grad hook and subsequent eval/cache configuration.
        rl.set_mode(model, training=True)
        rl.set_mode(model, training=False)
        c.save(
            args.output / "LOAD.json",
            dict(
                training=model.training,
                use_cache=model.config.use_cache,
                gradient_checkpointing=model.is_gradient_checkpointing,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                lora_dtypes=sorted(
                    {str(p.dtype) for n, p in model.named_parameters() if "lora_" in n}
                ),
                base_dtype=str(model.get_base_model().dtype),
                optimizer_created=False,
                model_config=model.config.to_dict(),
                attention_backend=model.config._attn_implementation,
                python=platform.python_version(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(),
                transformers=__import__("transformers").__version__,
                peft=__import__("peft").__version__,
            ),
        )
        client = rl.ScoredClient(
            model,
            tokenizer,
            args.output / "regenerated",
            deadline - 30,
            plan["model_manifest_sha256"],
            c.inputs.sha(args.output / "PLAN.json"),
            adapter=plan["adapter"],
        )
        full_started = time.time()
        for cid in plan["full_forward_calls"]:
            guard(10)
            old = calls[cid]
            if client.ids(old["request"]["prompt"]) != old["input_token_ids"]:
                raise ValueError("native tokenizer input IDs differ")
            cached = rl.audit.read(Path(plan["data"]) / "generation-logps" / f"{cid}.json")["logps"]
            with torch.no_grad():
                full = rl.action_logps(model, old).flatten().cpu().tolist()
            result = dict(
                call_id=cid,
                input_tokens=len(old["input_token_ids"]),
                output_token_ids=old["output_token_ids"],
                full_forward_logps=full,
                saved_cached_logps=cached,
                original_cached_vs_full=compare(cached, full, old["output_token_ids"]),
            )
            # Persist before another model call: even regeneration failure cannot erase the gap.
            c.save(args.output / "full_forward" / f"{cid}.json", result)
            full_records[cid] = result
            if len(full_records) % 32 == 0:
                c.probe.campaign.snapshot(
                    args.output / "STATUS.json",
                    dict(phase="full_forward", completed=len(full_records), planned=712),
                )
        full_summary = aggregate(full_records)
        full_summary["elapsed_seconds"] = time.time() - full_started
        c.save(args.output / "FULL-FORWARD-SUMMARY.json", full_summary)
        selected = cached_selection(plan["selected_calls"], full_records)
        c.save(
            args.output / "CACHED-SELECTION.json",
            dict(
                fixed_calls=plan["selected_calls"],
                posthoc_added=selected[44:],
                rule=plan["posthoc_cached_rule"],
                selected_calls=selected,
            ),
        )
        for cid in selected:
            guard()
            old = calls[cid]
            result = full_records[cid]
            cached, full = result["saved_cached_logps"], result["full_forward_logps"]
            with torch.no_grad():
                forced = cached_logps(model, old)
            result["forced_cached_logps"] = forced
            result["saved_vs_forced_cached"] = compare(cached, forced, old["output_token_ids"])
            result["forced_cached_vs_full"] = compare(forced, full, old["output_token_ids"])
            c.save(args.output / "same_tokens" / f"{cid}.json", result)
            records.append(result)
            c.probe.campaign.snapshot(
                args.output / "STATUS.json",
                dict(
                    completed=len(records),
                    planned=len(selected),
                    last_call=cid,
                    phase="same_token_cached",
                ),
            )
        # Native regeneration is optional, last, and never displaces the same-token diagnosis.
        for cid in plan["regenerated_calls"]:
            if STOP or time.time() > deadline - 100:
                break
            old, result = calls[cid], full_records[cid]
            cached, full = result["saved_cached_logps"], result["full_forward_logps"]
            req = old["request"]
            spec = {
                key: value
                for key, value in req.items()
                if key
                not in (
                    "input_token_ids",
                    "model",
                    "model_manifest_sha256",
                    "adapter_enabled",
                    "adapter_sha256",
                    "context_limit",
                    "truncation",
                    "sampling",
                    "adapter_path",
                    "adapter_commit_sha256",
                )
            }
            fresh = client.call(spec)
            if not fresh["available"]:
                raise RuntimeError("native regeneration unavailable; no retry")
            scores = rl.audit.read(args.output / "regenerated/generation-logps" / f"{cid}.json")[
                "logps"
            ]
            identical = fresh["output_token_ids"] == old["output_token_ids"]
            result["regenerated_ids_identical"] = identical
            result["regenerated_output_token_ids"] = fresh["output_token_ids"]
            result["regenerated_logps"] = scores
            if identical:
                result["old_cached_vs_regenerated_cached"] = compare(
                    cached, scores, old["output_token_ids"]
                )
                result["new_cached_vs_full"] = compare(scores, full, old["output_token_ids"])
            c.save(args.output / "comparisons" / f"{cid}.json", result)
            c.probe.campaign.snapshot(
                args.output / "STATUS.json",
                dict(
                    completed=len(records),
                    planned=len(selected),
                    last_call=cid,
                    last_ids_identical=identical,
                    phase="optional_regeneration",
                ),
            )
        c.save(
            args.output / "SUMMARY.json",
            dict(
                planned_calls=len(selected),
                completed_calls=len(records),
                full_forward=full_summary,
                regenerated_calls=sum("regenerated_ids_identical" in r for r in records),
                regenerated_ids_identical=sum(
                    r.get("regenerated_ids_identical", False) for r in records
                ),
                caveat="Full712 discrepancy reconstruction, fixed44 and posthoc3 cached diagnosis. "
                "Mismatch regeneration is reported, not repaired. "
                "No optimizer or environment steps.",
            ),
        )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if full_records and len(full_records) < 712:
            c.save(args.output / "PARTIAL-FULL-FORWARD-SUMMARY.json", aggregate(full_records))
        del model, client
        gc.collect()
        torch.cuda.empty_cache()
        c.save(
            args.output / f"TERMINAL-{owner}.json",
            dict(
                failure=failure,
                full_forward_completed=len(full_records),
                full_forward_planned=712,
                completed_calls=len(records),
                planned_calls=len(selected) if selected else None,
                complete=len(full_records) == 712
                and len(records) == len(selected)
                and failure is None,
                ended=time.time(),
                elapsed_seconds=time.time() - started,
                deadline=deadline,
            ),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
