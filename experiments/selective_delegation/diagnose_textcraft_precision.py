"""FP16 own-cache/full comparison at fixed cp1; no optimizer or precision promotion."""

import argparse
import fcntl
import gc
import json
import math
import os
import platform
import signal
import time
import uuid
from pathlib import Path

import diagnose_textcraft_replay as d
import psutil

c, rl = d.c, d.rl
REFERENCE = c.ROOT / "textcraft-replay-diagnostic-002"
STOP = False


def numeric_record(cached, full, ids):
    finite = all(math.isfinite(x) for x in cached + full)
    return dict(
        output_token_ids=ids,
        cached_logps=cached,
        full_logps=full,
        all_finite=finite,
        cached_probabilities=[math.exp(x) if math.isfinite(x) else None for x in cached],
        full_probabilities=[math.exp(x) if math.isfinite(x) else None for x in full],
        own_cached_vs_full=d.compare(cached, full, ids) if finite else None,
    )


def score_fresh(model, fresh, captured, old_ids):
    full = rl.action_logps(model, fresh).flatten().cpu().tolist()
    return dict(
        **numeric_record(captured, full, fresh["output_token_ids"]),
        ids_equal_bf16=fresh["output_token_ids"] == old_ids,
    )


def prepare(output):
    source_plan = rl.audit.read(REFERENCE / "PLAN.json")
    selected = rl.audit.read(REFERENCE / "CACHED-SELECTION.json")["selected_calls"]
    terminals = list(REFERENCE.glob("TERMINAL-*.json"))
    if len(terminals) != 1 or not rl.audit.read(terminals[0])["complete"] or len(selected) != 47:
        raise ValueError("complete BF16 diagnostic002 required")
    pins = {
        str(p): c.inputs.sha(p)
        for p in (
            REFERENCE / "PLAN.json",
            REFERENCE / "CACHED-SELECTION.json",
            REFERENCE / "FULL-FORWARD-SUMMARY.json",
            REFERENCE / "LOAD.json",
            terminals[0],
        )
    }
    records = {}
    for cid in selected:
        path = REFERENCE / "same_tokens" / f"{cid}.json"
        bf16 = rl.audit.read(path)
        if bf16["saved_vs_forced_cached"]["max_abs"] != 0:
            raise ValueError("identity/path mismatch requires review, not automatic dtype probe")
        call_path = Path(source_plan["data"]) / "calls" / f"{cid}.json"
        if c.inputs.sha(call_path) != source_plan["pins"][str(call_path)]:
            raise ValueError("saved native call changed")
        records[cid] = rl.audit.read(call_path)
        for p in (path, call_path):
            pins[str(p)] = c.inputs.sha(p)
    plan = dict(
        schema="textcraft-fp16-own-cache-diagnostic-v1",
        reference=str(REFERENCE),
        adapter=source_plan["adapter"],
        checkpoint_adapter_sha256=source_plan["checkpoint_adapter_sha256"],
        model=source_plan["model"],
        model_manifest_sha256=source_plan["model_manifest_sha256"],
        selected_calls=selected,
        regenerated_calls=source_plan["regenerated_calls"],
        selection="Unchanged002 fixed44 plus same3 posthoc numerical offenders; "
        "not performance evaluation",
        base_dtype="torch.float16",
        lora_dtype="torch.float32",
        attention_backend="sdpa",
        temperature=0.5,
        top_p=1.0,
        top_k=0,
        max_time=90,
        budget_seconds=600,
        pins=pins,
        no_optimizer=True,
        no_tolerance_change=True,
        no_promotion=True,
        source_sha256={str(Path(m.__file__)): c.inputs.sha(Path(m.__file__)) for m in (d, rl, c)},
        diagnostic_sha256=c.inputs.sha(Path(__file__)),
        caveat="FP16 changes policy arithmetic; old BF16 outputs are fixed conditioning inputs. "
        "Fresh FP16 generation is scored against its OWN full-forward emitted IDs.",
    )
    destination = output / "PLAN.json"
    if destination.exists():
        if rl.audit.read(destination) != plan:
            raise ValueError("immutable precision diagnostic changed")
    else:
        c.save(destination, plan)
    return plan, records


def run(args):
    global STOP
    plan, calls = prepare(args.output)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True, fixed_calls=47, fresh_calls=2, GPU_loaded=False)))
        return
    if list(args.output.glob("OWNER-*.json")):
        raise ValueError("no implicit retry")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one assigned GPU required")
    started = time.time()
    deadline = min(started + 600, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    if deadline < started + 180:
        raise ValueError("lease margin insufficient")
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
    model = client = base = None
    failure, fixed, fresh_results = None, [], []

    def stop(*_):
        global STOP
        STOP = True

    def guard():
        if STOP or time.time() > deadline - 100:
            raise TimeoutError("bounded diagnostic incomplete; saved results retained")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        guard()
        torch.set_num_threads(4)
        adapter = Path(plan["adapter"]["path"])
        if (
            c.inputs.sha(adapter / "COMMIT.json") != d.COMMIT_SHA
            or c.inputs.sha(adapter / "adapter_model.safetensors")
            != plan["checkpoint_adapter_sha256"]
        ):
            raise ValueError("exact cp1 required")
        tokenizer = AutoTokenizer.from_pretrained(
            c.BASE, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            c.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
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
                model_config=model.config.to_dict(),
                attention_backend=model.config._attn_implementation,
                python=platform.python_version(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(),
                transformers=__import__("transformers").__version__,
                peft=__import__("peft").__version__,
                optimizer_created=False,
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
        for cid in plan["selected_calls"]:
            guard()
            call = calls[cid]
            if client.ids(call["request"]["prompt"]) != call["input_token_ids"]:
                raise ValueError("native input IDs changed")
            with torch.no_grad():
                full = rl.action_logps(model, call).flatten().cpu().tolist()
            c.save(
                args.output / "full_forward" / f"{cid}.json",
                dict(
                    logps=full,
                    all_finite=all(math.isfinite(x) for x in full),
                    output_token_ids=call["output_token_ids"],
                ),
            )
            with torch.no_grad():
                cached = d.cached_logps(model, call)
            result = dict(call_id=cid, **numeric_record(cached, full, call["output_token_ids"]))
            bf16 = rl.audit.read(REFERENCE / "same_tokens" / f"{cid}.json")
            result["reference_bf16_own_cached_vs_full"] = bf16["forced_cached_vs_full"]
            c.save(args.output / "same_tokens" / f"{cid}.json", result)
            fixed.append(result)
            c.probe.campaign.snapshot(
                args.output / "STATUS.json",
                dict(fixed_completed=len(fixed), planned=47, last_call=cid),
            )
            if not result["all_finite"]:
                raise ValueError("nonfinite FP16 arithmetic recorded; no silent repair")
        for cid in plan["regenerated_calls"]:
            guard()
            old = calls[cid]
            spec = {
                k: v
                for k, v in old["request"].items()
                if k
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
            spec.update(
                base_dtype="torch.float16", lora_dtype="torch.float32", numeric_diagnostic=True
            )
            fresh = client.call(spec)
            if not fresh["available"]:
                raise RuntimeError("fresh native call unavailable; no retry")
            captured = rl.audit.read(args.output / "regenerated/generation-logps" / f"{cid}.json")
            with torch.no_grad():
                result = score_fresh(model, fresh, captured["logps"], old["output_token_ids"])
            c.save(args.output / "fresh_comparisons" / f"{cid}.json", result)
            fresh_results.append(result)
            if not result["all_finite"]:
                raise ValueError("nonfinite fresh FP16 arithmetic recorded")
        gaps = [t["absolute_gap"] for r in fixed for t in r["own_cached_vs_full"]["tokens"]]
        old_gaps = [
            t["absolute_gap"]
            for r in fixed
            for t in r["reference_bf16_own_cached_vs_full"]["tokens"]
        ]
        c.save(
            args.output / "SUMMARY.json",
            dict(
                fixed_completed=len(fixed),
                fresh_completed=2,
                token_count=len(gaps),
                fp16_max=max(gaps),
                fp16_mean=sum(gaps) / len(gaps),
                bf16_same_subset_max=max(old_gaps),
                bf16_same_subset_mean=sum(old_gaps) / len(old_gaps),
                all_finite=True,
                no_training=True,
                caveat="Numeric fixed-prefix comparison, not performance/promotion; "
                "subset not full712.",
            ),
        )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model, client, base
        gc.collect()
        torch.cuda.empty_cache()
        c.save(
            args.output / f"TERMINAL-{owner}.json",
            dict(
                failure=failure,
                fixed_completed=len(fixed),
                fresh_completed=len(fresh_results),
                complete=len(fixed) == 47 and len(fresh_results) == 2 and failure is None,
                started=started,
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
