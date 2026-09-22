"""Conditional two-update native TextCraft terminal-RLOO pilot; no launch authority.

Reuses all complete063 trajectories under exact056 for batch1. Later batches are
fresh, frozen-policy native episodes. No retries, reward filtering or optimizer
continuation from SFT. A failed/incomplete batch never becomes a successful endpoint.
"""

import argparse
import fcntl
import gc
import hashlib
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

import analyze_textcraft as audit
import analyze_textcraft_train_readiness as readiness
import eval_textcraft_train_readiness as reader
import textcraft_trajectory_loss as loss_math
import train_planner as checkpoints

c = reader.c
SEED = 2026092220
STOP = False


def guard(deadline, reserve=0):
    if STOP or time.time() >= deadline - reserve:
        raise TimeoutError("bounded terminal-RLOO deadline; incomplete work is unknown")


def checked_optimizer_step(
    optimizer,
    batch_dir,
    *,
    previous_step,
    maximum,
    mean,
    count,
    max_tolerance,
    mean_tolerance,
):
    """Persist the existing replay comparison before permitting any weight update."""
    passed = (
        count > 0
        and math.isfinite(maximum)
        and math.isfinite(mean)
        and maximum <= max_tolerance
        and mean <= mean_tolerance
    )
    c.save(
        batch_dir / "TRAIN-EVAL-REPLAY.json",
        dict(
            max_abs=maximum,
            mean_abs=mean,
            token_count=count,
            max_tolerance=max_tolerance,
            mean_tolerance=mean_tolerance,
            passed=passed,
            scope="Nonzero-advantage action tokens from the actual gradient pass",
        ),
    )
    if not passed:
        raise ValueError("train/eval replay discrepancy exceeded declared tolerance")
    c.save(batch_dir / "OPTIMIZER-STEP-STARTED.json", dict(previous_step=previous_step))
    optimizer.step()


def set_mode(model, *, training):
    """Same actor; only FP32 LoRA may train, every parameter frozen for rollout."""
    import torch

    params = []
    for name, parameter in model.named_parameters():
        is_lora = "lora_" in name
        if is_lora:
            if parameter.dtype != torch.float32:
                raise ValueError("FP32 LoRA required")
            params.append(parameter)
        parameter.requires_grad_(training and is_lora)
    if not params:
        raise ValueError("no LoRA actor parameters")
    model.train(training)
    model.config.use_cache = not training
    if training:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
    else:
        model.gradient_checkpointing_disable()
    return params


class GenerationCapture:
    """Explicit local model view, not a process-global patch or alternate policy."""

    def __init__(self, model):
        self.model = model
        self.logps = self.tokens = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, **kwargs):
        import torch

        self.logps = self.tokens = None
        result = self.model.generate(**kwargs, return_dict_in_generate=True, output_scores=True)
        prefix = kwargs["input_ids"].shape[1]
        self.tokens = result.sequences[0, prefix:].detach().cpu().tolist()
        self.logps = [
            float(torch.log_softmax(score.float(), -1)[0, token])
            for score, token in zip(result.scores, self.tokens, strict=True)
        ]
        return result.sequences


class ScoredClient(c.NativeClient):
    """Unchanged native call and receipt, plus an immutable generation-score sidecar."""

    def __init__(self, model, *args, **kwargs):
        self.capture = GenerationCapture(model)
        super().__init__(self.capture, *args, **kwargs)

    def call(self, spec):
        record = super().call(spec)
        if record["available"]:
            if self.capture.tokens != record["output_token_ids"]:
                raise ValueError("generation-score/output identity mismatch")
            c.save(
                self.output / "generation-logps" / (spec["call_id"] + ".json"),
                dict(
                    call_sha256=c.inputs.sha(self.output / "calls" / (spec["call_id"] + ".json")),
                    logps=self.capture.logps,
                ),
            )
        return record


def action_logps(model, record):
    import torch

    ids, target = loss_math.causal_inputs(record)
    logits = model(
        input_ids=torch.tensor([ids], device=model.device),
        use_cache=False,
        logits_to_keep=len(target),
    ).logits
    logps = torch.log_softmax(logits.float() / 0.5, dim=-1)
    return logps.gather(-1, torch.tensor([target], device=model.device).unsqueeze(-1)).squeeze(-1)


def advance(state, *, updated):
    return {
        **state,
        "sample_cursor": state["sample_cursor"] + 1,
        "step": state["step"] + int(updated),
        "zero_streak": 0 if updated else state["zero_streak"] + 1,
    }


def finished(state):
    return state["step"] >= 2 or state["sample_cursor"] >= 4 or state["zero_streak"] >= 2


def save_boundary(model, optimizer, output, state):
    actual_steps = {int(v["step"]) for v in optimizer.state.values() if "step" in v}
    if (actual_steps or {0}) != {state["step"]}:
        raise ValueError("Adam step differs from claimed committed boundary")

    class RootAdapterFiles:
        def save_pretrained(self, directory, **kwargs):
            model.save_pretrained(directory, selected_adapters=["textcraft_action"], **kwargs)
            # PEFT's named adapter writes a subdirectory. Flatten only this freshly
            # created checkpoint so existing root-level COMMIT/loader contracts apply.
            named = directory / "textcraft_action"
            if named.exists():
                for path in named.iterdir():
                    if not path.is_file() or (directory / path.name).exists():
                        raise ValueError("unexpected named-adapter checkpoint layout")
                    path.rename(directory / path.name)
                named.rmdir()

    directory = output / "boundaries" / f"sample-{state['sample_cursor']:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoints.save_checkpoint(RootAdapterFiles(), optimizer, directory, state)
    c.save(
        directory / "BOUNDARY.json",
        dict(
            checkpoint=str(checkpoint),
            state=state,
            commit_sha256=c.inputs.sha(checkpoint / "COMMIT.json"),
        ),
    )
    return checkpoint


def binding(checkpoint):
    commit = audit.read(checkpoint / "COMMIT.json")
    return dict(
        path=str(checkpoint),
        sha256=commit["files"]["adapter_model.safetensors"],
        commit_sha256=c.inputs.sha(checkpoint / "COMMIT.json"),
    )


def prepare(args):
    if getattr(args, "fp16_continuation_amendment", None):
        import textcraft_fp16_continuation

        return textcraft_fp16_continuation.prepare(args)
    if args.root.resolve() != c.ROOT or not 0 < args.hours <= 3:
        raise ValueError("fixed campaign and at most3hours required")
    source = args.root / "textcraft-train-readiness-001"
    if c.inputs.sha(source / "PLAN.json") != readiness.PLAN_SHA:
        raise ValueError("fixed063 planned batch required")
    original = audit.read(source / "PLAN.json")
    warm = reader.shared.endpoint(
        Path(original["adapter"]["path"]),
        training_plan_sha256=reader.public.PLAN_SHA,
        rows_sha256=reader.public.ROWS_SHA,
    )
    if warm["sha256"] != original["adapter"]["sha256"]:
        raise ValueError("exact public056 warm weights changed")
    modules = (
        reader,
        reader.shared,
        reader.public,
        c,
        c.bridge,
        c.inputs,
        c.probe,
        c.probe.runtime,
        c.probe.campaign,
        audit,
        readiness,
        loss_math,
        checkpoints,
    )
    paths = [Path(__file__).resolve(), *(Path(m.__file__).resolve() for m in modules)]
    plan = dict(
        schema="textcraft-terminal-rloo-two-update-v1",
        status="conditional_not_gpu_accepted",
        root=str(args.root.resolve()),
        warm=warm,
        tasks_prepared=original["prepared"],
        tasks_sha256=original["tasks_sha256"],
        manifest_sha256=original["manifest_sha256"],
        runtime_task_order=dict(
            source=str(c.ROOT / readiness.reader.SOURCE_TASKS),
            sha256=readiness.reader.SOURCE_TASKS_SHA,
            rule="Recover original063 dictionary order from hash-bound055 source; "
            "assert exact semantic equality with prepared tasks; preserve for every batch",
        ),
        model=original["model"],
        model_manifest_sha256=original["model_manifest_sha256"],
        planned_tasks=8,
        episodes_per_batch=32,
        maximum_optimizer_updates=2,
        maximum_sampled_batches=4,
        maximum_consecutive_flat_batches=2,
        budget_seconds=args.hours * 3600,
        seed=SEED,
        learning_rate=2e-5,
        weight_decay=0,
        clip_grad_norm=1.0,
        optimizer="fresh_AdamW",
        objective="-sum(full-task-group RLOO advantage * ALL emitted actor token logps)/32",
        temperature=0.5,
        per_call_cap=256,
        context=8192,
        episode_calls=96,
        episode_output_tokens=8192,
        generation_replay_max_abs_tolerance=0.25,
        generation_replay_mean_abs_tolerance=0.025,
        first_batch=dict(
            output=str(source),
            plan_sha256=readiness.PLAN_SHA,
            all_32_required=True,
            readiness_conditioned=True,
            generation_logps="not_recorded; recompute_under_exact056",
        ),
        fresh_jobs={
            str(batch): [dict(j, seed=SEED + batch * 10 + j["repeat"]) for j in original["jobs"]]
            for batch in (2, 3, 4)
        },
        maximum_new_scientific_episode_calls=3 * 32 * 96,
        maximum_qualification_calls=1,
        source_sha256={str(p): c.inputs.sha(p) for p in paths},
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            versions={
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "safetensors")
            },
        ),
        admission="Complete terminal063 independently native-regraded; at least one mixed task. "
        "Otherwise explicit CPU conditional skip, no model loading. No selective task filtering.",
        caveat="TRAIN readiness-conditioned exploratory policy gradient, not held-out gain. "
        "Fresh policy after each update; invalid-action paths retain terminal reward. "
        "Exceptions/incomplete batch halt without update. No SFT-control or evaluation inline.",
    )
    path = args.output / "PLAN.json"
    if path.exists():
        if audit.read(path) != plan:
            raise ValueError("immutable terminal-RLOO PLAN changed")
    else:
        c.save(path, plan)
    return plan


def native_batch(directory, plan, tokenizer, tasks, world):
    """Reuse existing request and native environment replay, without a fake dead owner."""
    plan_sha = c.inputs.sha(directory / "PLAN.json")
    rows = {p.stem: audit.read(p) for p in (directory / "episodes").glob("*.json")}
    calls = {p.stem: audit.read(p) for p in (directory / "calls").glob("*.json")}
    starts = {p.stem: audit.read(p) for p in (directory / "starts").glob("*.json")}
    if set(rows) != {j["episode_id"] for j in plan["jobs"]} or set(calls) != set(starts):
        raise ValueError("incomplete native32batch or unresolved native starts")
    checked = {}
    for job in plan["jobs"]:
        row = rows[job["episode_id"]]
        for index, cid in enumerate(row["call_ids"]):
            call, req = calls[cid], calls[cid]["request"]
            if (
                starts[cid]["request"] != req
                or starts[cid]["request_digest"] != call["request_digest"]
            ):
                raise ValueError("native start/call mismatch")
            spec = dict(
                call_id=f"{job['episode_id']}-c{index:03d}",
                prompt=req["prompt"],
                policy="flat",
                role="root",
                depth=0,
                node_id="n0",
                parent_node_id=None,
                episode_id=job["episode_id"],
                task_id=job["task_id"],
                global_call_index=index,
                cap=req["cap"],
                seed=job["seed"]
                + int(hashlib.sha256(job["task_id"].encode()).hexdigest()[:8], 16)
                + 1000 * index,
                condition=job["condition"],
                prompt_profile="original",
            )
            audit.audit_call(call, spec, plan, plan_sha, tokenizer)
        nodes = {
            nid: audit.read(directory / "nodes" / f"{job['episode_id']}-{nid}.json")
            for nid in row["node_ids"]
        }
        checked[job["episode_id"]] = audit.audit_episode(
            tasks[job["task_id"]], job, row, calls, nodes, plan, plan_sha, tokenizer, world
        )
    episodes = [rows[j["episode_id"]] for j in plan["jobs"]]
    credits = loss_math.batch_credits(episodes, calls)
    return episodes, calls, credits, checked


def run(args):
    global STOP
    plan = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True, GPU_loaded=False, planned_tasks=8, maximum_updates=2)))
        return
    output = args.output.resolve()
    if list(output.glob("OWNER-*.json")) or (output / "ADMISSION.json").exists():
        raise ValueError("no implicit retry/resume of prior attempt")
    source = Path(plan["first_batch"]["output"])
    continuation = "restore_checkpoint" in plan
    if continuation:
        import textcraft_fp16_continuation as fp16

        admission = dict(
            explicit_continuation=True,
            ancestor=plan["restore_checkpoint"],
            no_historical_batch_training_reuse=True,
        )
    else:
        admission = readiness.analyze(source)  # Requires actual source terminal/dead owner.
    c.save(output / "ADMISSION.json", admission)
    if not continuation and any(g["observed"] != 4 for g in admission["groups"].values()):
        raise ValueError("incomplete063 cannot become training zeros")
    if not continuation and not admission["classification_counts"].get("mixed", 0):
        c.save(
            output / "CONDITIONAL-SKIP.json",
            dict(
                reason="no_mixed063_task",
                GPU_loaded=False,
                actual_optimizer_steps=0,
                endpoint=None,
                scientific_training=False,
            ),
        )
        return
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + plan["budget_seconds"], lease - 600)
    if deadline < time.time() + 300:
        raise ValueError("insufficient allocation margin")
    lock = (c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner, started = uuid.uuid4().hex[:12], time.time()
    c.save(
        output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            started=started,
            deadline=deadline,
            allocation_end=lease,
            source=str(Path(__file__).resolve()),
        ),
    )
    model = optimizer = None
    failure, complete, endpoint, actual_steps = None, False, None, 0
    state = dict(
        step=0, sample_cursor=0, zero_streak=0, plan_sha256=c.inputs.sha(output / "PLAN.json")
    )
    if continuation:
        state = {
            **plan["restore_state"],
            "plan_sha256": c.inputs.sha(output / "PLAN.json"),
            "ancestor_commit_sha256": plan["ancestor_receipt_sha256"][
                str(Path(plan["restore_checkpoint"]) / "COMMIT.json")
            ],
        }
        actual_steps = 1

    def stop(*_):
        global STOP
        STOP = True
        c.STOP = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        tokenizer = AutoTokenizer.from_pretrained(
            c.BASE, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            c.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16 if continuation else torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base,
            plan["restore_checkpoint"] if continuation else plan["warm"]["path"],
            adapter_name="textcraft_action",
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
        parameters = set_mode(model, training=True)
        optimizer = (
            fp16.restore_optimizer_rng(parameters, Path(plan["restore_checkpoint"]))
            if continuation
            else torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0)
        )
        if continuation:
            fp16.pre_rollout_probe(model, parameters, plan, output)
        endpoint = save_boundary(model, optimizer, output, state)
        set_mode(model, training=False)
        c.save(
            output / "LOAD.json",
            dict(
                warm=plan["warm"],
                fresh_adam=not continuation,
                restored_optimizer_rng=continuation,
                restored_checkpoint=plan.get("restore_checkpoint"),
                base_dtype=str(model.get_base_model().dtype),
                lora_dtypes=sorted(
                    {str(p.dtype) for n, p in model.named_parameters() if "lora_" in n}
                ),
                model_config=model.config.to_dict(),
                attention_backend=model.config._attn_implementation,
                base_frozen=True,
                trainable_during_collection=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda,
            ),
        )
        original_plan = audit.read(source / "PLAN.json")
        tasks = {t["id"]: t for t in readiness.runtime_tasks(original_plan)}
        world = c.bridge.load_world()
        while not finished(state) and state["sample_cursor"] < plan["maximum_sampled_batches"]:
            guard(deadline, 300)
            sample = state["sample_cursor"] + 1
            batch_dir = output / "batches" / f"sample-{sample:04d}"
            batch_dir.mkdir(parents=True, exist_ok=False)
            if sample == 1:
                data_dir, batch_plan = source, original_plan
            else:
                data_dir = batch_dir / "rollout"
                batch_plan = {
                    **original_plan,
                    "jobs": plan["fresh_jobs"][str(sample)],
                    "adapter": binding(endpoint),
                    "source_sha256": plan["source_sha256"],
                }
                if continuation:
                    batch_plan["numeric_precision"] = dict(
                        base="float16", lora="float32", explicit_continuation=True
                    )
                c.save(data_dir / "PLAN.json", batch_plan)
                client = ScoredClient(
                    model,
                    tokenizer,
                    data_dir,
                    deadline,
                    plan["model_manifest_sha256"],
                    c.inputs.sha(data_dir / "PLAN.json"),
                    adapter=batch_plan["adapter"],
                )
                for job in batch_plan["jobs"]:
                    guard(deadline, 180)
                    row = c.episode(tasks[job["task_id"]], job, client, world, data_dir, deadline)
                    if not row["observed"]:
                        raise ValueError("incomplete native episode; no implicit retry/update")
                del client
            episodes, calls, credits, native = native_batch(
                data_dir, batch_plan, tokenizer, tasks, world
            )
            c.save(
                batch_dir / "BATCH.json",
                dict(
                    data=str(data_dir),
                    reused=(sample == 1),
                    episodes=episodes,
                    audits=native,
                    costs=c.cost(list(calls.values())),
                    native_receipt_sha256={
                        str(p): c.inputs.sha(p)
                        for folder in ("calls", "starts", "nodes", "episodes")
                        for p in (data_dir / folder).glob("*.json")
                    },
                ),
            )
            if sample == 1:
                # One counted fresh native replay qualifies actual generation at the reused policy.
                old = calls[credits[0].call_id]
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
                client = ScoredClient(
                    model,
                    tokenizer,
                    output / "qualification",
                    deadline,
                    plan["model_manifest_sha256"],
                    state["plan_sha256"],
                    adapter=plan["warm"],
                )
                replay = client.call(spec)
                if not replay["available"] or replay["output_token_ids"] != old["output_token_ids"]:
                    raise ValueError("same056 native generation failed exact first-call replay")
                del client
            before, differences, captured = {}, [], {}
            with torch.no_grad():
                for cid, call in calls.items():
                    guard(deadline, 180)
                    before[cid] = action_logps(model, call).flatten().cpu().tolist()
                    capture_path = data_dir / "generation-logps" / (cid + ".json")
                    if sample == 1 and cid == credits[0].call_id:
                        capture_path = output / "qualification/generation-logps" / (cid + ".json")
                    if capture_path.exists():
                        scores = audit.read(capture_path)["logps"]
                        captured[cid] = scores
                        differences.extend(
                            abs(a - b) for a, b in zip(scores, before[cid], strict=True)
                        )
            if continuation:
                fp16.record_before(batch_dir, before, captured)
            else:
                c.save(
                    batch_dir / "BEFORE_LOGPS.json",
                    dict(
                        logps=before,
                        provenance="Recomputed under unchanged sampled weights; "
                        "063 historical generation scores absent",
                        generation_replay_max_abs=max(differences) if differences else None,
                        generation_replay_mean_abs=sum(differences) / len(differences)
                        if differences
                        else None,
                    ),
                )
                if (
                    not differences
                    or not all(math.isfinite(x) for x in differences)
                    or max(differences) > 0.25
                    or sum(differences) / len(differences) > 0.025
                ):
                    raise ValueError("generation/replay discrepancy exceeded declared tolerance")
            set_mode(model, training=True)
            optimizer.zero_grad(set_to_none=True)
            longest = max(
                calls.values(), key=lambda r: len(r["input_token_ids"]) + len(r["output_token_ids"])
            )
            guard(deadline, 150)
            loss_math.replay_action_loss(model, longest, 1.0).backward()
            if any(p.grad is not None for n, p in model.named_parameters() if "lora_" not in n):
                raise ValueError("base received gradient")
            c.save(
                batch_dir / "BACKWARD-QUALIFICATION.json",
                dict(
                    call_id=longest["call_id"],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    base_gradient=False,
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            nonzero = [credit for credit in credits if credit.advantage != 0]
            update = dict(optimizer_called=False, nonzero_action_calls=len(nonzero))
            if nonzero:
                old_parameters = [p.detach().cpu().clone() for p in parameters]
                value, replay_gap, replay_sum, replay_count = 0.0, 0.0, 0.0, 0
                for credit in nonzero:
                    guard(deadline, 120)
                    logps = action_logps(model, calls[credit.call_id])
                    gaps = [
                        abs(a - b)
                        for a, b in zip(
                            logps.detach().flatten().cpu().tolist(),
                            before[credit.call_id],
                            strict=True,
                        )
                    ]
                    replay_gap = max(replay_gap, max(gaps))
                    replay_sum += sum(gaps)
                    replay_count += len(gaps)
                    objective = -credit.advantage * logps.sum() / 32
                    if not torch.isfinite(objective):
                        raise ValueError("nonfinite trajectory objective")
                    value += float(objective.detach())
                    objective.backward()
                norm = float(
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                )
                if not norm > 0:
                    raise ValueError("nonzero advantages yielded zero gradient")
                guard(deadline, 90)
                checked_optimizer_step(
                    optimizer,
                    batch_dir,
                    previous_step=state["step"],
                    maximum=replay_gap,
                    mean=replay_sum / replay_count,
                    count=replay_count,
                    max_tolerance=plan["generation_replay_max_abs_tolerance"],
                    mean_tolerance=plan["generation_replay_mean_abs_tolerance"],
                )
                actual_steps += 1
                delta = math.sqrt(
                    sum(
                        float((p.detach().cpu() - old).double().square().sum())
                        for p, old in zip(parameters, old_parameters, strict=True)
                    )
                )
                if not math.isfinite(delta) or delta <= 0:
                    raise ValueError("nonfinite/zero adapter update")
                update.update(
                    optimizer_called=True,
                    objective=value,
                    gradient_norm=norm,
                    adapter_l2_delta=delta,
                    train_eval_replay_max_abs=replay_gap,
                    train_eval_replay_mean_abs=replay_sum / replay_count,
                    train_eval_replay_token_count=replay_count,
                )
            next_state = {
                **advance(state, updated=bool(nonzero)),
                "batch_sha256": c.inputs.sha(batch_dir / "BATCH.json"),
                "update": update,
            }
            endpoint = save_boundary(model, optimizer, output, next_state)
            state = next_state  # A failed write must not claim a committed sampled boundary.
            set_mode(model, training=False)
            after = {}
            with torch.no_grad():
                for credit in nonzero:
                    guard(deadline, 5)
                    after[credit.call_id] = (
                        action_logps(model, calls[credit.call_id]).flatten().cpu().tolist()
                    )
            c.save(batch_dir / "AFTER_LOGPS.json", dict(logps=after, endpoint=str(endpoint)))
            c.probe.campaign.snapshot(output / "STATUS.json", state)
        complete = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        physical_paths = [
            *output.glob("batches/sample-*/rollout/calls/*.json"),
            *output.glob("qualification/calls/*.json"),
        ]
        c.save(
            output / "SUMMARY.json",
            dict(
                complete=complete,
                actual_optimizer_steps=actual_steps,
                new_optimizer_steps=actual_steps - (1 if continuation else 0),
                cumulative_optimizer_steps=actual_steps,
                new_sampled_batches=state["sample_cursor"] - (1 if continuation else 0),
                committed_optimizer_steps=state["step"],
                committed_sampled_batches=state["sample_cursor"],
                new_physical_cost=c.cost([audit.read(p) for p in physical_paths]),
                reused063_cost=None if continuation else admission["native_audit"]["physical_cost"],
                reused063_is_not_new_compute=True,
                failure=failure,
            ),
        )
        c.save(
            output / f"TERMINAL-{owner}.json",
            dict(
                complete=complete,
                failure=failure,
                stopped=STOP,
                ended=time.time(),
                elapsed_seconds=time.time() - started,
                actual_optimizer_steps=actual_steps,
                new_optimizer_steps=actual_steps - (1 if continuation else 0),
                cumulative_optimizer_steps=actual_steps,
                new_sampled_batches=state["sample_cursor"] - (1 if continuation else 0),
                committed_optimizer_steps=state["step"],
                committed_sampled_batches=state["sample_cursor"],
                endpoint=str(endpoint) if endpoint else None,
                endpoint_usable=complete
                and failure is None
                and (not continuation or state["step"] == 2),
                target_additional_update_achieved=not continuation or state["step"] == 2,
                deadline=deadline,
            ),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=c.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fp16-continuation-amendment", type=Path)
    run(parser.parse_args())
