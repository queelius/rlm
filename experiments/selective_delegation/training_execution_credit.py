"""Conditional TRAIN execution-credit screen: saved plans, empty-report finals only."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import signal
import time
import uuid
from pathlib import Path

import eval_planner as evaluation
import frozen_execution_probe as frozen
import plan_only_probe as runtime
import probe
import rl_planner

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")


def validate_source_call(call, prompt, role, seed, cap, adapter_sha, model, kind):
    request = call["request"]
    sampling = request if kind == "rl" else request["sampling"]
    expected = dict(
        temperature=0.5,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=cap,
        max_time=90.0,
    )
    if (
        request["prompt"] != prompt
        or request["role"] != role
        or request["seed"] != seed
        or request["model"] != model
        or request["adapter_sha256"] != adapter_sha
        or request["adapter_enabled"] != (adapter_sha is not None)
        or any(sampling.get(k) != v for k, v in expected.items())
        or sampling.get("do_sample", True) is not True
        or probe.runtime.digest(request) != call["request_digest"]
        or request["input_token_ids"] != call["input_token_ids"]
        or call["usage"]["prompt_tokens"] != len(call["input_token_ids"])
        or (
            call["available"]
            and call["usage"]["completion_tokens"] != len(call["output_token_ids"])
        )
    ):
        raise ValueError("source native request/receipt differs")


def reconstruct_source(episode, calls, case, source_plan, parent_index, setting):
    """Replay public prompt construction using actual earlier answers, never host answers."""
    kind = "rl" if setting == 0 else "frozen"
    if setting == 0:
        seeds = rl_planner.seed_schedule(1, parent_index, episode["candidate"])
        if episode["seeds"] != seeds:
            raise ValueError("source RL seed schedule differs")
        seed, final_seed = seeds["downstream"], seeds["downstream"] + 2
    else:
        seed = frozen.execution_seed(parent_index, setting - 1)
        final_seed = seed + 100
        if episode["seed"] != seed:
            raise ValueError("source frozen seed schedule differs")
    answers, trace = [], []
    plan = episode["plan"]
    for index, question in enumerate(plan["subquestions"]):
        try:
            resolved = evaluation.bind_question(question, answers)
        except ValueError:
            break
        cid = episode["episode_id"] + f"-helper-{index + 1}"
        if cid not in calls:
            raise ValueError("source helper missing without dependency failure")
        call = calls[cid]
        validate_source_call(
            call,
            evaluation.isolated_helper_prompt(case, resolved),
            "helper",
            seed + index + 1,
            384 // len(plan["subquestions"]),
            source_plan["helper_contract"]["adapter_binding"]["adapter_model.safetensors"],
            source_plan["model"],
            kind,
        )
        if not call["available"]:
            raise ValueError("source helper transport unavailable")
        try:
            answer = evaluation.parse_helper_answer(call["text"])
        except ValueError:
            break
        answers.append(answer)
        trace.append(
            dict(step=index + 1, question=question, resolved_question=resolved, answer=answer)
        )
    prompt = evaluation.final_prompt(case, plan, {"execution": "isolated", "steps": trace})
    final = calls.get(episode["episode_id"] + "-final")
    if final:
        if len(trace) != len(plan["subquestions"]):
            raise ValueError("source final after incomplete helper trace")
        validate_source_call(
            final, prompt, "final", final_seed, 128, None, source_plan["model"], kind
        )
    elif episode["status"] not in ("invalid_dependency", "invalid_helper"):
        raise ValueError("source final absent without declared protocol failure")
    return final, prompt, final_seed


def prepare(root, output, hours):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not 0 < hours <= 1 / 3 + 1e-9 or output == root:
        raise ValueError("separate output and maximum 20 minutes required")
    cases_path = root / "inputs-001/cases.jsonl"
    source = root / "rl-fullpass-001/batch-0001"
    contract = frozen.prepare(source, cases_path)
    hashes, audited = {}, {}
    for name in ("analysis-rl-fullpass-001.json", "analysis-frozen-execution-001.json"):
        path = root / name
        hashes[str(path)] = probe.campaign.sha(path)
        audited.update(runtime.read(path)["input_source_receipt_sha256"])

    def track(path):
        path = Path(path).resolve()
        digest = probe.campaign.sha(path)
        if audited.get(str(path)) != digest:
            raise ValueError("source differs from completed audit: " + str(path))
        hashes[str(path)] = digest
        return runtime.read(path)

    source_plan = track(source.parent / "PLAN.json")
    frozen_plan = track(root / "frozen-execution-001/PLAN.json")
    if any(frozen_plan[k] != v for k, v in contract.items()):
        raise ValueError("frozen source contract changed")
    cases = {c["id"]: c for c in map(json.loads, cases_path.read_text().splitlines())}
    hashes[str(cases_path)] = probe.campaign.sha(cases_path)
    controls = sorted(contract["case_ids"])[:2]
    jobs, historical = [], {}
    for slot in contract["slots"]:
        parent, candidate = slot["case_id"], slot["candidate"]
        root_call = track(source / "calls" / f"u01-{parent}-c{candidate}-root.json")
        for setting in range(5):
            directory = source if setting == 0 else root / "frozen-execution-001"
            identity = (
                f"u01-{parent}-c{candidate}"
                if setting == 0
                else f"{parent}-c{candidate}-e{setting - 1}"
            )
            episode = track(directory / "episodes" / (identity + ".json"))
            if episode["plan"] != slot["plan"] or not episode["available"]:
                raise ValueError("source plan/availability differs")
            calls = {
                cid: track(directory / "calls" / (cid + ".json")) for cid in episode["call_ids"]
            }
            historical.update(
                {str(directory / "calls" / (cid + ".json")): c for cid, c in calls.items()}
            )
            final, prompt, seed = reconstruct_source(
                episode, calls, cases[parent], source_plan, slot["parent_index"], setting
            )
            grade = probe.grade(final["text"] if final else "", cases[parent])
            if episode["reward"] != int(grade["correct"]) or (final and not final["available"]):
                raise ValueError("source official reward differs or final unavailable")
            jobs.append(
                dict(
                    case_id=parent,
                    repeat=setting,
                    policy=f"c{candidate}",
                    source_episode_id=identity,
                    seed=seed,
                    root=root_call,
                    factual=final,
                    old={**grade, "observed": True, "status": episode["status"]},
                    factual_prompt=prompt,
                    plan_only_prompt=runtime.plan_only_prompt(cases[parent], slot["plan"]),
                    generated_plan=slot["plan"],
                    control=parent in controls and candidate == 0,
                )
            )
    if len(jobs) != 320 or sum(j["old"]["correct"] for j in jobs if j["repeat"] == 0) != 45:
        raise ValueError("expected 320 slots and original reward 45/64")
    if sum(j["control"] and j["factual"] is not None for j in jobs) != 10:
        raise ValueError("fixed ten factual controls unavailable")
    adapter = Path(source_plan["adapter"])
    if evaluation.adapter_identity(adapter) != source_plan["adapter_binding"]:
        raise ValueError("inert SFT48 adapter identity differs")
    generation = evaluation.planner.BASE / "generation_config.json"
    if runtime.read(generation).get("repetition_penalty", 1.0) != 1.0:
        raise ValueError("base generation repetition penalty differs")
    hashes[str(generation)] = probe.campaign.sha(generation)
    plan = dict(
        schema="training-execution-credit-v1",
        root=str(root),
        cases=str(cases_path),
        cases_sha256=hashes[str(cases_path)],
        parents=contract["case_ids"],
        settings=5,
        candidates=4,
        planned_plan_only=320,
        planned_controls=10,
        maximum_new_calls=330,
        control_parent_ids=controls,
        model=source_plan["model"],
        model_manifest_sha256=contract["model_manifest_sha256"],
        inert_adapter=str(adapter),
        inert_adapter_binding=source_plan["adapter_binding"],
        helper_contract={"mode": "base", "model": source_plan["model"]},
        historical_helper_contract=contract["helper_contract"],
        source_hashes=hashes,
        historical_acquisition_cost=runtime.analyze_helper.measured(list(historical.values())),
        budget_seconds=hours * 3600,
        no_training=True,
        no_root_or_helper_generation=True,
        seed_policy="setting0 exact RL downstream+2; settings1..4 frozen execution seed+100",
        sampling=dict(
            do_sample=True, temperature=0.5, top_p=1.0, top_k=0, max_new_tokens=128, max_time=90.0
        ),
        runtime_metadata_difference="FinalClient base condition, inert disabled SFT48 wrapper; "
        "original flat versus frozen nested sampling. Repetition penalty uses fixed "
        "base generation_config=1.",
        caveat="TRAIN diagnostic, not generalization. E-P is action dependent: "
        "changes objective, not an unbiased policy baseline.",
        dependencies={
            str(Path(m.__file__).resolve()): probe.campaign.sha(m.__file__)
            for m in (runtime, evaluation, frozen, rl_planner, probe)
        },
        metric_sha256=probe.campaign.sha(probe.MUSIQUE / "metrics/answer.py"),
        collector_sha256=probe.campaign.sha(__file__),
    )
    return plan, {p: cases[p] for p in plan["parents"]}, jobs


def run(args):
    plan, cases, jobs = prepare(args.root, args.output, args.hours)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime.immutable(output / "PLAN.json", plan)
    if args.prepare_only:
        print(json.dumps({"jobs": len(jobs), "controls": 10, "maximum_calls": 330}))
        return
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if {p.stem[6:] for p in output.glob("OWNER-*.json")} != {
        p.stem[9:] for p in output.glob("TERMINAL-*.json")
    }:
        raise ValueError("previous owner unresolved")
    spent = sum(runtime.read(p)["elapsed_seconds"] for p in output.glob("TERMINAL-*.json"))
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + plan["budget_seconds"] - spent, lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient cumulative/allocation budget")
    tokenizer = AutoTokenizer.from_pretrained(
        evaluation.planner.BASE, local_files_only=True, trust_remote_code=False
    )
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started, stopped, failure, model = (
        uuid.uuid4().hex[:12],
        time.time(),
        False,
        None,
        None,
    )
    try:
        probe.runtime.save(
            output / f"OWNER-{invocation}.json",
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
            nonlocal stopped
            stopped = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        base = AutoModelForCausalLM.from_pretrained(
            evaluation.planner.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base, plan["inert_adapter"], is_trainable=False, autocast_adapter_dtype=True
        )
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        probe.runtime.save(
            output / f"LOAD-{invocation}.json",
            dict(
                helper_model_loaded=False,
                optimizer_created=False,
                final_adapter_enabled=False,
                trainable_parameters=0,
                environment={
                    "python": runtime.sys.version,
                    "executable": runtime.sys.executable,
                    **{
                        name: runtime.importlib.metadata.version(name)
                        for name in ("torch", "transformers", "peft")
                    },
                },
                torch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
            ),
        )
        client = runtime.FinalClient(
            model,
            tokenizer,
            output,
            deadline,
            plan["inert_adapter_binding"]["adapter_model.safetensors"],
        )
        schedule = [(j, "factual_replay") for j in jobs if j["control"]] + [
            (j, "plan_only") for j in jobs
        ]
        for job, mode in schedule:
            if stopped or time.time() >= deadline - 5 or (output / "STOP").exists():
                stopped = True
                break
            runtime.collect(client, job, cases[job["case_id"]], mode)
            probe.campaign.snapshot(
                output / "STATUS.json",
                dict(
                    state="collecting",
                    mode=mode,
                    returned=client.returned,
                    failed=client.failed,
                    updated=time.time(),
                ),
            )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            dict(
                failure=failure,
                stopped=stopped,
                ended=time.time(),
                elapsed_seconds=time.time() - started,
                deadline=deadline,
            ),
        )
        probe.campaign.snapshot(
            output / "STATUS.json",
            dict(state="failed" if failure else "finished_or_capped", failure=failure),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1 / 3)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
