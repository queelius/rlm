"""Frozen-plan final-only diagnostic plus outcome-independent factual replay controls."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import eval_planner as evaluation
import probe

POLICIES = ("sft", "rl")
CASES_SHA = "6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153"
SEED = 2026092173


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("immutable input/output changed: " + str(path))
    else:
        probe.runtime.save(path, value)


def control_parents(parents):
    return sorted(parents)[:4]


def plan_only_prompt(case, plan):
    return evaluation.final_prompt(case, plan, {"execution": "isolated", "steps": []})


def source_grade(episode, final, case):
    if final is None:
        if episode["status"] != "invalid_dependency":
            raise ValueError("missing factual final is not the declared dependency failure")
        return {**probe.grade("", case), "observed": True, "status": "invalid_dependency"}
    if not final["available"]:
        raise ValueError("unavailable factual source requires separate accounting")
    grade = probe.grade(final["text"], case)
    return {**grade, "observed": True, "status": "scored" if grade["valid"] else "invalid_final"}


def prepare(root, output, hours):
    if not 0 < hours <= 1 / 3 + 1e-9 or root == output:
        raise ValueError("separate output and maximum20 minutes required")
    report_path = root / "analysis-fresh-contract-policy-001.json"
    audit = read(report_path)
    audited, hashes = audit["input_native_source_sha256"], {}

    def tracked(path, *, require_audited=True):
        path = Path(path).resolve()
        digest = probe.campaign.sha(path)
        if require_audited and audited.get(str(path)) != digest:
            raise ValueError(
                "small source receipt differs from completed native audit: " + str(path)
            )
        hashes[str(path)] = digest
        return read(path)

    tracked(report_path, require_audited=False)
    cases_path = root / "fresh-dev-inputs-003/cases.jsonl"
    if probe.campaign.sha(cases_path) != CASES_SHA:
        raise ValueError("requires fixed fresh003 cases")
    hashes[str(cases_path)] = CASES_SHA
    cases = {c["id"]: c for c in map(json.loads, cases_path.read_text().splitlines())}
    if len(cases) != 64:
        raise ValueError("expected64 unique parents")
    controls = control_parents(cases)
    jobs, source_plans = [], {}
    sampling = dict(
        do_sample=True, temperature=0.5, top_p=1.0, top_k=0, max_new_tokens=128, max_time=90.0
    )
    for policy in POLICIES:
        source = Path(audit["groups"]["planner_" + policy]["source"])
        plan = tracked(source / "PLAN.json")
        source_plans[policy] = {"output": str(source), **plan}
        if (
            set(plan["case_ids"]) != set(cases)
            or plan["cases_sha256"] != CASES_SHA
            or plan["repeats"] != 2
            or policy not in plan["conditions"]
            or plan["execution"] != "isolated"
            or plan.get("mode", "planner") != "planner"
        ):
            raise ValueError("source is not the declared fresh planner panel")
        owners = list(source.glob("OWNER-*.json"))
        if not owners or {p.stem[6:] for p in owners} != {
            p.stem[9:] for p in source.glob("TERMINAL-*.json")
        }:
            raise ValueError("source owner not completed/released")
        for owner in owners:
            tracked(owner, require_audited=False)
            terminal = tracked(
                owner.with_name(owner.name.replace("OWNER-", "TERMINAL-")), require_audited=False
            )
            if terminal["failure"] or terminal["stopped"]:
                raise ValueError("source owner failed/stopped")
        for parent in plan["case_ids"]:
            for repeat in range(2):
                identity = f"{parent}-r{repeat}-{policy}-isolated"
                episode = tracked(source / "episodes" / (identity + ".json"))
                if (episode["case_id"], episode["repeat"], episode["condition"]) != (
                    parent,
                    repeat,
                    policy,
                ):
                    raise ValueError("episode identity mismatch")
                root_call = tracked(source / "calls" / (identity + "-root.json"))
                generated = evaluation.parse_plan(root_call["text"])
                if (
                    not root_call["available"]
                    or generated != episode["plan"]
                    or root_call["request"]["prompt"] != evaluation.planner_prompt(cases[parent])
                    or probe.runtime.digest(root_call["request"]) != root_call["request_digest"]
                ):
                    raise ValueError("original root cannot be reconstructed")
                final_ids = [cid for cid in episode["call_ids"] if cid.endswith("-final")]
                if len(final_ids) > 1:
                    raise ValueError("multiple source finals")
                final = tracked(source / "calls" / (final_ids[0] + ".json")) if final_ids else None
                trace = [
                    {k: t[k] for k in ("step", "question", "resolved_question", "answer")}
                    for t in episode.get("helper_trace", [])
                ]
                factual = evaluation.final_prompt(
                    cases[parent], generated, {"execution": "isolated", "steps": trace}
                )
                seed = plan["seed"] + int(probe.runtime.digest(parent)[:6], 16) + repeat * 100 + 2
                if final and (
                    final_ids != [identity + "-final"]
                    or final["request"]["prompt"] != factual
                    or probe.runtime.digest(final["request"]) != final["request_digest"]
                    or final["request"]["sampling"] != sampling
                    or final["request"]["seed"] != seed
                    or final["request"]["adapter_enabled"]
                    or final["request"]["model"] != str(evaluation.planner.BASE)
                ):
                    raise ValueError("factual final scientific request differs")
                old = source_grade(episode, final, cases[parent])
                jobs.append(
                    dict(
                        case_id=parent,
                        repeat=repeat,
                        policy=policy,
                        source_episode_id=identity,
                        seed=seed,
                        root=root_call,
                        factual=final,
                        old=old,
                        factual_prompt=factual,
                        plan_only_prompt=plan_only_prompt(cases[parent], generated),
                        generated_plan=generated,
                        control=parent in controls,
                        source_episode_cost=episode["deployed_cost"],
                    )
                )
    if len(jobs) != 256 or sum(j["factual"] is not None for j in jobs) != 255:
        raise ValueError("expected256 slots:255 factual finals and one dependency zero")
    first, other = [source_plans[p] for p in POLICIES]
    for field in (
        "cases_sha256",
        "case_ids",
        "repeats",
        "seed",
        "model",
        "model_manifest_sha256",
        "caps",
        "temperature",
        "top_p",
        "top_k",
        "helper_contract",
    ):
        if first[field] != other[field]:
            raise ValueError("source policy execution contract differs: " + field)
    adapter = Path(first["adapter"])
    binding = evaluation.adapter_identity(
        adapter
    )  # One small LoRA identity check; no base weight walk.
    if binding != first["adapter_files_sha256"] or read(adapter / "STATE.json")["step"] != 48:
        raise ValueError("inert SFT48 wrapper identity differs")
    manifest = evaluation.planner.BASE / "local-research-manifest.json"
    if probe.campaign.sha(manifest) != first["model_manifest_sha256"]:
        raise ValueError("base model manifest differs")
    hashes[str(manifest)] = first["model_manifest_sha256"]
    generation = evaluation.planner.BASE / "generation_config.json"
    hashes[str(generation)] = probe.campaign.sha(generation)
    frozen = dict(
        schema="plan-only-final-v1",
        root=str(root),
        cases=str(cases_path),
        cases_sha256=CASES_SHA,
        parents=list(cases),
        repeats=2,
        policies=list(POLICIES),
        source_plans=source_plans,
        control_parent_ids=controls,
        planned_plan_only=256,
        planned_controls=16,
        available_factual_finals=255,
        observed_source_dependency_zeros=1,
        maximum_new_calls=272,
        source_hashes=hashes,
        model=str(evaluation.planner.BASE),
        inert_adapter=str(adapter),
        inert_adapter_binding=binding,
        sampling=sampling,
        helper_contract={"mode": "base", "model": str(evaluation.planner.BASE)},
        runtime_metadata_difference="Native condition base and helper_contract base; "
        "source had trained_helper metadata. All finals use the same adapter-disabled base; "
        "no helper model is loaded and no root/helper generation occurs.",
        budget_seconds=hours * 3600,
        bootstrap_draws=20000,
        bootstrap_seed=SEED,
        dependencies={
            str(p): probe.campaign.sha(p)
            for p in (
                Path(__file__),
                Path(evaluation.__file__),
                Path(evaluation.planner.__file__),
                Path(analyze_helper.__file__),
                Path(probe.__file__),
                probe.MUSIQUE / "metrics/answer.py",
            )
        },
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        ),
    )
    return frozen, cases, jobs


class FinalClient(evaluation.HFClient):
    def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
        if (condition, role, max_new_tokens) != ("base", "final", 128):
            raise ValueError("final-only base generation required")
        saved = self.output / "calls" / (identity + ".json")
        if not saved.exists() and any((self.output / "starts").glob(identity + "-*.json")):
            raise RuntimeError("unresolved native start; no silent retry")
        first = self.returned == 0
        row = super().call(identity, prompt, condition, role, seed, max_new_tokens=max_new_tokens)
        for p in self.model.parameters():
            p.requires_grad_(False)
        if first and row["available"] and row["ended"] - row["started"] > 90:
            raise RuntimeError("first real scientific response exceeded90 seconds")
        return row


def collect(client, job, case, mode):
    identity = f"{job['case_id']}-r{job['repeat']}-{job['policy']}-{mode}"
    path = client.output / "episodes" / (identity + ".json")
    if path.exists():
        return read(path)
    row = dict(
        episode_id=identity,
        case_id=job["case_id"],
        repeat=job["repeat"],
        policy=job["policy"],
        mode=mode,
        old=job["old"],
        source_episode_id=job["source_episode_id"],
        source_root_call_id=job["root"]["call_id"],
        source_final_call_id=job["factual"]["call_id"] if job["factual"] else None,
        source_final_available=job["factual"] is not None,
        seed=job["seed"],
        observed=False,
        status="control_source_unavailable",
        call_ids=[],
        new=probe.grade("", case),
        new_physical_cost=evaluation.cost([]),
        hypothetical_deployed_cost=None,
    )
    if mode == "plan_only" or job["factual"]:
        prompt = job["plan_only_prompt"] if mode == "plan_only" else job["factual_prompt"]
        call = client.call(
            identity + "-final", prompt, "base", "final", job["seed"], max_new_tokens=128
        )
        grade = probe.grade(call["text"] if call["available"] else "", case)
        row.update(
            call_ids=[call["call_id"]],
            new=grade,
            observed=call["available"],
            status=("scored" if grade["valid"] else "invalid_final")
            if call["available"]
            else "unavailable",
            new_physical_cost=evaluation.cost([call]),
            hypothetical_deployed_cost=evaluation.cost([job["root"], call])
            if mode == "plan_only"
            else None,
        )
        if mode == "factual_replay":
            old = job["factual"]
            if call["request"]["input_token_ids"] != old["request"]["input_token_ids"]:
                raise ValueError("factual replay tokenizer/input identity changed")
            row.update(
                same_output_tokens=call.get("output_token_ids") == old.get("output_token_ids"),
                same_text=call.get("text") == old.get("text"),
                same_grade=all(grade[k] == job["old"][k] for k in ("valid", "correct", "f1")),
            )
    probe.runtime.save(path, row)
    if row["status"] == "unavailable":
        raise RuntimeError("native final unavailable; halt without retry")
    return row


def summarize(output, plan, cases, jobs, *, intervals=False):
    rows = [read(p) for p in (output / "episodes").glob("*.json")]
    calls = [read(p) for p in (output / "calls").glob("*.json")]
    starts = [read(p) for p in (output / "starts").glob("*.json")]
    returned = {r["call_id"] for r in calls}
    unresolved = [r for r in starts if r["call_id"] not in returned]
    groups = {}
    for policy in POLICIES:
        current = [r for r in rows if r["policy"] == policy and r["mode"] == "plan_only"]
        indexed = {(r["case_id"], r["repeat"]): r for r in current}
        source = [j for j in jobs if j["policy"] == policy]
        observed = [r for r in current if r["observed"]]
        changes = Counter()
        for row in observed:
            delta = int(row["new"]["correct"]) - int(row["old"]["correct"])
            if delta:
                category = (
                    "source_dependency_recovery"
                    if not row["source_final_available"]
                    else (
                        "both_valid"
                        if row["old"]["valid"] and row["new"]["valid"]
                        else "protocol_involved"
                    )
                )
                changes[("win_" if delta > 0 else "loss_") + category] += 1
        group = dict(
            planned=128,
            recorded=len(current),
            observed=len(observed),
            missing=128 - len(observed),
            factual_correct=sum(j["old"]["correct"] for j in source),
            new_correct=sum(r["new"]["correct"] for r in observed),
            new_em_lower_bound=sum(r["new"]["correct"] for r in observed) / 128,
            new_f1_lower_bound=sum(r["new"]["f1"] for r in observed) / 128,
            statuses=dict(Counter(r["status"] for r in current)),
            changes=dict(changes),
        )
        if intervals and len(observed) == 128:
            clusters = analyze_helper.component_clusters(list(cases.values()))
            group["component_clusters"] = clusters
            for metric in ("correct", "f1"):
                delta = {
                    p: mean(
                        float(indexed[p, r]["new"][metric]) - float(indexed[p, r]["old"][metric])
                        for r in range(2)
                    )
                    for p in plan["parents"]
                }
                group[metric + "_difference"] = analyze_helper.clustered_interval(
                    delta, clusters, 20000, SEED
                )
        groups[policy] = group
    controls = [r for r in rows if r["mode"] == "factual_replay"]
    reused = [j["root"] for j in jobs] + [j["factual"] for j in jobs if j["factual"]]
    return dict(
        groups=groups,
        control_statuses=dict(Counter(r["status"] for r in controls)),
        controls_recorded=len(controls),
        control_disagreements=[
            r["episode_id"]
            for r in controls
            if r["observed"] and not r.get("same_output_tokens", False)
        ],
        new_physical_cost=analyze_helper.measured(calls + unresolved),
        research_control_cost=analyze_helper.measured(
            [r for r in calls if "factual_replay" in r["call_id"]]
        ),
        reused_root_and_factual_final_acquisition_cost=analyze_helper.measured(reused),
        reused_source_policy_cost={
            p: sum(
                j["source_episode_cost"]["prompt_tokens"]
                + j["source_episode_cost"]["completion_tokens"]
                for j in jobs
                if j["policy"] == p
            )
            for p in POLICIES
        },
        unresolved_starts=len(unresolved),
        planned_new_plan_only=256,
        planned_controls=16,
        all_slots_recorded=len(rows) == 272,
        note="Missing is unobserved, not scientific failure. Reused source costs are historical; "
        "hypothetical plan-only deployment charges saved root plus new final, never helpers or "
        "replay controls. Summed call latency is not measured deployment wall-time speedup.",
    )


def run(args):
    root, output = args.root.resolve(), args.output.resolve()
    plan, cases, jobs = prepare(root, output, args.hours)
    output.mkdir(parents=True, exist_ok=True)
    immutable(output / "PLAN.json", plan)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "jobs": len(jobs),
                    "factual_finals": 255,
                    "controls_available": sum(j["control"] and bool(j["factual"]) for j in jobs),
                    "control_parents": plan["control_parent_ids"],
                    "max_new_calls": 272,
                }
            )
        )
        return
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    owners = {p.stem[6:] for p in output.glob("OWNER-*.json")}
    terminals = {p.stem[9:] for p in output.glob("TERMINAL-*.json")}
    if owners != terminals:
        raise ValueError("previous owner unresolved")
    spent = sum(read(p)["elapsed_seconds"] for p in output.glob("TERMINAL-*.json"))
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600 - spent, lease - 600)
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
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
            ),
        )
        client = FinalClient(
            model,
            tokenizer,
            output,
            deadline,
            plan["inert_adapter_binding"]["adapter_model.safetensors"],
        )
        schedule = [(job, "factual_replay") for job in jobs if job["control"]]
        schedule += [(job, "plan_only") for job in jobs]
        for job, mode in schedule:
            if stopped or time.time() >= deadline - 5 or (output / "STOP").exists():
                stopped = True
                break
            collect(client, job, cases[job["case_id"]], mode)
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
            if client.returned % 16 == 0:
                probe.campaign.snapshot(
                    output / "SUMMARY.json", summarize(output, plan, cases, jobs)
                )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan, cases, jobs))
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
    analysis = summarize(output, plan, cases, jobs, intervals=True)
    target = "ANALYSIS.json" if analysis["all_slots_recorded"] else f"ANALYSIS-{invocation}.json"
    immutable(output / target, analysis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1 / 3)
    parser.add_argument(
        "--prepare-only", "--validate-only", action="store_true", dest="prepare_only"
    )
    run(parser.parse_args())
