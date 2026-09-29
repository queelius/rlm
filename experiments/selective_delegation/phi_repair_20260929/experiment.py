"""Prepare or run the single approved Phi stable-visible repair comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import common as c


def prepare_data():
    from transformers import AutoTokenizer

    _, _, acquire, qualify = c.phi()
    source_manifest = c.read(c.SOURCE / "MANIFEST.json")
    c.require(
        c.sha(c.SOURCE / "rows.jsonl")
        == "f82cf785ae430ae9e0473c5bb34296835a7ebbfb773278c37a7b3ee4c80b8d1d",
        "frozen stable histories changed",
    )
    c.require(
        source_manifest["all_query_names_public"]
        and source_manifest["training_target_order_preserved"],
        "unqualified stable source",
    )
    source = [json.loads(line) for line in (c.SOURCE / "rows.jsonl").read_text().splitlines()]
    rows = [
        {
            k: row[k]
            for k in ("task_id", "step", "source_action_index", "prompt", "target", "feedback")
        }
        for row in source
    ]
    known_path = acquire.OUTPUT / "known/rows.jsonl"
    dose = c.read(acquire.OUTPUT / "QUALIFICATION.json")["teaching_doses"]["known"]
    c.require(c.sha(known_path) == dose["output_rows_sha256"], "Phi known source changed")
    known = [json.loads(line) for line in known_path.read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(
        acquire.MODEL, local_files_only=True, trust_remote_code=False
    )
    examples, baseline = (qualify.tokenize_rows(data, tokenizer) for data in (rows, known))
    tokens = c.match_targets(rows, known, examples, baseline)
    c.require(tokens == 9081 and len(rows) == 366, "native Phi dose differs")
    recipe = qualify.recipe
    order = recipe.epoch_order(366, c.SEED, 0)
    denominators = [
        sum(len(examples[i]["target_ids"]) for i in order[start : start + 16])
        for start in range(0, 366, 16)
    ]
    actual = [c.read(c.KNOWN / "steps" / f"{i:04d}.json")["target_tokens"] for i in range(1, 24)]
    c.require(denominators == actual, "existing known Phi minibatch target dose differs")
    tasks = (c.SOURCE / "tasks.jsonl").read_bytes()
    c.require(tasks == (acquire.OUTPUT / "known/tasks.jsonl").read_bytes(), "root/stock differs")
    c.PREPARED.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    for name, value in (("rows.jsonl", text.encode()), ("tasks.jsonl", tasks)):
        path = c.PREPARED / name
        if path.exists():
            c.require(path.read_bytes() == value, "immutable repair input differs")
        else:
            with path.open("xb") as stream:
                stream.write(value)
    audit = dict(
        schema="phi-repair-native-target-match-20260929-v1",
        rows=366,
        tasks=32,
        exact_targets_row_order_and_native_labels=True,
        supervised_tokens=tokens,
        prompt_tokens=sum(e["prompt_tokens"] for e in examples),
        known_prompt_tokens=sum(e["prompt_tokens"] for e in baseline),
        max_full_tokens=max(e["prompt_tokens"] + e["target_tokens"] for e in examples),
        max_target_tokens=max(e["target_tokens"] for e in examples),
        target_suffix=qualify.STOP_IDS,
        per_update_target_tokens=denominators,
        target_ids_by_original_row=[e["target_ids"] for e in examples],
        original_known_plan_sha256=c.sha(c.KNOWN / "PLAN.json"),
        original_known_checkpoint0_commit_sha256=c.sha(c.KNOWN / "checkpoint-0000/COMMIT.json"),
        initial_adapter_sha256=c.read(c.KNOWN / "checkpoint-0000/COMMIT.json")["files"][
            "adapter_model.safetensors"
        ],
        source_sha256={
            str(p): c.sha(p)
            for p in (
                c.SOURCE / "MANIFEST.json",
                c.SOURCE / "rows.jsonl",
                c.SOURCE / "tasks.jsonl",
                known_path,
                acquire.OUTPUT / "QUALIFICATION.json",
                Path(qualify.__file__),
            )
        },
        assistance="Existing offline gold-action stable scheduler; public query names. "
        "Prompts copied exactly, including their original Qwen-era budget counters. "
        "Only stale token caches removed; no truncation or new scheduling.",
    )
    c.save(c.PREPARED / "TOKEN-AUDIT.json", audit)
    manifest = dict(
        schema="phi-stable-visible-native-sft-20260929-v1",
        teacher="stable_visible",
        rows=366,
        tasks=32,
        eligible_task_count=32,
        native_successful_tasks=32,
        rows_sha256=c.sha(c.PREPARED / "rows.jsonl"),
        tasks_sha256=c.sha(c.PREPARED / "tasks.jsonl"),
        model=str(acquire.MODEL),
        model_manifest_sha256=c.sha(acquire.MODEL / "local-research-manifest.json"),
        native_template=True,
        token_audit_sha256=c.sha(c.PREPARED / "TOKEN-AUDIT.json"),
    )
    recipe.validate_rows(rows, manifest)
    c.save(c.PREPARED / "MANIFEST.json", manifest)
    c.train_run(prepare_only=True)
    plans = {}
    for world in (42, 50):
        plan, *_ = c.build_readout(world)
        c.save(c.readout_output(world) / "PENDING-PLAN.json", plan)
        plans[str(world)] = c.sha(c.readout_output(world) / "PENDING-PLAN.json")
    receipt = dict(
        schema="phi-repair-preparation-20260929-v1",
        teacher="stable_visible",
        training_plan_sha256=c.sha(c.TRAINING / "PLAN.json"),
        training_contract_sha256=c.sha(c.TRAINING / "PHI-CONTRACT.json"),
        pending_plan_sha256=plans,
        seed=c.SEED,
        planned_updates=23,
        new_attempts=16,
        model_loaded=False,
        gpu_used=False,
    )
    c.save(c.ROOT / "PREPARATION.json", receipt)
    return receipt


def endpoint():
    from safetensors import safe_open

    evaluate, _, acquire, qualify = c.phi()
    preparation = c.read(c.ROOT / "PREPARATION.json")
    dose = c.read(c.PREPARED / "TOKEN-AUDIT.json")
    evaluate.shared.collector.BASE = acquire.MODEL
    binding = evaluate.shared.endpoint(
        c.TRAINING / "checkpoint-0023",
        training_plan_sha256=preparation["training_plan_sha256"],
        rows_sha256=c.read(c.PREPARED / "MANIFEST.json")["rows_sha256"],
    )
    adapter = Path(binding["path"])
    config = c.read(adapter / "adapter_config.json")
    c.require(
        set(config["target_modules"]) == set(qualify.TARGET_MODULES), "Phi LoRA targets differ"
    )
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as file:
        names = file.keys()
        params = sum(math.prod(file.get_slice(k).get_shape()) for k in names)
        c.require(
            all(file.get_slice(k).get_dtype() == "F32" for k in names),
            "actual LoRA checkpoint is not FP32",
        )
    c.require(params == 11534336, "actual Phi trainable parameter count differs")
    denominators = [
        c.read(c.TRAINING / "steps" / f"{i:04d}.json")["target_tokens"] for i in range(1, 24)
    ]
    c.require(denominators == dose["per_update_target_tokens"], "actual target dose differs")
    initial = c.read(c.TRAINING / "checkpoint-0000/COMMIT.json")
    c.require(
        initial["files"]["adapter_model.safetensors"] == dose["initial_adapter_sha256"],
        "actual initialization differs from Phi-known",
    )
    paths = [
        c.TRAINING / "PLAN.json",
        c.TRAINING / "PHI-CONTRACT.json",
        c.TRAINING / "checkpoint-0000/COMMIT.json",
        adapter / "COMMIT.json",
        adapter / "STATE.json",
        c.PREPARED / "TOKEN-AUDIT.json",
        c.ROOT / "PREPARATION.json",
    ]
    paths += [Path(p) for p in binding["step_receipts_sha256"]]
    report = dict(
        schema="phi-repair-endpoint-audit-20260929-v1",
        passed=True,
        teacher="stable_visible",
        binding=binding,
        trainable_parameters=params,
        matched_initialization=True,
        actual_per_update_target_tokens=denominators,
        supervised_tokens=sum(denominators),
        small_files={str(p): c.sha(p) for p in paths},
        adapter_stat=c.stamp(adapter / "adapter_model.safetensors"),
        gpu_used=False,
        reuse="Adapter contents hashed once by native endpoint authentication; subsequent "
        "owners check unchanged stat plus bound small receipts. No base-weight ancestry rehash.",
    )
    c.save(c.ROOT / "ENDPOINT-AUDIT.json", report)
    return report


def native_audit(world):
    evaluate, _, _, _ = c.phi()
    output = c.readout_output(world)
    plan = c.read(output / "PLAN.json")
    c.require(plan["adapter"] == c.cached_endpoint(), "readout endpoint differs")
    result = evaluate.audit(output)
    c.save(output / "PHI-AUDIT.json", result)
    return result


def descriptors():
    preparation = c.read(c.ROOT / "PREPARATION.json")
    fixture = c.read(c.ROOT / "seam-fixture-001/VERIFICATION.json")
    c.require(fixture["passed"], "actual native seam fixture required")
    pins = {
        str(p): c.sha(p)
        for p in (
            c.HERE / "common.py",
            Path(__file__),
            c.HERE / "seam_fixture.py",
            c.ROOT / "PREPARATION.json",
            c.ROOT / "seam-fixture-001/VERIFICATION.json",
            c.PREPARED / "MANIFEST.json",
            c.PREPARED / "TOKEN-AUDIT.json",
            c.PREPARED / "rows.jsonl",
            c.PREPARED / "tasks.jsonl",
            c.TRAINING / "PLAN.json",
            c.TRAINING / "PHI-CONTRACT.json",
        )
    }
    pins.update(c.read(c.TRAINING / "PHI-CONTRACT.json")["source_sha256"])
    for path in (
        c.E / "teaching_synthesis_20260929/analyze.py",
        c.E / "controls_readout_20260928/readout.py",
    ):
        pins[str(path)] = c.sha(path)
    for world in (42, 50):
        path = c.readout_output(world) / "PENDING-PLAN.json"
        c.require(
            c.sha(path) == preparation["pending_plan_sha256"][str(world)],
            "pending readout template changed",
        )
        pins[str(path)] = c.sha(path)
        pins.update(c.read(path)["source_sha256"])
        for teacher in ("known", "discovery"):
            base = c.R / f"textcraft-phi-{teacher}-raw-w{world}-20260928-001"
            for name in ("PLAN.json", "PHI-AUDIT.json"):
                pins[str(base / name)] = c.sha(base / name)
    jobs = []

    def add(name, command, cap, output=None, world=None):
        argv = [str(c.PYTHON), str(Path(__file__).resolve()), command]
        if world is not None:
            argv += ["--world", str(world)]
        job = dict(name=name, argv=argv, cap_seconds=cap, pins=pins)
        if output is not None:
            job["output"] = str(output)
        jobs.append(job)

    add("phi-repair-stable-train23", "train", 2100, c.TRAINING)
    add("phi-repair-endpoint-audit", "endpoint", 300)
    for world in (42, 50):
        add(f"phi-repair-raw-w{world}", "readout", 2100, c.readout_output(world), world)
        add(f"phi-repair-native-audit-w{world}", "audit", 300, world=world)
    add("phi-repair-paired-report", "report", 300)
    result = dict(
        schema="phi-repair-prepared-jobs-20260929-v1",
        jobs=jobs,
        scientific_stages=3,
        new_attempts=16,
        expected_gpu_minutes=[15, 30],
        scientific_cap_minutes=90,
        executor_cap_seconds=sum(j["cap_seconds"] for j in jobs),
        dependency="Run in listed order. Training completion and actual endpoint audit gate "
        "both readouts; per-world audit gates aggregate interpretation. Output only denotes "
        "scientific owners, never CPU audits. Failed dependencies remain missing/unknown.",
    )
    c.save(c.ROOT / "PREPARED-JOBS.json", result)
    return result


def report():
    path = c.E / "teaching_synthesis_20260929/analyze.py"
    spec = importlib.util.spec_from_file_location("phi_repair_pair_math", path)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    rows, cells, receipts = {}, {}, {str(path): c.sha(path)}
    for teacher in ("known", "discovery", "stable_visible"):
        rows[teacher] = []
        for world in (42, 50):
            output = (
                c.readout_output(world)
                if teacher == "stable_visible"
                else (c.R / f"textcraft-phi-{teacher}-raw-w{world}-20260928-001")
            )
            plan, audit = (c.read(output / name) for name in ("PLAN.json", "PHI-AUDIT.json"))
            c.require(
                audit["passed"] and audit["plan_sha256"] == c.sha(output / "PLAN.json"),
                "native audit authority differs",
            )
            c.require(
                plan["teacher"] == teacher
                and plan["world_seed"] == world
                and plan["assistance"] == "raw"
                and len(plan["jobs"]) == 8,
                "wrong fixed teacher/world/count",
            )
            for name in ("PLAN.json", "PHI-AUDIT.json"):
                receipts[str(output / name)] = c.sha(output / name)
            for job, row in zip(plan["jobs"], audit["episodes"], strict=True):
                c.require(job["task_id"] == row["task_id"], "native task identity differs")
                rows[teacher].append(
                    dict(
                        task_id=job["task_id"],
                        panel=0,
                        fit_seed=c.SEED,
                        world=world,
                        rollout_seed=job["seed"],
                        score=row["native_score"] if row["observed"] else None,
                    )
                )
            cells[f"{teacher}:w{world}"] = dict(
                output=str(output),
                successes=audit["successes"],
                observed=audit["observed"],
                planned=8,
                unknown=audit["unknown"],
                cost=audit["physical_cost"],
                adapter=plan["adapter"],
            )
    result = dict(
        schema="phi-repair-paired-native-report-20260929-v1",
        cells=cells,
        comparisons={
            f"stable_visible-{teacher}": analysis.paired(rows[teacher], rows["stable_visible"])
            for teacher in ("known", "discovery")
        },
        receipts=receipts,
        caveat="Eight exposed roots, one fit and rollout seed, two worlds. Repair transfer "
        "is not a visibility-only causal effect; equal answer labels do not equal input dose.",
    )
    c.save(c.ROOT / "PAIRED-REPORT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "train", "endpoint", "readout", "audit", "descriptors", "report"),
    )
    parser.add_argument("--world", type=int, choices=(42, 50))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_data()
    elif args.command == "train":
        c.train_run()
        result = dict(training_returned=True)
    elif args.command == "endpoint":
        result = endpoint()
    elif args.command == "readout":
        c.run_readout(args.world)
        result = dict(readout_returned=True)
    elif args.command == "audit":
        result = native_audit(args.world)
    elif args.command == "descriptors":
        result = descriptors()
    else:
        result = report()
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("jobs", "small_files")}, indent=2)
    )
