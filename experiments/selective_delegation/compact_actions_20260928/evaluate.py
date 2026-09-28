"""Fixed breadth-p00 compact/full-binder readout; parent owns every GPU launch."""

import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import compact_bridge as compact
import eval_textcraft_public as public
import eval_textcraft_trained as shared
import prepare as data

ROOT = data.ROOT
TRAIN_OUTPUT = ROOT / "textcraft-compact-sft-20260928-001"
TRAIN_PLAN_SHA = "445a7b5762b4c2c7ea6892083757128148b83a00ce435f1f6d1516200b3e3f56"
TRAIN_CONTRACT_SHA = "19fcb2e78cd381ac066e6e84c4a501283a3586711d0b3b32b8e7225ba003c9bc"
BINDER = ROOT / "source-textcraft-recipe-binder-003"
BINDER_PINS = {
    "eval_textcraft.py": "1e8156909b2c17cbed20d3add5e429daece4d602979e6e44bde628ad7296c92b",
    "analyze_textcraft.py": "f1e095401698432fcf630b62baba88df9852e3b0699f711553ba583a382344d1",
}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def implementation(mode: str):
    if mode not in ("compact", "binder"):
        raise ValueError("unknown interface")
    library = compact.LIBRARY if mode == "compact" else BINDER
    if mode == "binder":
        for filename, digest in BINDER_PINS.items():
            if data.sha(library / filename) != digest:
                raise ValueError("sealed binder implementation changed")
    collector = load(library / "eval_textcraft.py", "compact_probe_collector_" + mode)
    auditor = load(library / "analyze_textcraft.py", "compact_probe_auditor_" + mode)
    if mode == "compact":
        collector.bridge = compact
    auditor.collector, auditor.bridge = collector, collector.bridge
    return collector, auditor


def world_for(seed: int):
    if seed not in (42, 50):
        raise ValueError("first fixed panel only in worlds42/50")
    original = compact.load_world()
    if seed == 42:
        return original
    module = sys.modules["pinned_textcraft_synth_generator_d9c5857d"]
    module.set_naming_mode(semantic=False)
    world = module.SynthRecipeDatabase()
    world.generate_all_recipes(seed=seed, items_per_domain_tier=25)
    return world


def binding(mode: str):
    if mode == "binder":
        adapter = ROOT / "textcraft-public-discovery-sft-001/checkpoint-0023"
        return shared.endpoint(
            adapter, training_plan_sha256=public.PLAN_SHA, rows_sha256=public.ROWS_SHA
        )
    if data.sha(TRAIN_OUTPUT / "COMPACT-CONTRACT.json") != TRAIN_CONTRACT_SHA:
        raise ValueError("compact training contract changed")
    contract = data.accepted.read(TRAIN_OUTPUT / "COMPACT-CONTRACT.json")
    for path, digest in contract["source_sha256"].items():
        if data.sha(Path(path)) != digest:
            raise ValueError("compact training/interface source changed")
    return shared.endpoint(
        TRAIN_OUTPUT / "checkpoint-0023",
        training_plan_sha256=TRAIN_PLAN_SHA,
        rows_sha256=contract["rows_sha256"],
    )


def build(args, require_endpoint=True):
    if not 0 < args.hours <= 0.75:
        raise ValueError("each16-episode readout is capped at45minutes")
    collector, auditor = implementation(args.mode)
    baseline = ROOT / f"textcraft-breadth-p00-w{args.world}-soriginal-binder-001"
    base = data.accepted.read(baseline / "PLAN.json")
    prepared = Path(base["prepared"])
    manifest = data.accepted.read(prepared / "MANIFEST.json")
    if data.sha(prepared / "MANIFEST.json") != base["manifest_sha256"] or (
        data.sha(prepared / "tasks.jsonl") != base["tasks_sha256"]
    ):
        raise ValueError("fixed breadth00 task source changed")
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    if len(tasks) != 8 or any(not t["id"].startswith("textcraft_synth.val.") for t in tasks):
        raise ValueError("exact eight fixed evaluation rows required")
    world = world_for(args.world)
    world_sha = data.worlds.digest(data.worlds.snapshot(world))
    if world_sha != base["world_sha256"] or world_sha != manifest["world_sha256"]:
        raise ValueError("fixed native world identity mismatch")
    if (
        base["seeds"] != [2026092204, 2026092205]
        or len(base["jobs"]) != 16
        or base["budget_seconds"] != 2700
    ):
        raise ValueError("fixed paired baseline schedule changed")
    jobs = [
        dict(job, condition=f"compact_probe_p00_w{args.world}_{args.mode}") for job in base["jobs"]
    ]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        collector.BASE, local_files_only=True, trust_remote_code=False
    )
    initial_tokens = [
        len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": collector.bridge.initial_prompt(t, "flat")}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        for t in tasks
    ]
    if max(initial_tokens) + 256 > 8192:
        raise ValueError("initial context cap exceeded")
    endpoint = binding(args.mode) if require_endpoint else None
    paths = {
        Path(__file__),
        Path(compact.__file__),
        Path(compact.native.__file__),
        Path(data.__file__),
        Path(collector.__file__),
        Path(auditor.__file__),
        Path(collector.inputs.__file__),
        Path(collector.probe.__file__),
        Path(collector.probe.runtime.__file__),
        Path(collector.probe.campaign.__file__),
        Path(shared.__file__),
        Path(public.__file__),
    }
    if args.mode == "binder":
        paths.add(Path(collector.recipe_binder.__file__))
    plan = dict(
        schema="textcraft-compact-interface-readout-20260928-v1",
        interface=args.mode,
        prepared=str(prepared),
        manifest_sha256=data.sha(prepared / "MANIFEST.json"),
        tasks_sha256=data.sha(prepared / "tasks.jsonl"),
        jobs=jobs,
        planned_episodes=16,
        planned_per_condition=16,
        parent_tasks=8,
        seeds=base["seeds"],
        conditions=sorted({j["condition"] for j in jobs}),
        profile="original",
        model=str(collector.BASE),
        model_manifest_sha256=base["model_manifest_sha256"],
        sampling=base["sampling"],
        max_global_calls=96,
        max_global_output_tokens=8192,
        max_new_tokens=256,
        input_plus_output_limit=8192,
        truncation=False,
        max_agent_depth={"flat": 0},
        root_depth=0,
        max_native_calls=1536,
        budget_seconds=args.hours * 3600,
        optimizer=None,
        seed_rule=base["seed_rule"],
        world_seed=args.world,
        world_sha256=world_sha,
        adapter=endpoint,
        endpoint_pending=not require_endpoint,
        baseline_output=str(baseline),
        baseline_plan_sha256=data.sha(baseline / "PLAN.json"),
        initial_token_audit={
            "prompt_tokens": initial_tokens,
            "max_prompt_plus_cap": max(initial_tokens) + 256,
        },
        source_sha256={str(path.resolve()): data.sha(path) for path in sorted(paths)},
        trusted_source=compact.trusted_provenance(),
        training_plan_sha256=TRAIN_PLAN_SHA if args.mode == "compact" else public.PLAN_SHA,
        compact_training_contract_sha256=TRAIN_CONTRACT_SHA if args.mode == "compact" else None,
        comparison="Same preexisting panel00,world,tasks,job order,seeds,sampling,budgets. "
        "Compact matched-epoch SFT versus fullformat discovery SFT plus observed-recipe binder. "
        "Changed schema/instruction/history/token dose; not equal compute or pure "
        "likelihood effect.",
        caveat="Exploratory one-training-seed fixed-panel comparison. Repeated sampling/worlds "
        "do not create independent task roots. Same-world training recipe overlap remains.",
    )
    return plan, tasks, world, tokenizer, collector, auditor


def audit(output: Path):
    plan = data.accepted.read(output / "PLAN.json")
    collector, auditor = implementation(plan["interface"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        collector.BASE, local_files_only=True, trust_remote_code=False
    )
    report = auditor.analyze(
        output, tokenizer, world=world_for(plan["world_seed"]), expected_task_count=8
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, choices=(42, 50), default=42)
    parser.add_argument("--mode", choices=("compact", "binder"), default="compact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.75)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        report = audit(args.output.resolve())
        data.save(args.output / "COMPACT-AUDIT.json", report)
        print(json.dumps(report["groups"], indent=2))
    else:
        plan, tasks, world, _, collector, _ = build(args, not args.validate_inputs_only)
        if args.validate_inputs_only:
            print(
                json.dumps(
                    {
                        "planned_episodes": 16,
                        "world": args.world,
                        "endpoint_pending": True,
                        "tasks_sha256": plan["tasks_sha256"],
                        "initial_token_audit": plan["initial_token_audit"],
                    },
                    indent=2,
                )
            )
        else:
            path = args.output / "PLAN.json"
            if path.exists():
                if data.accepted.read(path) != plan:
                    raise ValueError("immutable compact readout PLAN changed")
            else:
                data.save(path, plan)
            collector.run(args, prepared_run=(plan, tasks), adapter=plan["adapter"], world=world)
