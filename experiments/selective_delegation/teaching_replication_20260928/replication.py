"""Bounded preparation/readout binding around the frozen teaching-order implementation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
ORIGINAL = ROOT / "textcraft-teaching-order-20260928-001"
OUTPUT = ROOT / "textcraft-teaching-replication-20260928-001"
FROZEN = HERE.parent / "teaching_order_20260928"
PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)
FROZEN_SHAS = {
    "prepare.py": "8e6b8b3f2cded649cc43a4701aac09c72533bbac20e3f40cbbda03ed8621891e",
    "train.py": "4d8463326b2fa910696a4c629c946b98277f0151111b4d2c0079c5ee4da25f0b",
    "readout.py": "d488463674c76ab1c6fe20f5079ff867a44f8b9b46d249490e4aab28497fb235",
}
MODES = ("stable_visible", "random_visible")
PAIRED_FIELDS = (
    "tasks_sha256",
    "manifest_sha256",
    "world_sha256",
    "world_seed",
    "training_seed",
    "sampling",
    "model_manifest_sha256",
    "max_global_calls",
    "max_global_output_tokens",
    "max_new_tokens",
    "input_plus_output_limit",
    "truncation",
    "budget_seconds",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, "immutable artifact differs: " + str(path))
        return
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def schedule() -> list[dict]:
    return [
        dict(mode=mode, seed=seed, panel=panel, world=world)
        for seed, panel in ((2026092291, 0), (2026092208, 1))
        for world in (42, 50)
        for mode in MODES
    ]


def baseline(seed: int, panel: int, world: int, teacher: str) -> Path:
    require(seed in (2026092208, 2026092291), "unsupported fit seed")
    require(teacher in ("known", "discovery"), "explicit behavioral baseline required")
    label = "original" if seed == 2026092208 else "2291"
    prefix = "corrected-" if teacher == "known" else ""
    return ROOT / f"textcraft-breadth-p{panel:02d}-w{world}-s{label}-{prefix}raw-001"


def training(mode: str, seed: int, root: Path = OUTPUT) -> Path:
    require(mode in MODES and seed in (2026092208, 2026092291), "unsupported training cell")
    return (ORIGINAL if seed == 2026092208 else root) / f"train-{mode}-seed{seed}"


def output(cell: dict, root: Path = OUTPUT) -> Path:
    return root / (f"eval-{cell['mode']}-s{cell['seed']}-p{cell['panel']:02d}-w{cell['world']}-raw")


def check_binding(binding: dict, expected: dict) -> None:
    require(binding["path"] == expected["checkpoint"], "wrong fixed checkpoint")
    require(
        all(binding["state"].get(k) == v for k, v in dict(step=23, epoch=1, cursor=0).items()),
        "complete checkpoint23 required",
    )
    require(binding["training_plan_sha256"] == expected["plan_sha256"], "training PLAN differs")
    require(binding["training_rows_sha256"] == expected["rows_sha256"], "teacher rows differ")


def original_module():
    for name, digest in FROZEN_SHAS.items():
        require(sha(FROZEN / name) == digest, "frozen teaching source changed")
    sys.path.insert(0, str(FROZEN))
    spec = importlib.util.spec_from_file_location(
        "replication_frozen_readout", FROZEN / "readout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(
        Path(module.prepare.__file__).resolve() == FROZEN / "prepare.py", "mixed teacher imports"
    )
    return module


def expected_training(mode: str, seed: int, root: Path) -> dict:
    prepared = read(root / "PREPARATION.json")
    result = prepared["training"][f"{mode}:{seed}"]
    require(
        sha(Path(result["directory"]) / "PLAN.json") == result["plan_sha256"],
        "accepted training PLAN changed",
    )
    return result


def endpoint(mode: str, seed: int, root: Path) -> dict:
    expected = expected_training(mode, seed, root)
    _, shared, _ = original_module().load_runtime("raw")
    binding = shared.endpoint(
        Path(expected["checkpoint"]),
        training_plan_sha256=expected["plan_sha256"],
        rows_sha256=expected["rows_sha256"],
    )
    check_binding(binding, expected)
    return binding


def build(cell: dict, root: Path = OUTPUT) -> tuple:
    require(cell in schedule(), "cell outside fixed replication")
    expected = expected_training(cell["mode"], cell["seed"], root)
    args = argparse.Namespace(
        **cell, root=ORIGINAL if cell["seed"] == 2026092208 else root, execution="raw"
    )
    old = original_module()
    collector, world, _, plan, tasks, binding = old.inputs(args)
    check_binding(binding, expected)
    reference = read(
        baseline(cell["seed"], cell["panel"], cell["world"], "discovery") / "PLAN.json"
    )
    require(
        all(plan[k] == reference[k] for k in PAIRED_FIELDS), "replication changed paired inputs"
    )
    plan = copy.deepcopy(plan)
    plan["replication"] = dict(
        schema="teaching-order-replication-20260928-v1",
        **cell,
        preparation_sha256=sha(root / "PREPARATION.json"),
        behavioral_baselines={
            t: str(baseline(cell["seed"], cell["panel"], cell["world"], t))
            for t in ("known", "discovery")
        },
    )
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__))
    return collector, world, output(cell, root), plan, tasks, binding


def run_readout(cell: dict, root: Path, *, prepare_only=False, audit=False) -> None:
    collector, world, directory, plan, tasks, binding = build(cell, root)
    save(directory / "PLAN.json", plan)
    if audit:
        import analyze_textcraft_profiles as profiles
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            collector.BASE, local_files_only=True, trust_remote_code=False
        )
        result = profiles.audit.analyze(directory, tokenizer, world=world)
        result.pop("paired", None)
        result.pop("depth_strata", None)
        save(directory / "NATIVE-AUDIT.json", result)
        return
    collector.run(
        argparse.Namespace(output=directory, prepare_only=prepare_only),
        prepared_run=(plan, tasks),
        adapter=binding,
        world=world,
    )


def prepare(root: Path = OUTPUT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    for relative in [
        "SUMMARY.json",
        *(
            f"{mode}/{name}"
            for mode in MODES
            for name in ("MANIFEST.json", "rows.jsonl", "tasks.jsonl")
        ),
    ]:
        source, target = ORIGINAL / relative, root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha(target) == sha(source), "replication teacher copy changed")
        else:
            shutil.copy2(source, target)
    for mode in MODES:
        subprocess.run(
            [
                PYTHON,
                str(FROZEN / "train.py"),
                "--root",
                str(root),
                "--mode",
                mode,
                "--seed",
                "2026092291",
                "--prepare-only",
                "--resume",
            ],
            check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    study = dict(schema="teaching-order-replication-inputs-20260928-v1", training={}, cells=[])
    for mode in MODES:
        for seed in (2026092208, 2026092291):
            directory = training(mode, seed, root)
            plan = read(directory / "PLAN.json")
            require(
                plan["seed"] == seed and plan["planned_updates"] == 23, "fixed SFT dose differs"
            )
            require(
                plan["rows_sha256"] == read(ORIGINAL / mode / "MANIFEST.json")["rows_sha256"],
                "training data changed",
            )
            study["training"][f"{mode}:{seed}"] = dict(
                directory=str(directory),
                checkpoint=str(directory / "checkpoint-0023"),
                plan_sha256=sha(directory / "PLAN.json"),
                rows_sha256=plan["rows_sha256"],
                seed=seed,
            )
    for cell in schedule():
        refs = {
            t: baseline(cell["seed"], cell["panel"], cell["world"], t)
            for t in ("known", "discovery")
        }
        plans = {t: read(p / "PLAN.json") for t, p in refs.items()}
        require(
            all(plans["known"][k] == plans["discovery"][k] for k in PAIRED_FIELDS),
            "baseline paired inputs differ",
        )
        for teacher, directory in refs.items():
            audit, plan = read(directory / "NATIVE-AUDIT.json"), plans[teacher]
            require(
                audit["sha256"][str(directory / "PLAN.json")] == sha(directory / "PLAN.json"),
                "baseline audited PLAN changed",
            )
            require(
                plan["training_seed"] == cell["seed"] and len(plan["jobs"]) == 16,
                "wrong baseline fit seed or denominator",
            )
        study["cells"].append(
            dict(
                **cell,
                output=str(output(cell, root)),
                endpoint_pending=cell["seed"] == 2026092291,
                baselines={t: str(p) for t, p in refs.items()},
                baseline_plan_sha256={t: sha(p / "PLAN.json") for t, p in refs.items()},
            )
        )
    save(root / "PREPARATION.json", study)
    for cell in schedule():
        if cell["seed"] == 2026092208:
            run_readout(cell, root, prepare_only=True)
    return study


def descriptors(root: Path = OUTPUT) -> dict:
    study = read(root / "PREPARATION.json")
    paths = {
        Path(__file__),
        HERE / "report.py",
        HERE / "fixture.py",
        root / "PREPARATION.json",
        root / "CPU-FIXTURE.json",
        *[FROZEN / name for name in FROZEN_SHAS],
    }
    for relative in [
        "SUMMARY.json",
        *(f"{m}/{n}" for m in MODES for n in ("MANIFEST.json", "rows.jsonl", "tasks.jsonl")),
    ]:
        paths.add(root / relative)
    for info in study["training"].values():
        paths.add(Path(info["directory"]) / "PLAN.json")
    for cell in study["cells"]:
        for directory in cell["baselines"].values():
            paths.update(Path(directory) / name for name in ("PLAN.json", "NATIVE-AUDIT.json"))
        if not cell["endpoint_pending"]:
            paths.add(Path(cell["output"]) / "PLAN.json")
    pins = {str(path.resolve()): sha(path) for path in sorted(paths)}
    for cell in study["cells"]:
        for directory in cell["baselines"].values():
            pins.update(read(Path(directory) / "PLAN.json")["source_sha256"])
    pair_source = HERE.parent / "inventory_bottleneck_20260928/compare.py"
    pins[str(pair_source)] = sha(pair_source)
    for info in study["training"].values():
        plan = read(Path(info["directory"]) / "PLAN.json")
        recipe = ROOT / "source-048-textcraft-action-sft/train_textcraft_sft.py"
        require(sha(recipe) == plan["source_sha256"], "SFT recipe changed")
        pins[str(recipe)] = plan["source_sha256"]
        optimizer_recipe = recipe.with_name("train_planner.py")
        require(sha(optimizer_recipe) == plan["training_recipe_sha256"], "optimizer recipe changed")
        pins[str(optimizer_recipe)] = plan["training_recipe_sha256"]
    jobs = []

    def add(name, argv, cap, directory=None):
        job = dict(name=name, argv=argv, cap_seconds=cap, pins=pins)
        if directory is not None:
            job["output"] = str(directory)
        jobs.append(job)

    for mode in MODES:
        add(
            "replication-" + mode + "-train2291",
            [
                PYTHON,
                str(FROZEN / "train.py"),
                "--root",
                str(root),
                "--mode",
                mode,
                "--seed",
                "2026092291",
                "--resume",
            ],
            2100,
            training(mode, 2026092291, root),
        )
        add(
            "replication-" + mode + "-endpoint2291",
            [
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=",
                PYTHON,
                str(Path(__file__)),
                "--root",
                str(root),
                "--mode",
                mode,
                "--seed",
                "2026092291",
                "--endpoint-audit",
            ],
            300,
        )
    for cell in schedule():
        name = f"replication-{cell['mode']}-s{cell['seed']}-p{cell['panel']:02d}-w{cell['world']}"
        argv = [
            PYTHON,
            str(Path(__file__)),
            "--root",
            str(root),
            "--mode",
            cell["mode"],
            "--seed",
            str(cell["seed"]),
            "--panel",
            str(cell["panel"]),
            "--world",
            str(cell["world"]),
        ]
        add(name, argv, 3000, output(cell, root))
        add(name + "-audit", ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *argv, "--audit"], 600)
    add(
        "replication-paired-report",
        [
            "/usr/bin/env",
            "CUDA_VISIBLE_DEVICES=",
            PYTHON,
            str(HERE / "report.py"),
            "--replication",
            "--root",
            str(root),
            "--output",
            str(root / "REPLICATION-REPORT.json"),
        ],
        300,
    )
    result = dict(
        schema="teaching-replication-generic-jobs-20260928-v1",
        jobs=jobs,
        scientific_stages=10,
        new_attempts=128,
        expected_hours=[2.2, 3.5],
        cap_sum_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
        dependency="Every second-seed readout authenticates real complete checkpoint23; "
        "missing/failed endpoints cannot be silently replaced. Explicit CPU endpoint audits "
        "precede readouts, which independently recheck the endpoint. Original-seed actors "
        "are fixed completed checkpoints. No report selects a best arm or checkpoint.",
    )
    save(root / "PREPARED-JOBS.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--descriptors", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int, choices=(2026092208, 2026092291))
    parser.add_argument("--panel", type=int, choices=(0, 1))
    parser.add_argument("--world", type=int, choices=(42, 50))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--endpoint-audit", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root)
    elif args.descriptors:
        print(
            json.dumps({k: v for k, v in descriptors(args.root).items() if k != "jobs"}, indent=2)
        )
    elif args.endpoint_audit:
        binding = endpoint(args.mode, args.seed, args.root)
        save(
            training(args.mode, args.seed, args.root) / "ENDPOINT-AUDIT.json",
            dict(passed=True, binding=binding),
        )
    else:
        run_readout(
            dict(mode=args.mode, seed=args.seed, panel=args.panel, world=args.world),
            args.root,
            prepare_only=args.prepare_only,
            audit=args.audit,
        )
