"""Real fresh-TRAIN manifests around the immutable September28 native functions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "rl_resume_20260928"
sys.path.insert(0, str(LEGACY))
import probe_common as prior  # noqa: E402

c, rl, ROOT, WARM = prior.c, prior.rl, prior.ROOT, prior.WARM
read, sha = prior.read, prior.sha
implementation, adapter_binding = prior.implementation, prior.adapter_binding
audit_collection = prior.audit_collection
DATA = ROOT / "textcraft-fresh-train-20260928-001"
CURRICULUM_SHA = "76e387378c9a89b0d8e26180874db61ad40086721c6cfef9e3b78dd77f915e05"
GROUP_SHA = {
    "train": "07c410047a740b429912df9dbb0fb9c8cea3063b1e185c51f87ca645c5ab9d90",
    "diagnostic": "65e67e57063ee43d1c8c072988cca2081531b96aae4a47d582c2550f419c1c1a",
}


def load_dataset(directory: Path, phase: str) -> tuple[dict, list[dict]]:
    manifest = read(directory / "MANIFEST.json")
    expected = "train" if phase == "collect" else "diagnostic"
    if manifest.get("group") != expected:
        raise ValueError("collection requires training group; readout requires diagnostic group")
    if sha(directory / "tasks.jsonl") != manifest["tasks_sha256"]:
        raise ValueError("qualified original task bytes changed")
    rows = [json.loads(line) for line in (directory / "tasks.jsonl").read_text().splitlines()]
    if (
        manifest.get("split") != "train"
        or len(rows) != 8
        or len({r["id"] for r in rows}) != 8
        or [r["id"] for r in rows] != manifest["task_ids"]
        or any(not r["id"].startswith("textcraft_synth.train.") for r in rows)
    ):
        raise ValueError("exact eight official TRAIN identities required")
    if (
        manifest.get("world_seed") != 42
        or not manifest.get("world_sha256")
        or manifest.get("all_native_feasible") is not True
        or manifest.get("initial_context_qualified") is not True
    ):
        raise ValueError("CPU-qualified native world42 and initial context required")
    return manifest, rows


def jobs(rows, mode, phase, update):
    old = prior.jobs(rows, mode, phase, update)
    seed = 202609280100 + update * 10 if phase == "collect" else 202609280900
    return [dict(j, seed=seed + j["repeat"], condition=f"fresh_{mode}_{phase}") for j in old]


def dataset_binding(directory: Path) -> dict:
    directory = directory.resolve()
    if directory.parent != DATA or directory.name not in GROUP_SHA:
        raise ValueError("use a frozen September28 fresh-group directory")
    if sha(DATA / "MANIFEST.json") != CURRICULUM_SHA:
        raise ValueError("prospective curriculum/exclusion provenance changed")
    if sha(directory / "MANIFEST.json") != GROUP_SHA[directory.name]:
        raise ValueError("frozen group manifest changed")
    return dict(
        curriculum_manifest=str(DATA / "MANIFEST.json"), curriculum_manifest_sha256=CURRICULUM_SHA
    )


def source_pins(entrypoint: Path, mode: str) -> dict:
    pins = prior.source_pins(entrypoint, mode)
    for path in (Path(__file__).resolve(), HERE / entrypoint.name):
        pins[str(path)] = sha(path)
    return pins


def persist(path: Path, plan: dict) -> None:
    if path.exists():
        if read(path) != plan:
            raise ValueError("immutable fresh-study PLAN changed")
    else:
        c.save(path, plan)


def prepare_collection(args, entrypoint: Path):
    manifest, rows = load_dataset(args.dataset, args.phase)
    curriculum = dataset_binding(args.dataset)
    maximum = 2.5 if args.phase == "collect" else 1.5
    if not 0 < args.hours <= maximum or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("campaign output and declared collection/readout cap required")
    plan = dict(
        schema="textcraft-fresh-train-collection-20260928-v1",
        split="train",
        execution_mode=args.mode,
        phase=args.phase,
        update=args.update,
        dataset_group=manifest["group"],
        prepared=str(args.dataset.resolve()),
        tasks_sha256=manifest["tasks_sha256"],
        manifest_sha256=sha(args.dataset / "MANIFEST.json"),
        dataset_provenance=manifest,
        **curriculum,
        world_seed=42,
        world_sha256=manifest["world_sha256"],
        model=str(c.BASE),
        model_manifest_sha256=sha(c.BASE / "local-research-manifest.json"),
        adapter=adapter_binding(args.checkpoint),
        base_dtype="float16",
        lora_dtype="float32",
        jobs=jobs(rows, args.mode, args.phase, args.update),
        planned_episodes=32 if args.phase == "collect" else 16,
        profile="original",
        sampling=dict(temperature=0.5, top_p=1.0, top_k=0),
        max_global_calls=96,
        max_global_output_tokens=8192,
        max_new_tokens=256,
        input_plus_output_limit=8192,
        truncation=False,
        budget_seconds=args.hours * 3600,
        source_sha256=source_pins(entrypoint, args.mode),
        likelihood_target="Original sampled output_token_ids including errors/EOS; "
        "deterministic rewritten ingredient arguments are environment transitions only.",
        caveat="Official TRAIN split with prospectively disjoint training/diagnostic goals. "
        "Diagnostic is never fed to the optimizer. Shared world/item grammar limits independence; "
        "no official VAL/HOLDOUT goal is used. Initial context qualification "
        "is not full-trace fit.",
    )
    persist(args.output / "PLAN.json", plan)
    return plan, rows


def prepare_training(args):
    collection = read(args.collection / "PLAN.json")
    if (
        collection.get("schema") != "textcraft-fresh-train-collection-20260928-v1"
        or collection["phase"] != "collect"
        or collection["dataset_group"] != "train"
        or collection["planned_episodes"] != 32
        or collection["base_dtype"] != "float16"
    ):
        raise ValueError("fresh training-group FP16 collection required")
    dataset = Path(collection["prepared"])
    load_dataset(dataset, "collect")
    curriculum = dataset_binding(dataset)
    if sha(dataset / "MANIFEST.json") != collection["manifest_sha256"]:
        raise ValueError("training group qualification changed")
    if not 0 < args.hours <= 2 or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("campaign output and at most120minutes required")
    checkpoint, update = Path(collection["adapter"]["path"]), collection["update"]
    if adapter_binding(checkpoint) != collection["adapter"]:
        raise ValueError("behavior checkpoint changed")
    if update == 1:
        if checkpoint != WARM:
            raise ValueError("fresh first step must start at original publiccp23")
    else:
        state = read(checkpoint / "STATE.json")
        previous = checkpoint.parents[2]
        ancestor = read(previous / "PLAN.json")
        summary = read(previous / "SUMMARY.json")
        if (
            not summary.get("endpoint_usable")
            or summary["endpoint"] != str(checkpoint)
            or state["step"] != update - 1
            or state["plan_sha256"] != sha(previous / "PLAN.json")
            or ancestor["schema"] != "textcraft-fresh-train-rloo-20260928-v1"
            or ancestor["dataset_manifest_sha256"] != collection["manifest_sha256"]
            or ancestor["execution_mode"] != collection["execution_mode"]
        ):
            raise ValueError("fresh continuation must retain its actual dataset/interface lineage")
    plan = dict(
        schema="textcraft-fresh-train-rloo-20260928-v1",
        split="train",
        execution_mode=collection["execution_mode"],
        update=update,
        collection=str(args.collection.resolve()),
        collection_plan_sha256=sha(args.collection / "PLAN.json"),
        dataset=str(dataset),
        dataset_manifest_sha256=collection["manifest_sha256"],
        dataset_tasks_sha256=collection["tasks_sha256"],
        **curriculum,
        adapter=collection["adapter"],
        base_dtype="float16",
        lora_dtype="float32",
        learning_rate=2e-5,
        weight_decay=0,
        clip_grad_norm=1.0,
        optimizer="fresh_AdamW" if update == 1 else "restore_previous_AdamW",
        budget_seconds=args.hours * 3600,
        maximum_new_optimizer_updates=1,
        max_replay_gap=0.25,
        mean_replay_gap=0.025,
        objective="-sum_i[RLOO terminal-success advantage_i * "
        "sum_original_emitted_tokens log pi_T0.5]/32",
        original_estimator=str(LEGACY / "train.py"),
        action_target="Exact original sampled token IDs, never executed binder ingredients",
        source_sha256=source_pins(LEGACY / "train.py", collection["execution_mode"]),
        continuation="At most4 numbered fresh on-policy batches; restore prior committed Adam. "
        "No informative reward group or a failed numerical seam stops this arm.",
        diagnostic_exposure="Diagnostic TRAIN group is readout-only and not optimization data.",
        caveat="One gradient step per fresh batch. Equal optimizer steps/caps do not match "
        "realized token credit or trajectories across interfaces. "
        "Official held-out goals untouched.",
    )
    persist(args.output / "PLAN.json", plan)
    return plan


def load_legacy(filename: str):
    """Inject only a privately loaded module's dependencies; pinned source stays unchanged."""
    spec = importlib.util.spec_from_file_location(
        "fresh_private_" + filename[:-3], LEGACY / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.p = sys.modules[__name__]
    return module
