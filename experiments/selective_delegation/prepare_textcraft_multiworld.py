"""Outcome-blind world44/45/46 extension using the reviewed world43 constructor."""

import argparse
import json
import sys
from pathlib import Path

import prepare_textcraft_world43 as prior

WORLD_SEEDS = (44, 45, 46)


def world(seed):
    prior.bridge.load_world()
    generator = sys.modules["pinned_textcraft_synth_generator_d9c5857d"]
    result = generator.SynthRecipeDatabase()
    result.generate_all_recipes(seed=seed, items_per_domain_tier=25)
    return result


def prepare(root, output):
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError(output)
    original = root / "textcraft-inputs-001/tasks.jsonl"
    if prior.inputs.sha(original) != prior.TASK_SHA:
        raise ValueError("original eight roots changed")
    tasks = list(map(json.loads, original.read_text().splitlines()))
    output.mkdir(parents=True)
    prior.inputs.save(
        output / "SELECTION.json",
        dict(
            world_seeds=WORLD_SEEDS,
            task_ids=[t["id"] for t in tasks],
            original_tasks_sha256=prior.TASK_SHA,
            model_outcomes_used=False,
            rule="All original eight roots in each of44/45/46; no replacement/filter",
            frozen_before_native_replay=True,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        prior.BASE, local_files_only=True, trust_remote_code=False
    )
    manifests = []
    for seed in WORLD_SEEDS:
        native = world(seed)
        digest = prior.inventory.digest(prior.inventory.snapshot(native))
        destination = output / f"world{seed}"
        destination.mkdir()
        changed, audits = [], []
        for task in tasks:
            audit = dict(task_id=task["id"])
            try:
                item = dict(prior.construct(task, native), derived_world_seed=seed)
                audit.update(
                    statistics=prior.statistics(item, native),
                    qualification=prior.qualify(item, native, tokenizer),
                )
            except Exception as exc:
                item = dict(task, construction_failure=f"{type(exc).__name__}: {exc}")
                audit["failure"] = item["construction_failure"]
            changed.append(item)
            audits.append(audit)
        with (destination / "tasks.jsonl").open("x") as stream:
            for item in changed:
                stream.write(json.dumps(item) + "\n")
        manifest = dict(
            world_seed=seed,
            world_sha256=digest,
            tasks_sha256=prior.inputs.sha(destination / "tasks.jsonl"),
            task_count=8,
            audits=audits,
            ready=all(a.get("qualification", {}).get("native_score") == 1 for a in audits),
        )
        prior.inputs.save(destination / "MANIFEST.json", manifest)
        manifests.append(manifest)
    manifest = dict(
        schema="textcraft-frozen-multiworld-panel-v1",
        worlds=manifests,
        ready=all(m["ready"] for m in manifests),
        selection_sha256=prior.inputs.sha(output / "SELECTION.json"),
        source_sha256={
            str(Path(m.__file__).resolve()): prior.inputs.sha(Path(m.__file__))
            for m in (prior, prior.bridge, prior.inventory)
        },
        driver_sha256=prior.inputs.sha(Path(__file__)),
        trusted_source=prior.bridge.trusted_provenance(),
        caveat="Derived recipe worlds, same exposed roots and grammar; conservative "
        "overprovisioned inventories vary with world. Not independent domains or "
        "matched public observations; no model outcomes used.",
    )
    prior.inputs.save(output / "MANIFEST.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.output)
    print(json.dumps({"ready": result["ready"], "worlds": len(result["worlds"])}))
