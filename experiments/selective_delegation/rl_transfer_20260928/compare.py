"""Six fixed diagnostic-B cells: reused warm controls and the two-by-two RL transfer matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import transfer_common as t


def load_cell(directory: Path, actor: str, mode: str):
    summary = directory / "SUMMARY.json"
    if not summary.exists() or not t.f.read(summary).get("complete"):
        return dict(available=False, path=str(directory), reason="incomplete_or_absent"), {}
    if t.f.read(directory / "PLAN.json").get("fixture_only"):
        raise ValueError("scripted fixture is not a scientific readout")
    old = t.f.load_legacy("compare.py")
    plan, values, stats = old.load_cell(directory)
    _, tasks = t.f.load_dataset(t.f.DATA / "diagnostic", "readout")
    if (
        plan["schema"]
        != ("textcraft-fresh-train-collection-20260928-v1" if actor == "warm" else t.SCHEMA)
        or plan["dataset_group"] != "diagnostic"
        or plan["manifest_sha256"] != t.f.GROUP_SHA["diagnostic"]
        or plan["execution_mode"] != mode
        or plan["jobs"] != t.f.jobs(tasks, mode, "readout", 1)
    ):
        raise ValueError("actual frozen diagnostic role/interface/task-seed pairing required")
    expected_adapter = (
        t.f.adapter_binding(t.f.WARM) if actor == "warm" else t.admit_endpoint(actor)["adapter"]
    )
    if plan["adapter"] != expected_adapter:
        raise ValueError("readout used a different actor")
    contract = dict(
        tasks_sha256=t.f.sha(t.f.DATA / "diagnostic/tasks.jsonl"),
        model_manifest_sha256=t.f.sha(t.f.c.BASE / "local-research-manifest.json"),
        world_seed=42,
        world_sha256=t.f.read(t.f.DATA / "diagnostic/MANIFEST.json")["world_sha256"],
        base_dtype="float16",
        lora_dtype="float32",
        profile="original",
        sampling=dict(temperature=0.5, top_p=1.0, top_k=0),
        max_global_calls=96,
        max_global_output_tokens=8192,
        max_new_tokens=256,
        input_plus_output_limit=8192,
        truncation=False,
        planned_episodes=16,
    )
    if any(plan.get(key) != value for key, value in contract.items()):
        raise ValueError("diagnostic scientific contract changed")
    if actor != "warm" and plan["actor_lineage"] != t.admit_endpoint(actor):
        raise ValueError("trained actor lineage changed")
    rows = [
        t.f.read(directory / "episodes" / (job["episode_id"] + ".json")) for job in plan["jobs"]
    ]
    stats.update(
        available=True,
        training_execution_mode=None if actor == "warm" else actor,
        execution_mode=mode,
        termination_counts=dict(Counter(row["status"] for row in rows)),
        failed_finished_episodes=sum(
            row["status"] == "finished" and row["native_score"] == 0 for row in rows
        ),
    )
    return stats, values


def contrast(outcomes: dict, weights: dict, keys: list) -> dict:
    deltas = {
        key: sum(weight * outcomes[name][key] for name, weight in weights.items())
        if all(outcomes[name].get(key) is not None for name in weights)
        else None
        for key in keys
    }
    return t.f.load_legacy("compare.py").paired(deltas)


def analyze(study: Path, *, warm_study: Path = t.WARM_STUDY) -> dict:
    t.f.dataset_binding(t.f.DATA / "diagnostic")
    _, tasks = t.f.load_dataset(t.f.DATA / "diagnostic", "readout")
    keys = [(task["id"], repeat) for task in tasks for repeat in (0, 1)]
    cells, outcomes = {}, {}
    for actor in ("warm", "raw", "binder"):
        for mode in ("raw", "binder"):
            directory = (
                warm_study / mode / "readout-warm"
                if actor == "warm"
                else study / f"actor-{actor}" / f"readout-{mode}"
            )
            name = f"{actor}_actor_{mode}_execution"
            cells[name], outcomes[name] = load_cell(directory, actor, mode)
    gains = {
        f"{actor}_actor_{mode}_execution": contrast(
            outcomes,
            {f"{actor}_actor_{mode}_execution": 1, f"warm_actor_{mode}_execution": -1},
            keys,
        )
        for actor in ("raw", "binder")
        for mode in ("raw", "binder")
    }
    execution_gains = {
        actor: contrast(
            outcomes,
            {f"{actor}_actor_binder_execution": 1, f"{actor}_actor_raw_execution": -1},
            keys,
        )
        for actor in ("warm", "raw", "binder")
    }
    actor_gains = {
        mode: contrast(
            outcomes, {f"binder_actor_{mode}_execution": 1, f"raw_actor_{mode}_execution": -1}, keys
        )
        for mode in ("raw", "binder")
    }
    return dict(
        schema="textcraft-familiar-rl-fresh-B-paired-transfer-analysis-20260928-v1",
        cells=cells,
        rl_minus_matching_warm=gains,
        binder_minus_raw_execution=execution_gains,
        binder_trained_minus_raw_trained_actor=actor_gains,
        training_interface_by_execution_interaction=contrast(
            outcomes,
            dict(
                binder_actor_binder_execution=1,
                binder_actor_raw_execution=-1,
                raw_actor_binder_execution=-1,
                raw_actor_raw_execution=1,
            ),
            keys,
        ),
        diagnostic_manifest_sha256=t.f.GROUP_SHA["diagnostic"],
        independent_native_replay="Every available cell has complete native replay and receipt "
        "hash verification; unknown/incomplete cells produce no effect estimate.",
        scope="Eight fixed official TRAIN diagnostic clusters; roots disjoint from prior SFT, "
        "familiar RL and newA. Shared recipes/world, adaptive campaign and repeated diagnostic "
        "exposure limit generalization. No checkpoint or task selected from B outcomes. "
        "Intervals are descriptive task-cluster bootstrap, not multiplicity-adjusted confirmation.",
        caveat="RL actors differ in sampled training trajectories and credited-token exposure. "
        "Transfer versus warm is not evidence of extra-SFT superiority or recursive decomposition.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=t.STUDY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.study)
    t.f.c.save(args.output, report)
    print(json.dumps(report["rl_minus_matching_warm"], indent=2))
