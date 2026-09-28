"""Own-interface compact learning gain, paired with the full+binder first-step gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import compact_common as c

legacy = c.module(c.f.LEGACY / "compare.py", "compact_rl_private_comparison")


def endpoint(directory: Path) -> str | None:
    path = directory / "SUMMARY.json"
    result = c.read(path) if path.exists() else {}
    return result.get("endpoint") if result.get("endpoint_usable") else None


def load(directory: Path, mode: str, expected_adapter: str | None):
    required = ("PLAN.json", "SUMMARY.json", "NATIVE-AUDIT.json")
    if any(not (directory / name).exists() for name in required):
        return dict(available=False, path=str(directory), reason="missing completed readout"), {}
    summary = c.read(directory / "SUMMARY.json")
    if not summary.get("complete") or summary.get("failure"):
        return dict(available=False, path=str(directory), reason="incomplete or failed readout"), {}
    plan, values, stats = legacy.load_cell(directory)
    schema = (
        "textcraft-compact-rl-collection-20260928-v1"
        if mode == c.MODE
        else "textcraft-fresh-train-collection-20260928-v1"
    )
    _, tasks = c.f.load_dataset(c.f.DATA / "diagnostic", "readout")
    fixed = c.jobs(tasks, "readout")

    def identity(jobs):
        return [(j["task_id"], j["repeat"], j["seed"]) for j in jobs]

    if (
        plan["schema"] != schema
        or plan["execution_mode"] != mode
        or plan["dataset_group"] != "diagnostic"
        or plan["manifest_sha256"] != c.f.GROUP_SHA["diagnostic"]
        or identity(plan["jobs"]) != identity(fixed)
        or not expected_adapter
        or plan["adapter"]["path"] != expected_adapter
    ):
        raise ValueError("diagnostic interface, fixed pairing or actual actor lineage changed")
    failed_finishes = early_finishes = first_finishes = 0
    for job in plan["jobs"]:
        episode = c.read(directory / "episodes" / (job["episode_id"] + ".json"))
        failed_finish = episode["native_score"] == 0 and episode["status"] == "finished"
        failed_finishes += int(failed_finish)
        early_finishes += int(failed_finish and episode["global_calls"] <= 4)
        first_finishes += int(failed_finish and episode["global_calls"] == 1)
    return dict(
        **stats,
        available=True,
        plan=plan,
        unsuccessful_explicit_finishes=failed_finishes,
        unsuccessful_finishes_within_four_calls=early_finishes,
        unsuccessful_first_call_finishes=first_finishes,
    ), values


def difference(before: dict, after: dict, keys: list) -> dict:
    return {
        key: after[key] - before[key]
        if before.get(key) is not None and after.get(key) is not None
        else None
        for key in keys
    }


def analyze() -> dict:
    _, tasks = c.f.load_dataset(c.f.DATA / "diagnostic", "readout")
    keys = [(task["id"], repeat) for task in tasks for repeat in (0, 1)]
    cells, outcomes, training, gains = {}, {}, {}, {}
    for mode, root, warm in (
        (c.MODE, c.STUDY, c.WARM),
        ("binder", c.BASELINE / "binder", c.f.WARM),
    ):
        for actor in ("warm", "0001"):
            name = f"{mode}_{actor}"
            expected = str(warm) if actor == "warm" else endpoint(root / "train-0001")
            cells[name], outcomes[name] = load(root / f"readout-{actor}", mode, expected)
        gains[mode] = difference(outcomes[f"{mode}_warm"], outcomes[f"{mode}_0001"], keys)
        training[mode] = {
            name: c.read(root / "train-0001" / name)
            for name in ("SUMMARY.json", "UPDATE.json", "ADMISSION.json", "CONDITIONAL-SKIP.json")
            if (root / "train-0001" / name).exists()
        }
    available = [cell["plan"] for cell in cells.values() if cell["available"]]
    for plan in available[1:]:
        for name in (
            "tasks_sha256",
            "world_sha256",
            "model_manifest_sha256",
            "base_dtype",
            "lora_dtype",
            "sampling",
            "max_global_calls",
            "max_global_output_tokens",
            "max_new_tokens",
            "input_plus_output_limit",
            "truncation",
        ):
            if plan[name] != available[0][name]:
                raise ValueError("paired diagnostic scientific contract changed: " + name)
    interaction = difference(gains["binder"], gains[c.MODE], keys)
    return dict(
        schema="textcraft-compact-first-step-learning-comparison-20260928-v1",
        cells=cells,
        training=training,
        diagnostic_manifest_sha256=c.f.GROUP_SHA["diagnostic"],
        within_interface_native_success_gain={mode: legacy.paired(v) for mode, v in gains.items()},
        compact_minus_full_binder_learning_gain=legacy.paired(interaction),
        primary="Native terminal success. An unavailable/flat-reward endpoint stays unknown; "
        "it is never replaced by the warm actor or relabeled a failed episode.",
        uncertainty="Eight official TRAIN task clusters, two fixed execution seeds, one world. "
        "Cluster bootstrap is descriptive and does not establish unseen-task generalization.",
        limitation="Compact/full SFT have matched23 updates, not equal token exposure "
        "(6396 versus8820 target tokens). Compact rejects unobserved recipes whereas the full "
        "binder can fall back to native execution. Warm actors, schema/history and native "
        "transition semantics all differ. A difference in learning gains is not an isolated "
        "schema effect; raw cross-interface scores are not evidence of RL benefit. "
        "Realized RL trajectory/token doses differ; no RLM-recursion claim.",
        early_finish_definition="Unsuccessful explicit root finish, separately <=4 or exactly1 "
        "physical model calls. Entropy/diversity are descriptive behavior diagnostics.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze()
    c.f.persist(args.output, report)
    print(json.dumps(report["within_interface_native_success_gain"], indent=2))
