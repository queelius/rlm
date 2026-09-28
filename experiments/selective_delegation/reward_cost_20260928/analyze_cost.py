"""Success-primary first-step comparison, retaining cheap-stopping/exploration diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import cost_reward as r


def existing_analysis():
    spec = importlib.util.spec_from_file_location(
        "cost_private_fresh_compare", r.f.HERE / "compare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def behavior(episode: dict, calls: dict) -> dict:
    query_items, actions = set(), Counter()
    first_action = None
    for index, cid in enumerate(episode["call_ids"]):
        try:
            action = r.f.c.bridge.parse_action(calls[cid]["text"])
        except ValueError:
            continue
        name = action["action"]
        actions[name] += 1
        if index == 0:
            first_action = name
        if name == "get_info":
            query_items.update(action["items"])
    failed = episode["native_score"] == 0
    finished = episode["status"] == "finished"
    return dict(
        **r.reward_row(episode),
        status=episode["status"],
        unsuccessful_explicit_finish=int(failed and finished),
        unsuccessful_finish_within_four_calls=int(
            failed and finished and episode["global_calls"] <= 4
        ),
        unsuccessful_first_call_finish=int(failed and finished and first_action == "finish"),
        get_info_calls=actions["get_info"],
        distinct_requested_info_items=len(query_items),
        actions=dict(actions),
    )


def load(directory: Path, mode: str, expected_adapter: str | None):
    stats, outcomes = existing_analysis().load(directory, mode)
    if not stats["available"]:
        return stats, outcomes
    if not expected_adapter or stats["adapter"]["path"] != expected_adapter:
        raise ValueError("readout adapter differs from its prespecified warm or first-step actor")
    episodes = {}
    for job in stats["plan"]["jobs"]:
        episode = r.f.read(directory / "episodes" / (job["episode_id"] + ".json"))
        calls = {
            cid: r.f.read(directory / "calls" / (cid + ".json")) for cid in episode["call_ids"]
        }
        episodes[job["episode_id"]] = behavior(episode, calls)
    totals = {
        key: sum(e[key] for e in episodes.values())
        for key in (
            "physical_calls",
            "charged_errors",
            "unsuccessful_explicit_finish",
            "unsuccessful_finish_within_four_calls",
            "unsuccessful_first_call_finish",
            "get_info_calls",
            "distinct_requested_info_items",
        )
    }
    totals["distinct_requested_info_items_scope"] = (
        "Sum of within-episode distinct items, not global"
    )
    totals["mean_composite_reward_secondary"] = sum(
        e["composite_reward"] for e in episodes.values()
    ) / len(episodes)
    stats.update(behavior_totals=totals, episode_behavior=episodes)
    return stats, outcomes


def endpoint(directory: Path) -> str | None:
    path = directory / "SUMMARY.json"
    summary = r.f.read(path) if path.exists() else {}
    return summary.get("endpoint") if summary.get("endpoint_usable") else None


def analyze() -> dict:
    old = existing_analysis()
    _, tasks = r.f.load_dataset(r.f.DATA / "diagnostic", "readout")
    keys = [(t["id"], repeat) for t in tasks for repeat in (0, 1)]
    cells, outcomes, contrasts, training, flags = {}, {}, {}, {}, {}
    for mode in ("raw", "binder"):
        directories = {
            "warm": r.FRESH_STUDY / mode / "readout-warm",
            "terminal": r.FRESH_STUDY / mode / "readout-0001",
            "cost": r.STUDY / mode / "readout-0001",
        }
        for actor, directory in directories.items():
            expected = (
                str(r.f.WARM)
                if actor == "warm"
                else endpoint((r.STUDY if actor == "cost" else r.FRESH_STUDY) / mode / "train-0001")
            )
            name = f"{mode}_{actor}"
            cells[name], outcomes[name] = load(directory, mode, expected)
        contrasts[mode] = {
            f"cost_minus_{actor}": old.compare_values(
                outcomes[f"{mode}_{actor}"], outcomes[f"{mode}_cost"], keys
            )
            for actor in ("terminal", "warm")
        }
        # This is a prespecified warning, not a checkpoint selection gate or significance test.
        flags[mode] = {}
        for actor in ("terminal", "warm"):
            cost, reference = cells[f"{mode}_cost"], cells[f"{mode}_{actor}"]
            if not cost["available"] or not reference["available"]:
                flags[mode][actor] = dict(assessable=False, reason="missing complete paired cell")
                continue
            left, right = reference["behavior_totals"], cost["behavior_totals"]
            no_success_gain = cost["successes"] <= reference["successes"]
            cheaper = right["charged_errors"] < left["charged_errors"]
            early = (
                right["unsuccessful_finish_within_four_calls"]
                > left["unsuccessful_finish_within_four_calls"]
            )
            info_reduced = right["get_info_calls"] < left["get_info_calls"]
            flags[mode][actor] = dict(
                assessable=True,
                native_success_gain=cost["successes"] - reference["successes"],
                charged_error_delta=right["charged_errors"] - left["charged_errors"],
                failed_early_finish_delta=right["unsuccessful_finish_within_four_calls"]
                - left["unsuccessful_finish_within_four_calls"],
                get_info_call_delta=right["get_info_calls"] - left["get_info_calls"],
                cheap_stopping_retire_signal=no_success_gain
                and cheaper
                and (early or info_reduced),
                lower_errors_alone_do_not_promote=True,
            )
        training[mode] = {}
        for actor, study in (("cost", r.STUDY), ("terminal", r.FRESH_STUDY)):
            directory = study / mode / "train-0001"
            training[mode][actor] = {
                filename: r.f.read(directory / filename)
                for filename in (
                    "SUMMARY.json",
                    "ADMISSION.json",
                    "REWARD-TABLE.json",
                    "CONDITIONAL-SKIP.json",
                )
                if (directory / filename).exists()
            }
    available = [c["plan"] for c in cells.values() if c["available"]]
    for plan in available[1:]:
        for key in (
            "tasks_sha256",
            "model_manifest_sha256",
            "base_dtype",
            "sampling",
            "max_global_calls",
            "max_global_output_tokens",
            "max_new_tokens",
            "input_plus_output_limit",
            "world_sha256",
        ):
            if plan[key] != available[0][key]:
                raise ValueError("paired diagnostic scientific contract changed: " + key)
    return dict(
        schema="textcraft-error-cost-first-step-comparison-20260928-v1",
        diagnostic_manifest_sha256=r.f.GROUP_SHA["diagnostic"],
        cells=cells,
        paired_native_success=contrasts,
        training=training,
        retirement_diagnostics=flags,
        primary="Native success only; composite reward is a secondary multiobjective metric",
        early_finish_definition="Native failure with explicit root finish within <=4 model calls; "
        "also separately report any failed finish and failed first-call finish",
        exploration_caveat="get_info calls and within-episode queried item diversity are proxies. "
        "Higher counts alone do not establish useful exploration or improved learning.",
        interpretation="Retire/revise if cheaper failed behavior has no success/exploration "
        "benefit. The cheap-stopping flag aids detection but does not replace trajectory review. "
        "One small groupB readout is diagnostic, not conclusive held-out efficacy evidence. "
        "Missing or skipped terminal training is unknown, never a warm replacement control.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze()
    r.f.c.save(args.output, report)
    print(json.dumps(report["paired_native_success"], indent=2))
