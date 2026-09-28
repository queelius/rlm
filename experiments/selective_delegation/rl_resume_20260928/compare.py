"""Paired TRAIN readout of raw/binder learning gains; unknown outcomes stay unknown."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import probe_common as p


def paired(deltas: dict[tuple[str, int], float | None]) -> dict:
    grouped = defaultdict(list)
    for (task, _), delta in deltas.items():
        grouped[task].append(delta)
    known = [v for v in deltas.values() if v is not None]
    complete = len(known) == len(deltas) and bool(known)
    estimate = interval = None
    if complete:
        values = [mean(v) for v in grouped.values()]
        estimate = mean(values)
        rng = random.Random(20260928)
        sampled = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
        interval = [sampled[249], sampled[9749]]
    return dict(
        planned=len(deltas),
        known=len(known),
        unknown=len(deltas) - len(known),
        wins=sum(v > 0 for v in known),
        losses=sum(v < 0 for v in known),
        ties=sum(v == 0 for v in known),
        difference=estimate,
        task_cluster_ci95=interval,
        task_differences={
            k: mean(v) if all(x is not None for x in v) else None for k, v in grouped.items()
        },
    )


def load_cell(directory: Path) -> tuple[dict, dict, dict]:
    plan = p.read(directory / "PLAN.json")
    if plan["phase"] != "readout" or plan["split"] != "train":
        raise ValueError("paired diagnostic must remain on TRAIN readout seeds")
    summary = p.read(directory / "SUMMARY.json")
    if not summary["complete"] or summary["failure"]:
        raise ValueError("complete collection and independent native replay required")
    audit = p.read(directory / "NATIVE-AUDIT.json")
    for path, digest in audit["receipt_sha256"].items():
        if p.sha(Path(path)) != digest:
            raise ValueError("native-audited readout changed")
    rows = {
        (j["task_id"], j["repeat"]): p.read(directory / "episodes" / (j["episode_id"] + ".json"))
        for j in plan["jobs"]
    }
    values = {}
    for key, row in rows.items():
        checked = audit["audits"][row["episode_id"]]
        values[key] = checked["native_score"] if checked["replayed"] else None
    entropy = [
        h
        for path in (directory / "generation-logps").glob("*.json")
        for h in p.read(path)["token_entropies"]
    ]
    stats = dict(
        path=str(directory),
        plan_sha256=p.sha(directory / "PLAN.json"),
        native_audit_sha256=p.sha(directory / "NATIVE-AUDIT.json"),
        successes=audit["successes"],
        observed=audit["observed"],
        physical_cost=audit["physical_cost"],
        errors=audit["errors"],
        sampled_token_entropy_mean=mean(entropy),
        entropy_tokens=len(entropy),
        first_response_diversity=audit["first_response_diversity"],
        adapter=plan["adapter"],
    )
    return plan, values, stats


def analyze(root: Path, update: int) -> dict:
    cells, values, plans, training = {}, {}, {}, {}
    for mode in ("raw", "binder"):
        for endpoint in ("warm", f"{update:04d}"):
            key = f"{mode}_{endpoint}"
            plans[key], values[key], cells[key] = load_cell(root / mode / f"readout-{endpoint}")
        output = root / mode / f"train-{update:04d}"
        training[mode] = p.read(output / "SUMMARY.json")
        if not training[mode]["endpoint_usable"]:
            raise ValueError("failed or flat-reward training is not a learned endpoint")
        if plans[f"{mode}_{update:04d}"]["adapter"]["path"] != training[mode]["endpoint"]:
            raise ValueError("updated readout did not use its fixed training endpoint")
    reference = plans["raw_warm"]
    identity = [(j["task_id"], j["repeat"], j["seed"]) for j in reference["jobs"]]
    for plan in plans.values():
        if [(j["task_id"], j["repeat"], j["seed"]) for j in plan["jobs"]] != identity:
            raise ValueError("readout task/seed pairing changed")
        for name in (
            "tasks_sha256",
            "model_manifest_sha256",
            "base_dtype",
            "sampling",
            "max_global_calls",
            "max_global_output_tokens",
            "max_new_tokens",
            "input_plus_output_limit",
        ):
            if plan[name] != reference[name]:
                raise ValueError("readout scientific contract differs: " + name)
    if plans["raw_warm"]["adapter"] != plans["binder_warm"]["adapter"]:
        raise ValueError("raw/binder warm controls differ")
    gains = {}
    for mode in ("raw", "binder"):
        gains[mode] = {
            k: values[f"{mode}_{update:04d}"][k] - values[f"{mode}_warm"][k]
            for k in values[f"{mode}_warm"]
        }
    interaction = {k: gains["binder"][k] - gains["raw"][k] for k in gains["raw"]}
    return dict(
        schema="textcraft-assisted-rl-train-readout-20260928-v1",
        cells=cells,
        within_interface_learning_gain={k: paired(v) for k, v in gains.items()},
        binder_minus_raw_learning_gain=paired(interaction),
        training=training,
        inference_scope="Eight exposed TRAIN task clusters, two new execution seeds. "
        "Bootstrap conditions on this shared recipe world and adaptive task panel.",
        limitation="Own-interface endpoints only. A learning gain does not establish held-out "
        "benefit, extra-SFT superiority, cross-interface transfer, or an RLM recursion gain. "
        "Raw/binder gradient token doses differ with trajectories; entropy is descriptive.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--update", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.update)
    p.c.save(args.output, report)
    print(json.dumps(report["within_interface_learning_gain"], indent=2))
