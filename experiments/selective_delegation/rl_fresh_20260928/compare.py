"""All fixed fresh-TRAIN diagnostic checkpoints, including extra-SFT and missing cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fresh_common as f


def load(directory, mode):
    summary_path = directory / "SUMMARY.json"
    if not summary_path.exists() or not f.read(summary_path).get("complete"):
        return dict(available=False, path=str(directory), reason="incomplete_or_skipped"), {}
    old = f.load_legacy("compare.py")
    plan, values, stats = old.load_cell(directory)
    if (
        plan["schema"] != "textcraft-fresh-train-collection-20260928-v1"
        or plan["dataset_group"] != "diagnostic"
        or plan["execution_mode"] != mode
        or plan["manifest_sha256"] != f.GROUP_SHA["diagnostic"]
    ):
        raise ValueError("diagnostic cells must use the actual frozen groupB/interface")
    _, tasks = f.load_dataset(f.DATA / "diagnostic", "readout")
    if plan["jobs"] != f.jobs(tasks, mode, "readout", plan["update"]):
        raise ValueError("fixed paired diagnostic task/seeds changed")
    stats["available"] = True
    stats["plan"] = plan
    return stats, values


def compare_values(left, right, keys):
    old = f.load_legacy("compare.py")
    return old.paired({k: right[k] - left[k] if k in right and k in left else None for k in keys})


def analyze(study: Path, update: int):
    _, tasks = f.load_dataset(f.DATA / "diagnostic", "readout")
    keys = [(t["id"], repeat) for t in tasks for repeat in (0, 1)]
    cells, outcomes, gains, sft_comparisons = {}, {}, {}, {}
    for mode in ("raw", "binder"):
        for actor, directory in (
            ("warm", study / mode / "readout-warm"),
            ("rl", study / mode / f"readout-{update:04d}"),
            ("sft", study / "sft" / mode / "readout-0001"),
        ):
            name = f"{mode}_{actor}"
            cells[name], outcomes[name] = load(directory, mode)
            if cells[name]["available"]:
                if actor == "warm":
                    expected = str(f.WARM)
                else:
                    source = (
                        study
                        / (mode if actor == "rl" else "sft")
                        / (f"train-{update:04d}" if actor == "rl" else "train-0001")
                    )
                    expected = f.read(source / "SUMMARY.json")["endpoint"]
                if cells[name]["adapter"]["path"] != expected:
                    raise ValueError("readout actor does not match its prespecified checkpoint")
        gains[mode] = compare_values(outcomes[f"{mode}_warm"], outcomes[f"{mode}_rl"], keys)
        sft_comparisons[mode] = compare_values(
            outcomes[f"{mode}_sft"], outcomes[f"{mode}_rl"], keys
        )
    interaction = {}
    for key in keys:
        if all(key in values for name, values in outcomes.items() if not name.endswith("_sft")):
            interaction[key] = (
                outcomes["binder_rl"][key]
                - outcomes["binder_warm"][key]
                - outcomes["raw_rl"][key]
                + outcomes["raw_warm"][key]
            )
        else:
            interaction[key] = None
    training = {}
    for actor in ("raw", "binder", "sft"):
        directory = study / actor / (f"train-{update:04d}" if actor != "sft" else "train-0001")
        if (directory / "SUMMARY.json").exists():
            training[actor] = f.read(directory / "SUMMARY.json")
    return dict(
        schema="textcraft-fresh-rl-transfer-diagnostic-20260928-v1",
        update=update,
        diagnostic_manifest_sha256=f.GROUP_SHA["diagnostic"],
        cells=cells,
        rl_minus_warm=gains,
        rl_minus_extra_sft=sft_comparisons,
        binder_minus_raw_learning_gain=f.load_legacy("compare.py").paired(interaction),
        training=training,
        scope="GroupB comprises8 prospectively selected official TRAIN goals disjoint in "
        "target roots/IDs from groupA optimization and oldSFT. It is a transfer diagnostic, "
        "not official held-out confirmation. Shared world and recipe grammar remain.",
        endpoint_policy="All prospective checkpoints retained/reported; no diagnostic success "
        "selection of favorable checkpoints. Four-cycle admission uses usable prior optimizer "
        "boundaries and reward variation, never diagnostic scores.",
        sft_caveat="ExtraSFT is one fresh-Adam step on all original366 public teacher actions. "
        "Matches initial actor/LR/clip/step/numerics, not target-token dose, data, or likelihood "
        "temperature. For RL steps2–4 this remains a fixed one-step reference, not equal-step SFT.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--update", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.study, args.update)
    f.c.save(args.output, report)
    print(
        json.dumps(
            dict(
                update=args.update,
                rl_minus_warm=report["rl_minus_warm"],
                rl_minus_extra_sft=report["rl_minus_extra_sft"],
            ),
            indent=2,
        )
    )
