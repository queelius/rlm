"""Parent-dispatched cost-branch stages, with explicit conditional skips."""

from __future__ import annotations

import argparse
import json
import os
import sys

import cost_reward as r


def resolve(mode: str, kind: str) -> tuple:
    trained = r.STUDY / mode / "train-0001"
    if kind == "train":
        return (
            trained,
            [
                sys.executable,
                str(r.HERE / "train_cost.py"),
                "--collection",
                str(r.FRESH_STUDY / mode / "collect-0001"),
                "--output",
                str(trained),
                "--hours",
                "2",
            ],
            None,
        )
    output = r.STUDY / mode / "readout-0001"
    summary_path = trained / "SUMMARY.json"
    summary = r.f.read(summary_path) if summary_path.exists() else {}
    if not summary.get("endpoint_usable"):
        return output, None, "No usable cost-trained endpoint; no replacement actor/readout"
    from pathlib import Path

    checkpoint = Path(summary["endpoint"])
    plan = r.f.read(trained / "PLAN.json")
    state = r.f.read(checkpoint / "STATE.json")
    if (
        plan["schema"] != "textcraft-error-cost-first-step-rloo-20260928-v1"
        or plan["execution_mode"] != mode
        or state["step"] != 1
        or state["plan_sha256"] != r.f.sha(trained / "PLAN.json")
        or not checkpoint.resolve().is_relative_to(trained)
    ):
        raise ValueError("readout requires its true one-step cost checkpoint lineage")
    r.f.persist(
        output / "READOUT-LINEAGE.json",
        dict(
            training=str(trained),
            training_plan_sha256=r.f.sha(trained / "PLAN.json"),
            training_summary_sha256=r.f.sha(summary_path),
            checkpoint=str(checkpoint),
            source_sha256=r.source_pins(mode),
            fixed_diagnostic_manifest_sha256=r.f.GROUP_SHA["diagnostic"],
            checkpoint_selection="Only prespecified first step; no diagnostic-outcome gate",
        ),
    )
    return (
        output,
        [
            sys.executable,
            str(r.f.HERE / "collect.py"),
            "--dataset",
            str(r.f.DATA / "diagnostic"),
            "--mode",
            mode,
            "--phase",
            "readout",
            "--update",
            "1",
            "--hours",
            "1.5",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        None,
    )


def run(args):
    output, argv, reason = resolve(args.mode, args.kind)
    if argv is None:
        r.skip(output, reason)
        print(json.dumps(dict(skipped=True, reason=reason, GPU_loaded=False)))
        return
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("train", "readout"), required=True)
    parser.add_argument("--mode", choices=("raw", "binder"), required=True)
    run(parser.parse_args())
