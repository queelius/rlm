"""Native fresh16 warm056 versus fixed terminal RL and dose-matched extra-SFT audit."""

import argparse
from pathlib import Path

import analyze_textcraft_profiles as profiles
import eval_textcraft_endpoint_fresh as reader


def binding_from_plan(plan, kind):
    amendment = plan.get("stopped_run_amendment")
    path = None
    if amendment:
        path = Path(amendment["path"])
        if reader.c.inputs.sha(path) != amendment["sha256"]:
            raise ValueError("saved stopped-run amendment changed")
    return reader.endpoint(kind, Path(plan["training_output"]), stopped_amendment=path)


def match_slots(plans):
    reference = plans[0]

    def strip(jobs):
        return [{k: v for k, v in job.items() if k != "condition"} for job in jobs]

    for plan in plans[1:]:
        for key in (
            "tasks_sha256",
            "manifest_sha256",
            "world_sha256",
            "sampling",
            "seeds",
            "seed_rule",
            "model_manifest_sha256",
            "max_global_calls",
            "max_global_output_tokens",
            "max_new_tokens",
            "input_plus_output_limit",
            "budget_seconds",
            "truncation",
            "max_agent_depth",
        ):
            if plan[key] != reference[key]:
                raise ValueError("unmatched native007 contract: " + key)
        if strip(plan["jobs"]) != strip(reference["jobs"]):
            raise ValueError("different fixed parent/repeat/seed slots")


def analyze(warm, rl, sft):
    from transformers import AutoTokenizer

    c, audit = reader.c, profiles.audit
    paths = [warm, rl, sft]
    plans = [reader.control.read(path / "PLAN.json") for path in paths]
    if c.inputs.sha(warm / "PLAN.json") != reader.TEMPLATE_SHA:
        raise ValueError("exact accepted007 public056 baseline required")
    for plan, kind in zip(plans[1:], ("rl", "matched_sft"), strict=True):
        if plan.get("endpoint_kind") != kind or plan["adapter"] != binding_from_plan(plan, kind):
            raise ValueError("saved final endpoint identity changed")
    match_slots(plans)
    if (
        plans[1]["adapter"]["actual_optimizer_steps"]
        != plans[2]["adapter"]["actual_optimizer_steps"]
        or plans[1]["adapter"]["rl_receipt_sha256"] != plans[2]["adapter"]["rl_receipt_sha256"]
    ):
        raise ValueError("RL and extra-SFT dose/ancestry mismatch")
    stopped = plans[1].get("stopped_run_amendment")
    if stopped != plans[2].get("stopped_run_amendment"):
        raise ValueError("both comparison arms must share the stopped-run amendment")
    tokenizer = AutoTokenizer.from_pretrained(
        plans[0]["model"], local_files_only=True, trust_remote_code=False
    )
    arms = [audit.analyze(path, tokenizer, expected_task_count=16) for path in paths]
    rows = [
        {
            (row["task_id"], row["repeat"]): row
            for row in (audit.read(p) for p in (path / "episodes").glob("*.json"))
        }
        for path in paths
    ]
    for arm in arms:
        arm.pop("paired", None)
        arm.pop("depth_strata", None)
    return dict(
        schema="textcraft-fresh16-stopped-step1-comparison-v1"
        if stopped
        else "textcraft-fresh16-terminal-training-comparison-v1",
        stopped_run_amendment=stopped,
        arms=arms,
        comparisons={
            name: profiles.compare(plans[0]["jobs"], rows[a], rows[b])
            for name, a, b in [
                ("rl_minus_warm", 0, 1),
                ("sft_minus_warm", 0, 2),
                ("rl_minus_sft", 2, 1),
            ]
        },
        method=dict(
            parent_count=16,
            repeats=2,
            planned_per_arm=32,
            bootstrap_draws=20000,
            bootstrap_seed=2026092206,
            units="Root task with both seeds; shared recipes not independent",
            unknown="Never imputed zero; full effect/CI unavailable if any missing",
        ),
        caveat=(
            "Explicit stopped-RL002 one-step amendment; failed original remains unusable. "
            if stopped
            else "Fixed terminal policies, no checkpoint selection. "
        )
        + "Existing fresh007 panel "
        "reused; not newly untouched. Token/update matching is not FLOP, history, "
        "information or objective matching.",
        source_sha256={
            str(Path(m.__file__).resolve()): c.inputs.sha(Path(m.__file__))
            for m in (reader, profiles, audit)
        },
        analyzer_sha256=c.inputs.sha(Path(__file__)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-output", type=Path, default=reader.TEMPLATE.parent)
    parser.add_argument("--rl-output", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable analysis exists")
    result = analyze(args.warm_output, args.rl_output, args.sft_output)
    reader.c.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Fixed endpoint TextCraft fresh16 comparison\n\n" + result["caveat"] + "\n\n"
        )
        for name, value in result["comparisons"].items():
            stream.write(
                f"{name}: {value['wins']} wins/{value['losses']} losses/"
                f"{value['unknown_pairs']} unknown; delta="
                f"{value['complete_panel_difference']}, CI={value['ci95']}.\n\n"
            )
