"""Compact reproducible readout of public-only depth4/5 inventory diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cluster_ratio(rows, numerator, denominator, seed=2026092806, draws=10000):
    """Task bootstrap keeps all worlds, model seeds, and repeats together."""
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        groups[row["task_id"]][0] += int(numerator(row))
        groups[row["task_id"]][1] += int(denominator(row))
    values = [groups[k] for k in sorted(groups)]
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        denom = sum(v[1] for v in sample)
        if denom:
            samples.append(sum(v[0] for v in sample) / denom)
    samples.sort()
    numerator_sum, denominator_sum = (sum(v[i] for v in values) for i in (0, 1))
    return dict(
        numerator=numerator_sum,
        denominator=denominator_sum,
        proportion=numerator_sum / denominator_sum,
        task_clusters=len(values),
        numerator_positive_tasks=sum(n > 0 for n, _ in values),
        bootstrap_seed=seed,
        bootstrap_draws=draws,
        percentile_95=[samples[int(len(samples) * q)] for q in (0.025, 0.975)],
    )


def compact_aggregate(value):
    return {k: v for k, v in value.items() if k != "episode_seconds_sum"}


def main(args):
    summary_path, details_path = args.analysis / "SUMMARY.json", args.analysis / "DETAILS.json"
    summary = json.loads(summary_path.read_text())
    assert summary["source_sha256"] == digest(Path(__file__).with_name("analyze.py"))
    details = json.loads(details_path.read_bytes())
    rows = details["rows"]
    assert len(rows) == 768 and len(details["sha256"]) == 1752
    binder = [r for r in rows if r["mode"] == "binder"]
    failed = [r for r in binder if r["flags"]["failed"]]
    intervals = {
        "failure_public_completion_feasible": cluster_ratio(
            binder,
            lambda r: r["flags"]["failed_with_public_completion_feasible"],
            lambda r: r["flags"]["failed"],
        ),
        "failure_ready_prerequisite": cluster_ratio(
            binder,
            lambda r: r["flags"]["failed_with_ready_prerequisite"],
            lambda r: r["flags"]["failed"],
        ),
        "failure_repeated_stock_error": cluster_ratio(
            binder,
            lambda r: r["flags"]["failed"] and r["flags"]["repeated_insufficient"],
            lambda r: r["flags"]["failed"],
        ),
    }
    primary = [
        r
        for r in binder
        if r["panel"] == 0 and r["world"] == 42 and r["training_seed"] == "original"
    ]
    examples = []
    for episode, calls, relevant in (
        ("t04-r1-flat-original", [1, 5, 40, 41, 42, 43, 44, 45], ["c0_i1_20", "c2_ore"]),
        ("t05-r1-flat-original", [0, 32, 33], ["o9_i3_19", "o8_i4", "o0_i5_20"]),
        ("t06-r0-flat-original", [3, 48, 49, 50, 51], ["a5_i2", "a8_i2", "a3_i2", "a1_i4"]),
    ):
        row = next(r for r in primary if r["episode_id"] == episode)
        examples.append(
            dict(
                selection="Illustration from fixed primary slice; selected after counting outcomes",
                node_path=row["node_path"],
                node_sha256=details["sha256"][row["node_path"]],
                task_id=row["task_id"],
                targets=row["targets"],
                final_relevant_inventory={k: row["final_inventory"].get(k, 0) for k in relevant},
                calls=[
                    {k: t[k] for k in ("call_id", "action", "feedback")}
                    for t in row["trace"]
                    if t["index"] in calls
                ],
            )
        )
    closure_cross = Counter(
        f"closure_{r['final_public']['recipe_closure_complete']}_"
        f"feasible_{r['final_public']['completion_feasible']}"
        for r in failed
    )
    report = dict(
        schema="depth45-public-inventory-finding-20260928-v1",
        selection=dict(primary=summary["primary_selection"], secondary=summary["broadening"]),
        primary=compact_aggregate(summary["primary"]),
        secondary={k: compact_aggregate(v) for k, v in summary["secondary"].items()},
        binder_by_depth={
            k: compact_aggregate(v)
            for k, v in summary["by_depth"].items()
            if k.startswith("binder")
        },
        binder_failure_closure_feasibility=dict(closure_cross),
        binder_failure_repeated_errors=sum(r["flags"]["repeated_insufficient"] for r in failed),
        binder_failure_repeated_queries=sum(r["flags"]["repeated_query"] for r in failed),
        exploratory_task_cluster_intervals=intervals,
        context_cap_diagnostics=dict(
            episodes=32,
            repeated_stock_error_episodes=sum(
                r["flags"]["repeated_insufficient"] for r in binder if r["status"] == "context_cap"
            ),
            context_limit=8192,
            per_episode_call_cap=96,
            per_episode_output_token_cap=8192,
            causal_interpretation="Correlation only; not proof that loops caused context caps",
        ),
        examples=examples,
        definitions=summary["definitions"],
        limitations=summary["limitations"],
        prior_memory=dict(
            first_actor_full_then_notebook=[15, 14, 32],
            second_actor_full_then_notebook=[15, 6, 32],
            rl_actor_full_then_notebook=[14, 14, 32],
            interpretation="No reliable fact-only notebook gain. Existing pilots did not compute "
            "goal-conditioned remaining quantities, batch rounding, or shared-child demand.",
        ),
        proposed_probe=dict(
            status="Design only; not implemented or launched",
            treatment="Alphabetically ordered public-fact table with host-computed remaining "
            "goal demand, stock deficit, batch count, and explicit unknown-recipe frontier",
            control="Same fact table, instructions and field names; derived columns masked. "
            "Control padding matches treatment added input-token count at each current state.",
            shared="Same frozen discovery actor, binder, full history, action schema, native "
            "semantics, budgets and paired sampling seeds; no action suggestions or "
            "automatic crafts",
            panel="Exploratory already-exposed p00/world42, four depth4/5 task identities, two "
            "fresh fixed repeats per task per arm: 16 total attempts, no held-out claim",
            proposed_repeat_seeds=[2026092804, 2026092805],
            primary_endpoint="Paired native root completion; all planned slots accounted for",
            secondary_endpoints=[
                "Repeated stock-deficient crafts per attempted episode",
                "Root-ready unsuccessful finish",
                "Calls, actual input/output tokens, context caps, wall time, unknown outcomes",
            ],
            historical_seconds_per_attempt=summary["primary"]["episode_seconds_mean"],
            historical_16_attempt_minutes=summary["primary"]["episode_seconds_mean"] * 16 / 60,
            estimated_gpu_minutes_total=[35, 55],
            cap_minutes_per_arm=45,
            promotion="Completion improvement must replicate on an unread task panel and second "
            "actor seed before a general claim; do not expand solely because repeated errors fall",
            confounds="Host computation is deliberately added. Display, arithmetic, padding and "
            "policy adaptation are not isolated by this two-arm pilot; no pure-memory claim.",
        ),
        provenance=dict(
            native_audits=96,
            verified_receipt_and_task_files=1752,
            native_graph_oracle_used=False,
            model_weights_read_or_hashed=False,
            analysis_artifacts={str(p): digest(p) for p in (summary_path, details_path)},
            source_sha256={
                str(p): digest(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("analyze.py"),
                    Path(__file__).with_name("test_public.py"),
                )
            },
        ),
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(intervals, indent=2))
    print(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path, default=ROOT / "analysis-inventory-bottleneck-20260928"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("FINDING.json"))
    main(parser.parse_args())
