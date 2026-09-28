"""Paired native-completion readout of the fixed public-demand pilot, preserving unknowns."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

OUTPUT = Path(
    "/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/"
    "textcraft-public-demand-20260928-001"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def paired(jobs, masked, demand):
    differences, rows = defaultdict(list), []
    wins = losses = ties = unknown = 0
    for job in jobs:
        left, right = masked.get(job["episode_id"]), demand.get(job["episode_id"])
        known = bool(left and right and left["observed"] and right["observed"])
        delta = right["native_score"] - left["native_score"] if known else None
        rows.append(dict(task_id=job["task_id"], repeat=job["repeat"], known=known, delta=delta))
        if not known:
            unknown += 1
            continue
        differences[job["task_id"]].append(delta)
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
    estimate = interval = None
    if unknown == 0:
        estimate = (wins - losses) / len(jobs)
        rng = random.Random(2026092806)
        values = [mean(differences[k]) for k in sorted(differences)]
        samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
        interval = [samples[int(len(samples) * q)] for q in (0.025, 0.975)]
    return dict(
        planned_pairs=len(jobs),
        task_identities=len({j["task_id"] for j in jobs}),
        complete_pairs=wins + losses + ties,
        unknown_pairs=unknown,
        wins=wins,
        losses=losses,
        ties=ties,
        difference=estimate,
        difference_bounds=[(wins - losses + s * unknown) / len(jobs) for s in (-1, 1)],
        task_cluster_95=interval,
        bootstrap_seed=2026092806,
        rows=rows,
    )


def main(args):
    plans, audits, episodes, hashes = {}, {}, {}, {}
    for mode in ("masked", "demand"):
        directory = args.root / mode
        plan_path, audit_path = directory / "PLAN.json", directory / "NATIVE-AUDIT.json"
        plan, audit = read(plan_path), read(audit_path)
        if plan["table_mode"] != mode or audit["table_mode"] != mode:
            raise ValueError("table arm identity mismatch")
        if audit["sha256"][str(plan_path)] != sha(plan_path):
            raise ValueError("PLAN differs from native audit")
        rows = {}
        for path in (directory / "episodes").glob("*.json"):
            if sha(path) != audit["sha256"][str(path)]:
                raise ValueError("native audited episode changed")
            rows[path.stem] = read(path)
            hashes[str(path)] = sha(path)
        plans[mode], audits[mode], episodes[mode] = plan, audit, rows
        hashes.update({str(p): sha(p) for p in (plan_path, audit_path)})
    for key in (
        "adapter",
        "tasks_sha256",
        "manifest_sha256",
        "world_sha256",
        "seeds",
        "sampling",
        "budget_seconds",
        "max_global_calls",
        "max_global_output_tokens",
        "input_plus_output_limit",
        "table_contract",
        "source_sha256",
    ):
        if plans["masked"][key] != plans["demand"][key]:
            raise ValueError("unmatched paired field: " + key)
    jobs = plans["masked"]["jobs"]

    def normalized(plan):
        return [{k: v for k, v in job.items() if k != "condition"} for job in plan["jobs"]]

    if normalized(plans["masked"]) != normalized(plans["demand"]) or len(jobs) != 8:
        raise ValueError("fixed eight paired slots differ")
    result = dict(
        schema="public-demand-paired-readout-20260928-v1",
        contrast="demand minus masked",
        primary=paired(jobs, episodes["masked"], episodes["demand"]),
        arms={
            mode: {
                k: audit[k]
                for k in (
                    "planned",
                    "observed",
                    "successes",
                    "success_rate_bounds",
                    "physical_cost",
                    "terminal",
                    "unresolved_starts",
                    "diagnostic_counts",
                    "diagnostic_episode_flags",
                )
            }
            for mode, audit in audits.items()
        },
        source_sha256=sha(Path(__file__)),
        sha256=hashes,
        caveat="Four exposed task identities in one shared world; two repeats each. "
        "Exploratory task-cluster interval only when all pairs are known. Host arithmetic "
        "assistance and display/padding effects are not separately identified. State-conditional "
        "input lengths are matched, not cumulative inference cost. No held-out claim.",
    )
    path = args.output or args.root / "PAIRED.json"
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result["primary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
