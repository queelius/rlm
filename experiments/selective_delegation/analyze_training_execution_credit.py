"""Paired TRAIN terminal reward versus action-dependent execution-benefit diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

import analyze_helper
import training_execution_credit as collector

runtime, evaluation, probe = collector.runtime, collector.evaluation, collector.probe
SEED = 2026092177


def sign(value):
    return (value > 0) - (value < 0)


def control_comparison(identity, call, original, grade, old):
    observed = call["available"]
    return dict(
        episode_id=identity,
        observed=observed,
        same_output_tokens=call.get("output_token_ids") == original.get("output_token_ids")
        if observed
        else None,
        same_text=call.get("text") == original.get("text") if observed else None,
        same_grade=all(grade[k] == old[k] for k in ("valid", "correct", "f1"))
        if observed
        else None,
    )


def group_credit(executed, plan_only):
    if len(executed) != 4 or len(plan_only) != 4:
        raise ValueError("four candidates required")
    if any(x is None for x in executed + plan_only):
        return None
    benefit = [e - p for e, p in zip(executed, plan_only, strict=True)]

    def advantages(values):
        return [(4 * x - sum(values)) / 3 for x in values]

    terminal, credit = advantages(executed), advantages(benefit)
    labels = []
    for a, b in zip(terminal, credit, strict=True):
        labels.append(
            "both_zero"
            if a == b == 0
            else "terminal_only"
            if b == 0
            else "execution_only"
            if a == 0
            else "agree"
            if sign(a) == sign(b)
            else "opposite"
        )
    return dict(
        executed=executed,
        plan_only=plan_only,
        execution_benefit=benefit,
        terminal_advantages=terminal,
        execution_advantages=credit,
        alignment=labels,
    )


def analyze(output):
    output = Path(output).resolve()
    plan = runtime.read(output / "PLAN.json")
    expected, cases, jobs = collector.prepare(
        Path(plan["root"]), output, plan["budget_seconds"] / 3600
    )
    if expected != plan:
        raise ValueError("immutable source/collector contract changed")
    hashes = {
        str(output / "PLAN.json"): probe.campaign.sha(output / "PLAN.json"),
        **plan["source_hashes"],
    }
    calls, rows, controls, missing, statuses = {}, {}, [], [], Counter()
    allowed_calls, allowed_episodes = set(), set()
    for job in jobs:
        for mode in ("plan_only", "factual_replay") if job["control"] else ("plan_only",):
            identity = f"{job['case_id']}-r{job['repeat']}-{job['policy']}-{mode}"
            allowed_episodes.add(identity)
            allowed_calls.add(identity + "-final")
            path = output / "episodes" / (identity + ".json")
            if not path.exists():
                if mode == "plan_only":
                    missing.append(identity)
                continue
            row = runtime.read(path)
            hashes[str(path)] = probe.campaign.sha(path)
            if (
                row["episode_id"] != identity
                or row["old"] != job["old"]
                or row["seed"] != job["seed"]
                or row["call_ids"] != [identity + "-final"]
                or row["source_episode_id"] != job["source_episode_id"]
            ):
                raise ValueError("episode source/identity mismatch")
            call_path = output / "calls" / (identity + "-final.json")
            call = runtime.read(call_path)
            calls[call["call_id"]] = call
            hashes[str(call_path)] = probe.campaign.sha(call_path)
            request = call["request"]
            prompt = job["plan_only_prompt"] if mode == "plan_only" else job["factual_prompt"]
            if (
                request["prompt"] != prompt
                or request["seed"] != job["seed"]
                or request["sampling"] != plan["sampling"]
                or request["model"] != plan["model"]
                or request["role"] != "final"
                or request["condition"] != "base"
                or request["adapter_enabled"]
                or request["adapter_sha256"] is not None
                or request["helper_contract"] != plan["helper_contract"]
                or request["input_token_ids"] != call["input_token_ids"]
                or (
                    call["available"]
                    and (
                        call["usage"]["prompt_tokens"] != len(call["input_token_ids"])
                        or call["usage"]["completion_tokens"] != len(call["output_token_ids"])
                    )
                )
                or probe.runtime.digest(request) != call["request_digest"]
                or row["observed"] != call["available"]
            ):
                raise ValueError("native final request differs")
            grade = probe.grade(call["text"] if call["available"] else "", cases[job["case_id"]])
            if row["new"] != grade:
                raise ValueError("official final grade differs")
            if mode == "factual_replay":
                original = job["factual"]
                if request["input_token_ids"] != original["request"]["input_token_ids"]:
                    raise ValueError("factual replay native input differs")
                controls.append(control_comparison(identity, call, original, grade, job["old"]))
            else:
                rows[job["case_id"], job["repeat"], int(job["policy"][1:])] = row
                statuses[row["status"]] += 1
    actual_calls = {p.stem for p in (output / "calls").glob("*.json")}
    actual_episodes = {p.stem for p in (output / "episodes").glob("*.json")}
    if not actual_calls <= allowed_calls or not actual_episodes <= allowed_episodes:
        raise ValueError("unexpected output calls/episodes")
    # Count returned calls even if the process stopped before writing its episode.
    for cid in actual_calls - calls.keys():
        path = output / "calls" / (cid + ".json")
        calls[cid] = runtime.read(path)
        hashes[str(path)] = probe.campaign.sha(path)
    starts = [runtime.read(p) for p in (output / "starts").glob("*.json")]
    unresolved = [r for r in starts if r["call_id"] not in calls]
    groups, both_valid, alignment, mass = [], [], Counter(), Counter()
    indexed_jobs = {(j["case_id"], j["repeat"], int(j["policy"][1:])): j for j in jobs}
    for parent in plan["parents"]:
        for setting in range(5):
            keys = [(parent, setting, c) for c in range(4)]
            old = [indexed_jobs[k]["old"] for k in keys]
            new = [rows.get(k) for k in keys]
            credit = group_credit(
                [int(x["correct"]) for x in old],
                [int(x["new"]["correct"]) if x and x["observed"] else None for x in new],
            )
            if credit is None:
                continue
            credit.update(
                case_id=parent,
                setting=setting,
                both_valid=all(
                    a["valid"] and b["new"]["valid"] for a, b in zip(old, new, strict=True)
                ),
            )
            groups.append(credit)
            if credit["both_valid"]:
                both_valid.append(credit)
            alignment.update(credit["alignment"])
            for a, label in zip(credit["terminal_advantages"], credit["alignment"], strict=True):
                mass[label] += abs(a)
    observed = [r for r in rows.values() if r["observed"]]
    changes = Counter()
    for row in observed:
        delta = int(row["old"]["correct"]) - int(row["new"]["correct"])
        if delta:
            kind = (
                "both_valid" if row["old"]["valid"] and row["new"]["valid"] else "protocol_involved"
            )
            changes[("execution_useful_" if delta > 0 else "execution_harmful_") + kind] += 1
    stability = {metric: Counter() for metric in ("terminal_advantages", "execution_advantages")}
    by_group = {(g["case_id"], g["setting"]): g for g in groups}
    complete_parents = [p for p in plan["parents"] if all((p, s) in by_group for s in range(5))]
    benefit_histograms = Counter()
    for parent in complete_parents:
        for candidate in range(4):
            benefit_histograms[
                tuple(by_group[parent, s]["execution_benefit"][candidate] for s in range(5))
            ] += 1
        for first, second in combinations(range(5), 2):
            for metric, counts in stability.items():
                left, right = by_group[parent, first][metric], by_group[parent, second][metric]
                for a, b in zip(left, right, strict=True):
                    counts[
                        "sign_agree_nonzero"
                        if a * b > 0
                        else "sign_opposite"
                        if a * b < 0
                        else "both_zero"
                        if a == b == 0
                        else "one_zero"
                    ] += 1
                for i, j in combinations(range(4), 2):
                    a, b = sign(left[i] - left[j]), sign(right[i] - right[j])
                    counts[
                        "rank_agree"
                        if a * b > 0
                        else "rank_opposite"
                        if a * b < 0
                        else "rank_both_tied"
                        if a == b == 0
                        else "rank_one_tied"
                    ] += 1
    intervals = None
    clusters = analyze_helper.component_clusters(list(cases.values()))
    if len(observed) == 320:
        differences = {
            p: mean(
                int(rows[p, s, c]["old"]["correct"]) - int(rows[p, s, c]["new"]["correct"])
                for s in range(5)
                for c in range(4)
            )
            for p in plan["parents"]
        }
        intervals = analyze_helper.clustered_interval(differences, clusters, 20000, SEED)
    return dict(
        output=str(output),
        method=dict(
            bootstrap_draws=20000,
            bootstrap_seed=SEED,
            clusters=clusters,
            parent_count=16,
            component_count=len(clusters),
            planned_groups=80,
            estimand="E-P; positive means executed helper report improves final correctness",
        ),
        planned_attempts=320,
        recorded=len(rows),
        observed=len(observed),
        missing_episodes=missing,
        unavailable=sum(not r["observed"] for r in rows.values()),
        statuses=dict(statuses),
        historical_statuses=dict(Counter(j["old"]["status"] for j in jobs)),
        factual_correct=sum(j["old"]["correct"] for j in jobs),
        factual_valid=sum(j["old"]["valid"] for j in jobs),
        factual_f1=mean(j["old"]["f1"] for j in jobs),
        plan_only_correct=sum(r["new"]["correct"] for r in observed),
        plan_only_valid=sum(r["new"]["valid"] for r in observed),
        by_setting=[
            dict(
                setting=s,
                planned=64,
                observed=sum(r["repeat"] == s for r in observed),
                executed_correct=sum(j["old"]["correct"] for j in jobs if j["repeat"] == s),
                plan_only_correct=sum(r["new"]["correct"] for r in observed if r["repeat"] == s),
                credit_alignment=dict(
                    Counter(label for g in groups if g["setting"] == s for label in g["alignment"])
                ),
            )
            for s in range(5)
        ],
        plan_only_em_lower_bound=sum(r["new"]["correct"] for r in observed) / 320,
        plan_only_f1_lower_bound=sum(r["new"]["f1"] for r in observed) / 320,
        changes=dict(changes),
        complete_groups=len(groups),
        unavailable_groups=80 - len(groups),
        alignment=dict(alignment),
        terminal_absolute_advantage_mass=dict(mass),
        both_valid_groups=len(both_valid),
        both_valid_alignment=dict(Counter(x for g in both_valid for x in g["alignment"])),
        groups=groups,
        complete_five_seed_parents=complete_parents,
        cross_seed_stability={k: dict(v) for k, v in stability.items()},
        per_plan_benefit_sequences=[
            dict(sequence=list(k), plans=v) for k, v in sorted(benefit_histograms.items())
        ],
        execution_minus_plan_only=intervals,
        controls=controls,
        planned_controls=10,
        new_physical_cost=analyze_helper.measured(list(calls.values()) + unresolved),
        control_cost=analyze_helper.measured(
            [c for k, c in calls.items() if "factual_replay" in k]
        ),
        historical_acquisition_cost=plan["historical_acquisition_cost"],
        unresolved_starts=len(unresolved),
        unlinked_calls=sorted(
            actual_calls
            - {cid for r in rows.values() for cid in r["call_ids"]}
            - {c["episode_id"] + "-final" for c in controls}
        ),
        source_hashes=hashes,
        analyzer_sha256=probe.campaign.sha(__file__),
        caveat=plan["caveat"]
        + " Five seeds and four candidates are not independent parents. Unknown outcomes "
        "excluded from credit/stability; lower bounds retain all320 planned slots. "
        "Factual replay output mismatch must qualify credit interpretation.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise ValueError("report already exists")
    report = analyze(args.output)
    probe.runtime.save(args.report, report)
    text = (
        "# TRAIN execution credit\n\n"
        + report["caveat"]
        + "\n\n"
        + f"Observed {report['observed']}/320 attempts, {report['complete_groups']}/80 groups. "
        + f"Executed {report['factual_correct']}/320 correct; "
        + f"plan-only {report['plan_only_correct']}/320 (unknowns not scientific zeros).\n\n"
        + f"Credit alignment: {report['alignment']}. "
        + f"Both-valid groups: {report['both_valid_groups']}; {report['both_valid_alignment']}.\n\n"
        + f"Outcome changes: {report['changes']}. "
        + f"Cluster interval: {report['execution_minus_plan_only']}.\n\n"
        + f"Cross-seed stability: {report['cross_seed_stability']}.\n\n"
        + f"New physical cost: {report['new_physical_cost']}. "
        + f"Historical cost separately: {report['historical_acquisition_cost']}.\n\n"
        + f"Factual controls: {len(report['controls'])}/10 recorded, "
        + f"{sum(c['observed'] for c in report['controls'])}/10 observed; "
        + "token disagreements "
        + f"{sum(c['same_output_tokens'] is False for c in report['controls'])}.\n"
    )
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(text)
    print(json.dumps({"observed": report["observed"], "report": str(args.report)}))


if __name__ == "__main__":
    main()
