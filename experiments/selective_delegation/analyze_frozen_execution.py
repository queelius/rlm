"""Held-seed plan-selection analysis for frozen execution receipts."""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import analyze_helper
import frozen_execution_probe as collector
import probe

BOOTSTRAP_SEED = 2026092165


def reward_variability(rows, parents=None):
    """Observed execution variance and binary-reward credit/rank stability; ties explicit."""
    values = {(r["case_id"], r["candidate"], r["execution_repeat"]): r["reward"] for r in rows}
    if len(values) != len(rows):
        raise ValueError("duplicate frozen candidate/seed")
    parents = sorted(parents if parents is not None else {r["case_id"] for r in rows})
    per_plan, histogram, signs, ranks = [], Counter(), Counter(), Counter()
    pairs = disagreements = 0

    def sign(value):
        return (value > 0) - (value < 0)

    for parent in parents:
        for candidate in range(4):
            rewards = [values.get((parent, candidate, r)) for r in range(4)]
            complete = all(v is not None for v in rewards)
            successes = sum(rewards) if complete else None
            if complete:
                histogram[str(successes)] += 1
            for a, b in combinations(rewards, 2):
                if a is not None and b is not None:
                    pairs += 1
                    disagreements += a != b
            per_plan.append(
                {
                    "case_id": parent,
                    "candidate": candidate,
                    "rewards": rewards,
                    "complete": complete,
                    "successes_of_four": successes,
                }
            )
        if any(values.get((parent, c, r)) is None for c in range(4) for r in range(4)):
            continue
        for a, b in combinations(range(4), 2):
            first = [values.get((parent, c, a)) for c in range(4)]
            second = [values.get((parent, c, b)) for c in range(4)]
            if any(v is None for v in first + second):
                continue
            for c in range(4):
                # RLOO is (4*r_c - sum rewards)/3; only its sign is needed here.
                x, y = sign(4 * first[c] - sum(first)), sign(4 * second[c] - sum(second))
                label = (
                    "both_zero"
                    if x == y == 0
                    else "one_zero"
                    if 0 in (x, y)
                    else ("sign_flip" if x != y else "same_positive" if x > 0 else "same_negative")
                )
                signs[label] += 1
            for c, d in combinations(range(4), 2):
                x, y = sign(first[c] - first[d]), sign(second[c] - second[d])
                label = (
                    "tied_both"
                    if x == y == 0
                    else "one_seed_tie"
                    if 0 in (x, y)
                    else ("reversed_order" if x != y else "same_strict_order")
                )
                ranks[label] += 1
    return {
        "per_plan": per_plan,
        "success_count_histogram": dict(sorted(histogram.items())),
        "same_plan_seed_pairs": pairs,
        "same_plan_disagreeing_seed_pairs": disagreements,
        "same_plan_disagreement_fraction": disagreements / pairs if pairs else None,
        "advantage_sign_counts": dict(signs),
        "advantage_sign_comparisons": sum(signs.values()),
        "rank_order_counts": dict(ranks),
        "rank_order_comparisons": sum(ranks.values()),
        "note": "Credit signs compare fresh four-candidate RLOO rewards across seed pairs; "
        "rank counts compare candidate-pair order across seed pairs, with ties separate. "
        "Missing/null rewards are unavailable, never zero. Seed-pair counts are descriptive, "
        "not independent sample sizes.",
    }


def held_seed_selection(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = row["candidate"], row["execution_repeat"]
        if key in groups[row["case_id"]]:
            raise ValueError("duplicate frozen candidate/seed")
        if key[0] not in range(4) or key[1] not in range(4) or row["reward"] not in (0, 1, None):
            raise ValueError("unexpected candidate/seed/reward")
        groups[row["case_id"]][key] = row
    results = []
    for parent, group in sorted(groups.items()):
        if set(group) != {(c, r) for c in range(4) for r in range(4)} or any(
            row["reward"] is None for row in group.values()
        ):
            continue
        for held in range(4):
            one_seed = (held + 1) % 4
            other = [r for r in range(4) if r != held]
            one = min(range(4), key=lambda c: (-group[c, one_seed]["reward"], c))
            three = min(range(4), key=lambda c: (-sum(group[c, r]["reward"] for r in other), c))
            results.append(
                {
                    "parent": parent,
                    "held_seed": held,
                    "one_training_seed": one_seed,
                    "three_training_seeds": other,
                    "one": group[one, held]["reward"],
                    "three": group[three, held]["reward"],
                    "one_candidate": one,
                    "three_candidate": three,
                }
            )
    return results


def analyze(output, cases_path, draws=20000, seed=BOOTSTRAP_SEED):
    output, cases_path = Path(output).resolve(), Path(cases_path).resolve()
    hashes = {}

    def read(path):
        path = Path(path).resolve()
        hashes[str(path)] = collector.sha(path)
        return json.loads(path.read_text())

    plan = read(output / "PLAN.json")
    expected = collector.prepare(plan["source"], cases_path)
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen source/input/seed contract changed")
    hashes.update(plan["source_hashes"])
    hashes[str(cases_path)] = collector.sha(cases_path)
    for path, digest in plan["dependencies"].items():
        if collector.sha(path) != digest:
            raise ValueError("collector dependency changed")
        hashes[path] = digest
    cases = {r["id"]: r for r in map(json.loads, cases_path.read_text().splitlines())}
    calls = {}
    starts = {}
    for directory, target in (("calls", calls), ("starts", starts)):
        for path in (output / directory).glob("*.json"):
            row = read(path)
            if row["call_id"] != path.stem or path.stem in target:
                raise ValueError("duplicate/malformed native receipt identity")
            target[path.stem] = row
    used = set()
    helper_sha = plan["helper_contract"]["adapter_binding"]["adapter_model.safetensors"]

    class Replay:
        def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
            if identity not in calls or identity in used:
                raise RuntimeError("missing/reused frozen native call")
            call = calls[identity]
            request = call["request"]
            enabled = role == "helper"
            expected = {
                "prompt": prompt,
                "condition": condition,
                "role": role,
                "seed": seed,
                "model": plan["model"],
                "adapter_enabled": enabled,
                "adapter_name": "helper_sft" if enabled else None,
                "adapter_sha256": helper_sha if enabled else None,
            }
            if (
                probe.runtime.digest(request) != call["request_digest"]
                or any(request.get(k) != v for k, v in expected.items())
                or any(
                    call.get(k) != expected[k]
                    for k in (
                        "condition",
                        "role",
                        "adapter_enabled",
                        "adapter_name",
                        "adapter_sha256",
                    )
                )
                or call["input_token_ids"] != request["input_token_ids"]
            ):
                raise RuntimeError("frozen native prompt/seed/role/adapter differs")
            sampling = {
                "do_sample": True,
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "max_time": 90.0,
                "max_new_tokens": max_new_tokens,
            }
            if request["sampling"] != sampling:
                raise RuntimeError("frozen native sampling/cap differs")
            for field, ids in (
                ("prompt_tokens", "input_token_ids"),
                ("completion_tokens", "output_token_ids"),
            ):
                value = call.get("usage", {}).get(field)
                if value is not None and (ids not in call or value != len(call[ids])):
                    raise RuntimeError("native token usage differs")
            if (
                identity not in starts
                or starts[identity]["request_digest"] != call["request_digest"]
            ):
                raise RuntimeError("native start/return identity differs")
            used.add(identity)
            return call

    rows, indexed = [], {}
    for path in (output / "episodes").glob("*.json"):
        row = read(path)
        if row["episode_id"] != path.stem or path.stem in indexed:
            raise ValueError("duplicate/malformed episode")
        indexed[path.stem] = row
    expected_ids = set()
    for slot in plan["slots"]:
        for repeat in range(4):
            identity = f"{slot['case_id']}-c{slot['candidate']}-e{repeat}"
            expected_ids.add(identity)
            if identity not in indexed:
                continue
            rebuilt = collector.execute_slot(Replay(), cases[slot["case_id"]], slot, repeat)
            if rebuilt != indexed[identity]:
                raise ValueError(
                    "frozen saved outcome differs from request replay/official regrade"
                )
            rows.append(rebuilt)
    if set(indexed) - expected_ids:
        raise ValueError("episode outside planned inventory")
    selections = held_seed_selection(rows)
    complete = sorted({r["parent"] for r in selections})
    parent_deltas = {
        p: mean(r["three"] - r["one"] for r in selections if r["parent"] == p) for p in complete
    }
    clusters = analyze_helper.component_clusters([cases[p] for p in complete])
    interval = (
        analyze_helper.clustered_interval(parent_deltas, clusters, draws, seed)
        if complete
        else None
    )
    unresolved = [row for cid, row in starts.items() if cid not in calls]
    roots = [
        read(path)
        for path in plan["source_hashes"]
        if "/calls/" in path and path.endswith("-root.json")
    ]
    for path in (Path(__file__), Path(collector.__file__), Path(analyze_helper.__file__)):
        hashes[str(path)] = collector.sha(path)
    return {
        "output": str(output),
        "source_plan": plan,
        "planned_parents": 16,
        "planned_plans": 64,
        "planned_execution_slots": 256,
        "recorded_slots": len(rows),
        "missing_slots": 256 - len(rows),
        "unavailable_slots": sum(r["reward"] is None for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "reward_variability": reward_variability(rows, parents=plan["case_ids"]),
        "all_planned_complete": len(rows) == 256
        and all(r["reward"] is not None for r in rows)
        and not unresolved,
        "selection_complete_parents": complete,
        "selection_unavailable_parents": sorted(set(plan["case_ids"]) - set(complete)),
        "held_seed_selections": selections,
        "one_seed_held_reward": mean(r["one"] for r in selections) if selections else None,
        "three_seed_held_reward": mean(r["three"] for r in selections) if selections else None,
        "three_minus_one": interval,
        "method": {
            "draws": draws,
            "seed": seed,
            "component_clusters": clusters,
            "selection": "For held h, one selects on (h+1)%4; three selects on all other "
            "repeats. Both tie-break by smallest candidate index. Held reward never selects. "
            "Average four exclusions within parent before component-cluster bootstrap.",
            "missing": "Any absent/null candidate-by-seed makes the parent unavailable for "
            "selection analysis; never convert transport failure into zero. "
            "Full256 inventory retained separately.",
        },
        "new_physical_cost": analyze_helper.measured(list(calls.values()) + unresolved),
        "shared_historical_root_cost": analyze_helper.measured(roots),
        "physical_including_roots_once": analyze_helper.measured(
            roots + list(calls.values()) + unresolved
        ),
        "unlinked_call_ids": sorted(set(calls) - used),
        "unresolved_start_ids": [r["call_id"] for r in unresolved],
        "input_source_receipt_sha256": hashes,
        "caution": "Exploratory same-parent frozen-plan selector diagnostic, not a deployable "
        "router or unbiased oracle. Only16 training parents, with component dependence. "
        "Four executions cost more than one; no new planner training or generalization claim.",
    }


def markdown(report):
    lines = [
        "# Frozen-plan execution-noise diagnostic",
        "",
        f"Source: `{report['output']}`",
        "",
        f"Recorded {report['recorded_slots']}/256 slots; missing {report['missing_slots']}; "
        f"unavailable generations {report['unavailable_slots']}. "
        f"Complete: {report['all_planned_complete']}.",
        "",
        f"Selection-eligible parents: {len(report['selection_complete_parents'])}/16; "
        f"component clusters: {len(report['method']['component_clusters'])}.",
        "",
    ]
    variability = report["reward_variability"]
    lines += [
        "Primary mechanism: execution reward variability.",
        "",
        "Per-plan successes/4 histogram: "
        f"{json.dumps(variability['success_count_histogram'], sort_keys=True)}.",
        f"Same-plan seed-pair disagreements: {variability['same_plan_disagreeing_seed_pairs']}/"
        f"{variability['same_plan_seed_pairs']} observed pairs.",
        "RLOO advantage signs: "
        f"{json.dumps(variability['advantage_sign_counts'], sort_keys=True)}.",
        "Candidate-pair rank agreement: "
        f"{json.dumps(variability['rank_order_counts'], sort_keys=True)}.",
        "",
        "Secondary TRAIN-label-using held-seed selector:",
        "",
    ]
    if report["three_minus_one"] is not None:
        pair = report["three_minus_one"]
        lines += [
            f"Held reward: one seed {report['one_seed_held_reward']:.2%}; "
            f"three seeds {report['three_seed_held_reward']:.2%}.",
            f"Three minus one: {100 * pair['estimate']:+.2f}pp "
            f"[{100 * pair['ci95'][0]:+.2f}, {100 * pair['ci95'][1]:+.2f}] exploratory95% CI.",
            "",
        ]
    lines += [
        f"New physical cost: {json.dumps(report['new_physical_cost'], sort_keys=True)}",
        "",
        report["method"]["selection"],
        "",
        report["method"]["missing"],
        "",
        report["caution"],
    ]
    return "\n".join(lines) + "\n"


def write_report(report, path):
    path = Path(path)
    if path.exists() or path.with_suffix(".md").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    with path.with_suffix(".md").open("x") as stream:
        stream.write(markdown(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    write_report(analyze(args.output, args.cases), args.report)
