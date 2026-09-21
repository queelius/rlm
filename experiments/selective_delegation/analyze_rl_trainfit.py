"""Paired TRAIN-only fit analysis; neither checkpoint selection nor generalization evidence."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import eval_helper
import eval_rl_trainfit as collector
import frozen_execution_probe as frozen

SEED = 2026092167


def value(row, metric):
    if row is None or row.get("reward") is None:
        return 0
    return row["reward"] if metric == "em" else row.get("score", {}).get("f1", 0)


def paired_changes(before, after):
    if len(before) != len(after) or not before:
        raise ValueError("equal nonempty planned inventories required")
    result = {
        "planned": len(before),
        "missing_after": sum(r is None or r.get("reward") is None for r in after),
    }
    counts = Counter()
    for a, b in zip(before, after, strict=True):
        if a is None or b is None or a.get("reward") is None or b.get("reward") is None:
            continue
        direction = (
            "wins"
            if b["reward"] > a["reward"]
            else "losses"
            if b["reward"] < a["reward"]
            else "ties"
        )
        both_valid = a.get("score", {}).get("valid", False) and b.get("score", {}).get(
            "valid", False
        )
        counts[direction + ("_both_valid" if both_valid else "_protocol")] += 1
    result.update(counts)
    for metric in ("em", "f1"):
        result[metric + "_before"] = mean(value(r, metric) for r in before)
        result[metric + "_after"] = mean(value(r, metric) for r in after)
        result[metric + "_difference"] = result[metric + "_after"] - result[metric + "_before"]
    return result


def analyze(output, cases_path):
    output, cases_path = Path(output).resolve(), Path(cases_path).resolve()
    plan = eval_helper.read(output / "PLAN.json")
    current = collector.prepare(plan["source"], cases_path)
    if any(plan.get(key) != val for key, val in current.items()):
        raise ValueError("source/checkpoint/input contract changed")
    hashes = dict(plan["source_hashes"])
    for path, digest in plan["dependencies"].items():
        if frozen.sha(path) != digest:
            raise ValueError("collector source changed")
        hashes[path] = digest
    cases = {r["id"]: r for r in map(json.loads, cases_path.read_text().splitlines())}
    before, after, pairs, all_records = [], [], [], []
    parent_diffs = {"em": {}, "f1": {}}
    source = Path(plan["source"]) / "batch-0001"
    observed_ids, expected_ids = set(), set()
    for index, cid in enumerate(plan["case_ids"]):
        bgroup, agroup = [], []
        for candidate in range(4):
            a, old = collector.replay_episode(
                source,
                cases[cid],
                index,
                candidate,
                plan["baseline_adapter_sha256"],
                plan["helper_contract"],
            )
            identity = a["episode_id"]
            expected_ids.add(identity)
            b, new = None, []
            if (output / "episodes" / (identity + ".json")).exists():
                b, new = collector.replay_episode(
                    output,
                    cases[cid],
                    index,
                    candidate,
                    plan["checkpoint_binding"]["adapter_model.safetensors"],
                    plan["helper_contract"],
                )
                observed_ids.add(identity)
                if old[0]["input_token_ids"] != new[0]["input_token_ids"]:
                    raise ValueError("paired root input token IDs differ")
            before.append(a)
            after.append(b)
            bgroup.append(a)
            agroup.append(b)
            all_records.extend(old)
            pairs.append(
                {
                    "case_id": cid,
                    "candidate": candidate,
                    "before_reward": a["reward"],
                    "after_reward": b["reward"] if b else None,
                    "before_status": a["status"],
                    "after_status": b["status"] if b else "missing",
                    "plan_equal": a.get("plan") == b.get("plan") if b else None,
                    "root_output_token_ids_equal": old[0]["output_token_ids"]
                    == new[0].get("output_token_ids")
                    if new
                    else None,
                    "before_root_tokens": len(old[0]["output_token_ids"]),
                    "after_root_tokens": len(new[0].get("output_token_ids", [])) if new else None,
                }
            )
        for metric in parent_diffs:
            parent_diffs[metric][cid] = mean(
                value(b, metric) - value(a, metric) for a, b in zip(bgroup, agroup, strict=True)
            )
    if {p.stem for p in output.glob("episodes/*.json")} - expected_ids:
        raise ValueError("unexpected episode outside64-slot inventory")
    calls = [eval_helper.read(p) for p in output.glob("calls/*.json")]
    ids = {r["call_id"] for r in calls}
    unresolved = [eval_helper.read(p) for p in output.glob("starts/*.json") if p.stem not in ids]
    linked = {cid for row in after if row for cid in row["call_ids"]}
    for path in [
        output / "PLAN.json",
        cases_path,
        Path(__file__),
        Path(analyze_helper.__file__),
        *output.glob("episodes/*.json"),
        *output.glob("calls/*.json"),
        *output.glob("starts/*.json"),
    ]:
        hashes[str(path)] = frozen.sha(path)
    clusters = analyze_helper.component_clusters([cases[cid] for cid in plan["case_ids"]])
    return {
        "output": str(output),
        "plan": plan,
        "paired": paired_changes(before, after),
        "planned_parents": 16,
        "planned_candidates": 64,
        "recorded": len(observed_ids),
        "missing": 64 - len(observed_ids),
        "pairs": pairs,
        "before_status": dict(Counter(r["status"] for r in before)),
        "after_status": dict(Counter(r["status"] if r else "missing" for r in after)),
        "plan_changes": sum(p["plan_equal"] is False for p in pairs),
        "root_token_changes": sum(p["root_output_token_ids_equal"] is False for p in pairs),
        "intervals": {
            metric: analyze_helper.clustered_interval(values, clusters, 20000, SEED)
            for metric, values in parent_diffs.items()
        },
        "method": {
            "bootstrap_draws": 20000,
            "bootstrap_seed": SEED,
            "component_clusters": clusters,
            "denominator": "All64 planned candidates; average4 paired differences per parent. "
            "Missing scores zero for planned metrics, not protocol outcomes.",
        },
        "new_physical_cost": analyze_helper.measured(calls + unresolved),
        "reused_baseline_cost_once": analyze_helper.measured(all_records),
        "unlinked_calls": sorted(ids - linked),
        "unresolved_starts": len(unresolved),
        "input_source_receipt_sha256": hashes,
        "caution": "Same16 TRAIN parents and original rollout seeds; post-training fit diagnostic, "
        "not generalization. Frozen helper36/base final; original helper2/final seed collision "
        "retained for matched contract. Checkpoint selected by terminal stopping rule, not score. "
        "Missing outcomes can bias planned-zero differences; "
        "inspect completion before interpreting.",
    }


def write_report(report, path):
    path = Path(path)
    if path.exists() or path.with_suffix(".md").exists():
        raise FileExistsError(path)
    pair = report["paired"]
    lines = [
        "# Frozen-policy TRAIN fit",
        "",
        f"Source: `{report['output']}`",
        "",
        f"{report['recorded']}/64 recorded candidates on16 TRAIN parents; "
        f"missing {report['missing']}.",
        f"EM: {pair['em_before']:.2%} → {pair['em_after']:.2%}; "
        f"F1: {pair['f1_before']:.2%} → {pair['f1_after']:.2%}.",
        f"Protocol/content outcomes: `{json.dumps(pair, sort_keys=True)}`.",
        f"Changed plans: {report['plan_changes']}; "
        f"changed root token sequences: {report['root_token_changes']}.",
        f"Paired component-cluster intervals: `{json.dumps(report['intervals'], sort_keys=True)}`.",
        f"New physical cost: `{json.dumps(report['new_physical_cost'], sort_keys=True)}`.",
        "",
        report["caution"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    with path.with_suffix(".md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    write_report(analyze(args.output, args.cases), args.report)
