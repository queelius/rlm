"""Compact audited teaching-order result and fixed replication readout, CPU only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replication as study  # noqa: E402

SNAPSHOT = study.ROOT / "controls-readout-20260928-001/watch/snapshot-6d3cd53eb3b0b5f1.json"
SNAPSHOT_SHA = "09a6853d9b17b0c4e184c411e7d08f711cb4d3d3f370296f770951f308d8c6c6"
PAIR_SOURCE = study.HERE.parent / "inventory_bottleneck_20260928/compare.py"
RECEIPTS = {}


def read(path: Path):
    RECEIPTS[str(path)] = study.sha(path)
    return study.read(path)


def pair(left: list[dict], right: list[dict]) -> dict:
    def keyed(rows):
        values = {(r["task_id"], r["world"], r["seed"]): r["score"] for r in rows}
        study.require(len(values) == len(rows), "duplicate paired slots")
        return values

    a, b = keyed(left), keyed(right)
    study.require(a.keys() == b.keys(), "paired task/world/seed slots differ")
    spec = importlib.util.spec_from_file_location("replication_pair_math", PAIR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs, arms = [], [{}, {}]
    for i, key in enumerate(sorted(a)):
        task, world, seed = key
        eid = str(i)
        jobs.append(dict(episode_id=eid, task_id=task, repeat=f"{world}:{seed}"))
        for values, arm in zip((a, b), arms, strict=True):
            arm[eid] = dict(observed=values[key] is not None, native_score=values[key])
    result = module.paired(jobs, *arms)
    result.pop("rows")
    result["per_task"] = [
        dict(
            task_id=task,
            paired_slots=sum(k[0] == task for k in a),
            left_successes=sum(v == 1 for k, v in a.items() if k[0] == task),
            right_successes=sum(v == 1 for k, v in b.items() if k[0] == task),
            unknown_pairs=sum(a[k] is None or b[k] is None for k in a if k[0] == task),
        )
        for task in sorted({k[0] for k in a})
    ]
    return result


def native(directory: Path, fallback: dict | None = None) -> tuple[dict, dict]:
    actual_plan = (directory / "PLAN.json").exists()
    plan = read(directory / "PLAN.json") if actual_plan else fallback
    study.require(plan is not None, "no actual or fixed scheduling PLAN")
    jobs = plan["jobs"]
    study.require(
        len(jobs) == 16 and len({(j["task_id"], j["seed"]) for j in jobs}) == 16,
        "fixed 16 distinct task/seed slots required",
    )
    path = directory / "NATIVE-AUDIT.json"
    audit = read(path) if path.exists() else None
    if audit:
        study.require(
            actual_plan
            and audit["sha256"][str(directory / "PLAN.json")]
            == RECEIPTS[str(directory / "PLAN.json")],
            "native-audited PLAN changed",
        )
        study.require(
            not audit["unresolved_starts"] and not audit["calls_without_episode"],
            "unbound native calls",
        )
    rows = []
    for job in jobs:
        outcome = audit["audits"].get(job["episode_id"], {}) if audit else {}
        score = outcome.get("native_score") if outcome.get("replayed") else None
        study.require(score in (None, 0, 1), "nonbinary native score")
        rows.append(
            dict(task_id=job["task_id"], world=plan["world_seed"], seed=job["seed"], score=score)
        )
    result = dict(
        path=str(directory),
        actual_plan=actual_plan,
        planned=16,
        observed=sum(r["score"] is not None for r in rows),
        successes=sum(r["score"] == 1 for r in rows),
        attempts=rows,
    )
    result["unknown"] = 16 - result["observed"]
    if audit:
        study.require(
            result["observed"] == sum(g["observed"] for g in audit["groups"].values()),
            "observed denominator differs",
        )
        study.require(
            result["successes"] == sum(g["won"] for g in audit["groups"].values()),
            "native success total differs",
        )
        result["cost"] = audit["physical_cost"]
        result["errors"] = dict(
            sum((Counter(r.get("errors", {})) for r in audit["audits"].values()), Counter())
        )
    return plan, result


def initial_controls() -> dict:
    summary = read(study.ORIGINAL / "SUMMARY.json")
    replay = read(study.ORIGINAL / "CPU-AUDIT.json")
    known = study.ROOT / "textcraft-quantity-matched-seed2026092208-001"
    roots = {"known": known, **{m: study.training(m, 2026092208) for m in study.MODES}}
    rows, init, tokens, plans = {}, {}, {}, {}
    for name, directory in roots.items():
        plan = read(directory / "PLAN.json")
        plans[name] = plan
        path = Path(plan["prepared"]) / "rows.jsonl"
        RECEIPTS[str(path)] = study.sha(path)
        study.require(RECEIPTS[str(path)] == plan["rows_sha256"], "teacher rows changed")
        rows[name] = [json.loads(line) for line in path.read_text().splitlines()]
        init[name] = read(directory / "checkpoint-0000/COMMIT.json")["files"][
            "adapter_model.safetensors"
        ]
        tokens[name] = [
            read(directory / "steps" / f"{i:04d}.json")["target_tokens"] for i in range(1, 24)
        ]
    reference = [
        (r["task_id"], r["target"], [t for t in r["labels"] if t != -100]) for r in rows["known"]
    ]
    for mode in study.MODES:
        study.require(
            [(r["task_id"], r["target"], [t for t in r["labels"] if t != -100]) for r in rows[mode]]
            == reference,
            "optimizer target presentation differs",
        )
        study.require(init[mode] == init["known"], "recorded initialization differs")
        study.require(
            tokens[mode] == tokens["known"] == replay["paired_batch_target_tokens"]["2026092208"],
            "actual batch target denominators differ",
        )
        for key in (
            "seed",
            "model",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "effective_batch",
            "microbatch",
            "planned_updates",
            "lora",
            "dtype",
            "training_recipe_sha256",
        ):
            study.require(
                plans[mode][key] == plans["known"][key], "matched training recipe differs: " + key
            )
    return dict(
        tasks=32,
        rows=366,
        updates=23,
        target_tokens_including_eos=sum(tokens["known"]),
        exact_literal_targets_and_unmasked_label_ids_in_original_row_order=True,
        actual_per_update_target_tokens=tokens["known"],
        initial_adapter_sha256=init["known"],
        initialization_authority="Existing checkpoint-0000 COMMIT hashes, not fresh weight rehash",
        craft_order_and_action_multisets_preserved=True,
        query_visibility=dict(known=32, total_queries=167, stable_visible=167, random_visible=167),
        prompt_tokens={
            name: sum(r["prompt_tokens"] for r in examples) for name, examples in rows.items()
        },
        native_order_matches_discovery_tasks={
            m: summary[m]["native_action_order_matches_discovery_tasks"] for m in study.MODES
        },
        assistance="Offline gold-action scheduler; public query names and stock-feasible "
        "native replay. "
        "Conditioning histories and input-token dose differ; no visibility-only causal isolation.",
    )


def initial() -> dict:
    study.require(study.sha(SNAPSHOT) == SNAPSHOT_SHA, "frozen result cutoff changed")
    snapshot = read(SNAPSHOT)
    cells, comparisons = {}, []
    for world in (42, 50):
        for teacher in ("known", "discovery", *study.MODES):
            if teacher in study.MODES:
                directory = study.ORIGINAL / f"eval-{teacher}-s2026092208-p00-w{world}-raw"
                name = f"order_{teacher}_w{world}"
            else:
                directory = study.baseline(2026092208, 0, world, teacher)
                name = f"base_{'known_recipe' if teacher == 'known' else teacher}_w{world}"
            _, result = native(directory)
            study.require(
                result["observed"] == 16
                and result["successes"] == snapshot["cells"][name]["successes"],
                "fixed native result changed",
            )
            cells[f"{teacher}_w{world}"] = result
        for mode in study.MODES:
            for teacher in ("known", "discovery"):
                comparisons.append(
                    dict(
                        name=f"{mode}_minus_{teacher}_w{world}",
                        result=pair(
                            cells[f"{teacher}_w{world}"]["attempts"],
                            cells[f"{mode}_w{world}"]["attempts"],
                        ),
                    )
                )
    totals = {}
    for teacher in ("known", "discovery", *study.MODES):
        members = [cells[f"{teacher}_w{w}"] for w in (42, 50)]
        totals[teacher] = dict(
            successes=sum(c["successes"] for c in members),
            planned=32,
            calls=sum(c["cost"]["calls"] for c in members),
            completion_tokens=sum(c["cost"]["completion_tokens"] for c in members),
            transport_failures=sum(c["cost"]["failed_calls"] for c in members),
        )
    for mode in study.MODES:
        for teacher in ("known", "discovery"):
            comparisons.append(
                dict(
                    name=f"{mode}_minus_{teacher}_fixed_two_worlds",
                    result=pair(
                        [r for w in (42, 50) for r in cells[f"{teacher}_w{w}"]["attempts"]],
                        [r for w in (42, 50) for r in cells[f"{mode}_w{w}"]["attempts"]],
                    ),
                )
            )
    return dict(
        schema="teaching-order-initial-result-20260928-v1",
        evidence_cutoff=snapshot["cutoff_utc"],
        cells=cells,
        totals=totals,
        comparisons=comparisons,
        controls=initial_controls(),
        caveat="Eight exposed root identities, two correlated worlds, two rollout seeds and "
        "one fit seed. Bootstrap resamples the eight roots, holding worlds/repeats together; "
        "shared recipes remain correlated. No discovery-superiority, unseen-structure or "
        "general novelty claim. Fixed dose controls are still separate pending evidence.",
    )


def replication(root: Path) -> dict:
    prepared = read(root / "PREPARATION.json")
    cells, comparisons = {}, []
    for spec in prepared["cells"]:
        tag = f"{spec['mode']}_s{spec['seed']}_p{spec['panel']:02d}_w{spec['world']}"
        baseline_plans = {}
        for teacher, directory in spec["baselines"].items():
            baseline_plans[teacher], cells[tag + ":" + teacher] = native(Path(directory))
            study.require(
                study.sha(Path(directory) / "PLAN.json") == spec["baseline_plan_sha256"][teacher],
                "explicit baseline changed",
            )
        actual, cells[tag] = native(Path(spec["output"]), baseline_plans["discovery"])
        if cells[tag]["actual_plan"]:
            for field in study.PAIRED_FIELDS:
                study.require(
                    actual[field] == baseline_plans["discovery"][field], "paired field: " + field
                )
            expected = prepared["training"][f"{spec['mode']}:{spec['seed']}"]
            study.check_binding(actual["adapter"], expected)
            study.require(actual["teacher"] == spec["mode"], "wrong repaired teacher")
        for teacher in ("known", "discovery"):
            comparisons.append(
                dict(
                    name=tag + "_minus_" + teacher,
                    result=pair(cells[tag + ":" + teacher]["attempts"], cells[tag]["attempts"]),
                )
            )
    return dict(
        schema="teaching-order-fixed-replication-result-20260928-v1",
        cells=cells,
        comparisons=comparisons,
        caveat="Fixed two-slice replication: second fit seed on "
        "panel00, first fit seed on panel01. Do not pool these as independent trials or "
        "select an ordering. Panel01 was previously exposed in breadth analysis; new "
        "root items, not a held-out recipe universe. Missing endpoints/outcomes remain unknown.",
    )


def markdown(report: dict) -> str:
    lines = ["# Teaching-order repair readout", "", "Cutoff: " + report["cutoff_utc"], ""]
    if "evidence_cutoff" in report:
        lines += ["Fixed native-evidence cutoff: " + report["evidence_cutoff"], ""]
    if "totals" in report:
        lines += [
            "The matched-target repair recovers much of the known-teacher gap on this fixed "
            "exploratory slice. Stable matches discovery's total, not demonstrated superiority.",
            "",
            "| Teacher | Successes | Calls | Output tokens |",
            "| --- | ---: | ---: | ---: |",
        ]
        for teacher, totals in report["totals"].items():
            lines.append(
                f"| {teacher} | {totals['successes']}/32 | {totals['calls']} "
                f"| {totals['completion_tokens']} |"
            )
    lines += [
        "",
        "| Paired contrast | Wins/losses/ties/unknown | Difference pp | Root-bootstrap 95% pp |",
        "| --- | --- | ---: | --- |",
    ]
    for comparison in report["comparisons"]:
        r = comparison["result"]
        count = "/".join(str(r[k]) for k in ("wins", "losses", "ties", "unknown_pairs"))
        point = "unknown" if r["difference"] is None else f"{100 * r['difference']:+.2f}"
        ci = r["task_cluster_95"]
        interval = "unknown" if ci is None else str([round(100 * v, 2) for v in ci])
        lines.append(f"| {comparison['name']} | {count} | {point} | {interval} |")
    if "controls" in report:
        c = report["controls"]
        lines += [
            "",
            "Controls: 32 TRAIN tasks, 366 rows, exactly 8,820 label tokens including EOS, "
            "23 updates. Both repairs preserve literal targets and unmasked label IDs in the "
            "original known-teacher row order; all 23 actual minibatch target denominators "
            "match. Checkpoint-0 COMMIT receipts record identical initial LoRA weights. "
            "Base, fit seed, optimizer recipe and original craft order match. Public-name "
            "queries increase from 32/167 to 167/167. Both repaired native sequences match "
            "discovery on only 8/32 TRAIN tasks. Prompt-token totals differ: known "
            f"{c['prompt_tokens']['known']:,}, stable {c['prompt_tokens']['stable_visible']:,}, "
            f"random {c['prompt_tokens']['random_visible']:,}.",
            "",
            c["assistance"],
        ]
        pooled = next(
            c["result"]
            for c in report["comparisons"]
            if c["name"] == "stable_visible_minus_known_fixed_two_worlds"
        )
        lines += [
            "",
            "| Root task ID | Known successes /4 | Stable successes /4 |",
            "| --- | ---: | ---: |",
        ]
        for row in pooled["per_task"]:
            lines.append(
                f"| {row['task_id']} | {row['left_successes']} | {row['right_successes']} |"
            )
    unknown = sum(c["result"]["unknown_pairs"] for c in report["comparisons"])
    decision = (
        "Await the remaining fixed native audits; missing outcomes are not failures or evidence "
        "for an efficacy decision."
        if unknown
        else "Retain both orderings; inspect fit-seed and additional-root slices separately "
        "with fixed 46/69-update dose controls. A replicated rescue supports this trace-repair "
        "package, not isolated visibility causality; mixed replication should narrow the claim "
        "before expansion."
    )
    lines += ["", report["caveat"], "", "Next decision: " + decision, ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=study.OUTPUT)
    parser.add_argument("--replication", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replication(args.root) if args.replication else initial()
    result.update(
        cutoff_utc=datetime.now(timezone.utc).isoformat(),
        receipts=RECEIPTS,
        source_sha256={
            str(p): study.sha(p) for p in (Path(__file__), Path(study.__file__), PAIR_SOURCE)
        },
    )
    study.save(args.output, result)
    with args.output.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
    print(json.dumps(dict(output=str(args.output), comparisons=len(result["comparisons"]))))
