"""Fixed teaching-controls synthesis using saved native audits, never scientific runtimes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "analysis-teaching-20260929-001"
SEED = 2026092901
IDENTITY = ("task_id", "panel", "fit_seed", "world", "rollout_seed")
RECEIPTS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(path):
    value = str(path)
    return value.replace(str(ROOT), "R").replace(str(EXPERIMENTS), "E")


def read(path):
    RECEIPTS[portable(path)] = sha(path)
    return json.loads(Path(path).read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    RECEIPTS[portable(path)] = sha(path)
    return result


def paired(left: list[dict], right: list[dict]) -> dict:
    def keyed(rows):
        values = {tuple(r[k] for k in IDENTITY): r["score"] for r in rows}
        require(len(values) == len(rows), "duplicate paired identities")
        return values

    a, b = keyed(left), keyed(right)
    require(a.keys() == b.keys(), "different paired identities")
    require(bool(a), "empty contrast")
    rows = [
        dict(
            zip(IDENTITY, key, strict=True),
            left=a[key],
            right=b[key],
            delta=None if a[key] is None or b[key] is None else b[key] - a[key],
        )
        for key in sorted(a)
    ]
    known = [r for r in rows if r["delta"] is not None]
    wins = sum(r["delta"] > 0 for r in known)
    losses = sum(r["delta"] < 0 for r in known)
    unknown = len(rows) - len(known)
    by_panel = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_panel[row["panel"]][row["task_id"]].append(row["delta"])
    require(
        sum(len(tasks) for tasks in by_panel.values()) == len({r["task_id"] for r in rows}),
        "root identity crosses panel strata",
    )
    interval = None
    if not unknown:
        # Balanced repetitions within a panel; worlds/fit/rollout repeats stay in one root block.
        for tasks in by_panel.values():
            require(
                len({len(values) for values in tasks.values()}) == 1,
                "unequal per-root repetition within a panel",
            )
        strata = [
            (
                sum(map(len, tasks.values())) / len(rows),
                [mean(values) for _, values in sorted(tasks.items())],
            )
            for _, tasks in sorted(by_panel.items())
        ]
        rng = random.Random(SEED)
        draws = sorted(
            sum(weight * mean(rng.choices(values, k=len(values))) for weight, values in strata)
            for _ in range(20000)
        )
        interval = [draws[500], draws[19500]]
    return dict(
        planned_pairs=len(rows),
        known_pairs=len(known),
        unknown_pairs=unknown,
        wins=wins,
        losses=losses,
        ties=len(known) - wins - losses,
        difference=(wins - losses) / len(rows) if not unknown else None,
        difference_bounds=[(wins - losses + s * unknown) / len(rows) for s in (-1, 1)],
        root_clusters=sum(len(t) for t in by_panel.values()),
        strata={str(p): len(t) for p, t in sorted(by_panel.items())},
        root_cluster_95=interval,
        bootstrap_seed=SEED,
        bootstrap_draws=20000,
        rows=rows,
    )


def gather():
    controls = module("synthesis_controls", EXPERIMENTS / "controls_readout_20260928/readout.py")
    replication = module(
        "synthesis_replication", EXPERIMENTS / "teaching_replication_20260928/report.py"
    )
    cells = {}

    def add(path, family, teacher, mode, step, panel, seed, plan, scores, cost, errors, observed):
        key = portable(path)
        if key in cells:
            return
        jobs = plan["jobs"]
        rows = [
            dict(
                task_id=j["task_id"],
                panel=panel,
                fit_seed=seed,
                world=plan["world_seed"],
                rollout_seed=j["seed"],
                score=scores[j["task_id"], j["seed"]],
            )
            for j in jobs
        ]
        cells[key] = dict(
            artifact=key,
            family=family,
            teacher=teacher,
            interface=mode,
            checkpoint=step,
            panel=panel,
            fit_seed=seed,
            world=plan["world_seed"],
            planned=len(jobs),
            observed=observed,
            unknown=len(jobs) - observed,
            successes=sum(r["score"] == 1 for r in rows),
            attempts=rows,
            cost=cost,
            errors=errors,
            actor=portable(plan["adapter"]["path"]) if plan.get("adapter") else None,
            actor_sha256=plan["adapter"]["sha256"] if plan.get("adapter") else None,
            training_seconds=plan.get("adapter", {}).get("state", {}).get("training_seconds"),
        )

    specs, fixed_contrasts = controls.inventory(ROOT)
    loaded = {}
    for spec in specs.values():
        cell = controls.load_cell(spec)
        loaded[spec["name"]] = cell
        require(cell["_plan"] is not None, "fixed controls PLAN missing")
        RECEIPTS.update({portable(k): v for k, v in cell["receipts"].items()})
        teacher = {
            "public": "discovery",
            "quantity_corrected_original": "known",
            "known_recipe": "known",
        }.get(spec["teacher"], spec["teacher"])
        add(
            spec["path"],
            spec["family"],
            teacher,
            spec["mode"],
            spec["step"],
            0,
            2026092208,
            cell["_plan"],
            cell["_values"],
            cell.get("physical_cost"),
            cell.get("errors"),
            cell["observed"],
        )
    for contrast in fixed_contrasts:
        controls.compare_cells(
            loaded[contrast["left"]], loaded[contrast["right"]], contrast["same_actor"]
        )
    # Reuse the frozen adapter's lineage and per-world input checks for all added cells.
    results = replication.replication(replication.study.OUTPUT)
    prepared = read(replication.study.OUTPUT / "PREPARATION.json")
    for spec in prepared["cells"]:
        tag = f"{spec['mode']}_s{spec['seed']}_p{spec['panel']:02d}_w{spec['world']}"
        references = [(spec["mode"], spec["output"], results["cells"][tag])]
        references += [
            (t, p, results["cells"][tag + ":" + t]) for t, p in spec["baselines"].items()
        ]
        for teacher, directory, cell in references:
            plan = read(Path(directory) / "PLAN.json")
            scores = {(r["task_id"], r["seed"]): r["score"] for r in cell["attempts"]}
            add(
                directory,
                "qwen",
                teacher,
                "raw",
                23,
                spec["panel"],
                spec["seed"],
                plan,
                scores,
                cell.get("cost"),
                cell.get("errors"),
                cell["observed"],
            )
    RECEIPTS.update({portable(k): v for k, v in replication.RECEIPTS.items()})
    for path in (Path(replication.study.__file__), controls.PAIR_SOURCE):
        RECEIPTS[portable(path)] = sha(path)
    return cells


def training_controls():
    original = read(EXPERIMENTS / "teaching_replication_20260928/FINDING.json")["controls"]
    summary = {"original_fit": original}
    for seed in (2026092208, 2026092291):
        directory = ROOT / f"textcraft-quantity-matched-seed{seed}-001"
        directories = {"known": directory}
        order_root = ROOT / (
            "textcraft-teaching-order-20260928-001"
            if seed == 2026092208
            else "textcraft-teaching-replication-20260928-001"
        )
        directories.update(
            {m: order_root / f"train-{m}-seed{seed}" for m in ("stable_visible", "random_visible")}
        )
        initial, denominators, plans = {}, {}, {}
        for teacher, path in directories.items():
            plans[teacher] = read(path / "PLAN.json")
            require(plans[teacher]["seed"] == seed, "wrong training seed")
            initial[teacher] = read(path / "checkpoint-0000/COMMIT.json")["files"][
                "adapter_model.safetensors"
            ]
            denominators[teacher] = [
                read(path / "steps" / f"{i:04d}.json")["target_tokens"] for i in range(1, 24)
            ]
        require(len(set(initial.values())) == 1, "recorded initialization differs")
        require(
            all(v == denominators["known"] for v in denominators.values()), "target dose differs"
        )
        require(sum(denominators["known"]) == 8820, "wrong label-token dose")
        summary[str(seed)] = dict(
            initial_adapter_sha256=initial["known"],
            actual_target_tokens_per_update=denominators["known"],
            rows_sha256={t: p["rows_sha256"] for t, p in plans.items()},
            matched_recorded_initialization=True,
            matched_per_update_answer_token_dose=True,
        )
    return summary


def cohort(cells, **criteria):
    selected = [c for c in cells.values() if all(c[k] == v for k, v in criteria.items())]
    require(bool(selected), "empty fixed cohort")
    rows = [r for c in selected for r in c["attempts"]]
    cost_known = all(c["cost"] is not None for c in selected)
    costs = None
    if cost_known:
        costs = {
            k: sum(c["cost"].get(k, 0) for c in selected)
            for k in (
                "calls",
                "completion_tokens",
                "prompt_tokens",
                "native_service_seconds",
                "failed_calls",
            )
        }
    return dict(
        criteria=criteria,
        cells=[c["artifact"] for c in selected],
        planned=len(rows),
        observed=sum(c["observed"] for c in selected),
        successes=sum(c["successes"] for c in selected),
        unknown=sum(c["unknown"] for c in selected),
        attempts=rows,
        cost=costs,
    )


def analyze():
    cells = gather()
    cohorts, comparisons = {}, {}
    for name, panel, seed in (
        ("pilot", 0, 2026092208),
        ("added_fit", 0, 2026092291),
        ("added_roots", 1, 2026092208),
    ):
        for teacher in ("known", "discovery", "stable_visible", "random_visible"):
            cohorts[f"{name}:{teacher}"] = cohort(
                cells,
                family="qwen",
                interface="raw",
                checkpoint=23,
                panel=panel,
                fit_seed=seed,
                teacher=teacher,
            )
        for mode in ("stable_visible", "random_visible"):
            for base in ("known", "discovery"):
                comparisons[f"{name}:{mode}-{base}"] = paired(
                    cohorts[f"{name}:{base}"]["attempts"], cohorts[f"{name}:{mode}"]["attempts"]
                )
    for teacher in ("known", "discovery", "stable_visible", "random_visible"):
        members = [cohorts[f"{phase}:{teacher}"] for phase in ("added_fit", "added_roots")]
        costs = (
            {k: sum(c["cost"][k] for c in members) for k in members[0]["cost"]}
            if all(c["cost"] for c in members)
            else None
        )
        cohorts[f"added_combined:{teacher}"] = dict(
            planned=64,
            observed=sum(c["observed"] for c in members),
            successes=sum(c["successes"] for c in members),
            unknown=sum(c["unknown"] for c in members),
            cost=costs,
            attempts=[r for c in members for r in c["attempts"]],
        )
    for mode in ("stable_visible", "random_visible"):
        for base in ("known", "discovery"):
            comparisons[f"added_combined:{mode}-{base}"] = paired(
                cohorts[f"added_combined:{base}"]["attempts"],
                cohorts[f"added_combined:{mode}"]["attempts"],
            )
    for teacher in ("known", "discovery"):
        for step in (46, 69):
            name = f"dose:{teacher}:{step}"
            cohorts[name] = cohort(
                cells,
                family="qwen",
                interface="raw",
                checkpoint=step,
                panel=0,
                fit_seed=2026092208,
                teacher=teacher,
            )
            comparisons[name + "-23"] = paired(
                cohorts[f"pilot:{teacher}"]["attempts"], cohorts[name]["attempts"]
            )
            for mode in ("stable_visible", "random_visible"):
                comparisons[f"pilot:{mode}-{teacher}{step}"] = paired(
                    cohorts[name]["attempts"], cohorts[f"pilot:{mode}"]["attempts"]
                )
    for teacher in ("known", "discovery"):
        for interface in ("raw", "binder"):
            name = f"phi:{teacher}:{interface}"
            cohorts[name] = cohort(
                cells,
                family="phi",
                interface=interface,
                checkpoint=23,
                panel=0,
                fit_seed=2026092208,
                teacher=teacher,
            )
        comparisons[f"phi:{teacher}:binder-raw"] = paired(
            cohorts[f"phi:{teacher}:raw"]["attempts"], cohorts[f"phi:{teacher}:binder"]["attempts"]
        )
    for interface in ("raw", "binder"):
        comparisons[f"phi:{interface}:discovery-known"] = paired(
            cohorts[f"phi:known:{interface}"]["attempts"],
            cohorts[f"phi:discovery:{interface}"]["attempts"],
        )
    return dict(
        schema="teaching-synthesis-20260929-v1",
        cutoff_utc=datetime.now(timezone.utc).isoformat(),
        cells=cells,
        cohorts=cohorts,
        comparisons=comparisons,
        training_controls=training_controls(),
        method="20,000 paired root-cluster bootstrap draws; fixed worlds/fit/rollout repeats "
        "stay together. Added-combined resamples eight roots within each panel, retaining "
        "panel weights. Intervals describe the exposed fixed design, not benchmark-level "
        "uncertainty; shared recipes can correlate roots. Unknown pairs suppress estimates "
        "and intervals. No best endpoint.",
        audit_scope="Existing native audit authority, with small PLAN/receipt lineage checks; "
        "not fresh replay or model/trajectory rehash. Actual Qwen16/Phi8 slots per cell.",
        receipt_sha256=RECEIPTS,
        source_sha256=sha(Path(__file__)),
    )


def markdown(report):
    lines = [
        "# Teaching controls: synthesis",
        "",
        "Cutoff: " + report["cutoff_utc"],
        "",
        "All 40 unique cells are natively audited: 576/576 actual attempts, zero unknowns "
        "and zero failed transport calls. Native action/schema errors remain in costs/outcomes.",
        "",
        "The strongest current interpretation is improved learnability at the tested optimization "
        "budget. Repair gains reproduce across a second fit seed and additional root goals; extra "
        "training partially rescues the original known-recipe teacher. Phi tests the original "
        "teacher/interface packages, not transfer of the repair method.",
        "",
        "## Repair replication (both fixed worlds)",
        "",
        "| Measurement stratum | Known | Discovery | Stable repair | Random repair |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phase in ("pilot", "added_fit", "added_roots", "added_combined"):
        values = []
        for teacher in ("known", "discovery", "stable_visible", "random_visible"):
            c = report["cohorts"][f"{phase}:{teacher}"]
            values.append(
                f"{c['successes']}/{c['planned']}"
                + (f" ({c['unknown']} unknown)" if c["unknown"] else "")
            )
        lines.append("| " + phase + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "Pilot = panel00/fit2208. Added-fit = panel00/fit2291; added-roots = panel01/fit2208. "
        "The added measurements were fixed after the pilot but reuse exposed breadth panels. "
        "Added-combined has 16 roots; each individual stratum has eight. Do not count worlds, "
        "fits or rollout repeats as new compositional problems.",
        "",
        "| Selected paired contrast | Wins/losses/ties/unknown | Difference pp | Root 95% pp |",
        "| --- | --- | ---: | --- |",
    ]
    selected = [
        f"{p}:{m}-known"
        for p in ("pilot", "added_fit", "added_roots", "added_combined")
        for m in ("stable_visible", "random_visible")
    ]
    selected += [
        "added_combined:stable_visible-discovery",
        "added_combined:random_visible-discovery",
        "dose:known:46-23",
        "dose:known:69-23",
        "pilot:stable_visible-known69",
        "pilot:random_visible-known69",
        "phi:raw:discovery-known",
        "phi:binder:discovery-known",
        "phi:discovery:binder-raw",
        "phi:known:binder-raw",
    ]
    for name in selected:
        c = report["comparisons"][name]
        counts = "/".join(str(c[k]) for k in ("wins", "losses", "ties", "unknown_pairs"))
        delta = "unknown" if c["difference"] is None else f"{100 * c['difference']:+.2f}"
        ci = (
            "unknown"
            if c["root_cluster_95"] is None
            else str([round(100 * v, 2) for v in c["root_cluster_95"]])
        )
        lines.append(f"| {name} | {counts} | {delta} | {ci} |")
    lines += [
        "",
        "## Dose, cost and model family",
        "",
        "| Fixed cohort | Success/planned | Native calls | Output tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    names = [
        "pilot:known",
        "dose:known:46",
        "dose:known:69",
        "pilot:discovery",
        "dose:discovery:46",
        "dose:discovery:69",
        "pilot:stable_visible",
        "pilot:random_visible",
        *[
            f"added_combined:{t}"
            for t in ("known", "discovery", "stable_visible", "random_visible")
        ],
        *[f"phi:{t}:{m}" for t in ("known", "discovery") for m in ("raw", "binder")],
    ]
    for name in names:
        c = report["cohorts"][name]
        cost = c["cost"] or {}
        lines.append(
            f"| {name} | {c['successes']}/{c['planned']} | {cost.get('calls', 'unknown')} "
            f"| {cost.get('completion_tokens', 'unknown')} |"
        )
    lines += [
        "",
        "Known continuation reaches 8/32 at both fixed 46- and 69-update endpoints, from "
        "1/32 at update23. World-level outcomes change from 3/16+5/16 to 5/16+3/16, so "
        "selecting the better checkpoint per world would inflate evidence. Discovery moves "
        "14→15→17/32. Repair's 23-update pilot reaches 14/32 stable or 12/32 random. This "
        "supports a learning-efficiency advantage at tested doses, not an irreparable teacher, "
        "an asymptotic advantage, or a dose-independent causal effect. Cost includes every "
        "audited attempt, not only successes. Fewer calls can also reflect stopping or errors.",
        "Known cumulative training time was 182→364→545 seconds at these fixed endpoints; "
        "stable repair took 184 seconds for update23. These are trainer-state durations, "
        "not owner wall time. The direct repair-versus-continued intervals remain broad.",
        "",
        "Both repair fit seeds preserve recorded initialization, 366 targets in known-teacher "
        "row order, all 23 minibatch target-token denominators, and 8,820 answer tokens including "
        "EOS. Original crafting order is retained; schedules use offline gold actions while "
        "querying publicly visible names. Input history and input-token dose change "
        "(414,682 known; 431,274 stable; 435,710 random). The random schedule is not a globally "
        "optimal teacher; both orderings remain in the report without selecting the winner.",
        "",
        "Phi: discovery scores 4/16 raw and 6/16 binder, versus known 1/16 raw and 0/16 "
        "binder. Eight roots and one rollout seed/world give little precision; both binder "
        "gains are VAL263 in different worlds. This supports limited original-package dependence "
        "across model families, not repair-method transfer. There is no Phi repaired actor yet. "
        "Equal 23 updates across families do not equalize tokenizer, labels, LoRA dimensions "
        "or compute. Failed base probes are excluded from these valid trained-cell contrasts.",
        "",
        "## Next two comparisons",
        "",
        "1. **Phi stable-visible repair**, same 366 known-teacher targets/minibatch order, fixed "
        "seed/update23 and two raw world readouts (8 roots/one rollout seed each). Pair with "
        "existing Phi-known/discovery. This directly tests repair-method transfer; no rescue "
        "should narrow a family-general claim. Add binder only if a competent repaired raw "
        "policy leaves an execution-interface question.",
        "2. **A fixed fresh-root/world repair test**, conditional on Phi's result: compare the "
        "existing known, discovery and stable actors at one predetermined fit/checkpoint. "
        "Before rollouts, select eight roots outside all exposed breadth/pilot goals and all "
        "SFT query products, plus an unused world seed; verify feasibility from compact "
        "manifests. Retain the same 16-attempt budget and report lower-recipe/graph overlap. "
        "This tests whether repair extends beyond the exposed problem family, not wholly "
        "unseen composition. If no such root slice exists, say so and prioritize the narrower "
        "cross-world claim rather than relabeling shared recipes as unseen structure. A null "
        "Phi repair result should narrow the family-general claim before more dose grids.",
        "",
        "## Limits and authority",
        "",
        report["method"],
        "",
        report["audit_scope"],
        " All comparisons are exploratory and adaptive; no multiplicity-adjusted claim, "
        "absence-of-effect equivalence, unseen-structure claim, or universal novelty claim. "
        "Root disjointness does not remove shared lower-level recipes. JSON retains every "
        "cell, paired root count, cost, explicit checkpoint and small-receipt hash. Per-attempt "
        "scores are retained once per cell so all paired rows can be reconstructed.",
        "",
    ]
    return "\n".join(lines)


def compact(report):
    """Keep attempt scores once per cell and paired root counts once per contrast."""
    for cell in report["cells"].values():
        cell["attempts"] = [[r["task_id"], r["rollout_seed"], r["score"]] for r in cell["attempts"]]
    for row in report["cohorts"].values():
        row.pop("attempts")
    for comparison in report["comparisons"].values():
        grouped = defaultdict(list)
        for row in comparison.pop("rows"):
            grouped[row["panel"], row["task_id"]].append(row)
        comparison["root_pairs"] = [
            [
                panel,
                task,
                len(rows),
                sum(r["left"] == 1 for r in rows),
                sum(r["right"] == 1 for r in rows),
                sum(r["delta"] is None for r in rows),
            ]
            for (panel, task), rows in sorted(grouped.items())
        ]
    report["table_columns"] = dict(
        cell_attempts=["task_id", "rollout_seed", "score"],
        paired_roots=["panel", "task_id", "planned", "left_success", "right_success", "unknown"],
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compact(analyze())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with args.output.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
    print(
        json.dumps(
            dict(
                output=str(args.output),
                cells=len(result["cells"]),
                audited=sum(c["observed"] for c in result["cells"].values()),
                unknown=sum(c["unknown"] for c in result["cells"].values()),
            )
        )
    )
