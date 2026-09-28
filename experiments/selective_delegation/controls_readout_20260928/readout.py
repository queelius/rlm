"""One-shot fixed controls readout from native audit authority; CPU/stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
PAIR_SOURCE = Path(__file__).resolve().parent.parent / "inventory_bottleneck_20260928/compare.py"
INPUT_FIELDS = ("tasks_sha256", "manifest_sha256", "world_sha256", "world_seed")
PAIRED_FIELDS = INPUT_FIELDS + (
    "model_manifest_sha256",
    "sampling",
    "max_global_calls",
    "max_global_output_tokens",
    "max_new_tokens",
    "input_plus_output_limit",
    "truncation",
    "budget_seconds",
)
ORIGINALS = {
    "discovery": "textcraft-public-discovery-sft-001",
    "known_recipe": "textcraft-quantity-matched-seed2026092208-001",
}
AUDIT_NAMES = {"qwen": "NATIVE-AUDIT.json", "phi": "PHI-AUDIT.json"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def paired_math():
    # This existing module imports only the standard library; never a scientific runtime.
    spec = importlib.util.spec_from_file_location("controls_paired_math", PAIR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.paired


def inventory(root: Path) -> tuple[dict, list]:
    cells, contrasts = {}, []

    def cell(name, path, family, kind, teacher, mode, world, step, training, original=None):
        cells[name] = dict(
            name=name,
            path=str(root / path),
            family=family,
            kind=kind,
            teacher=teacher,
            mode=mode,
            world=world,
            step=step,
            training=str(root / training),
            checkpoint=str(root / training / f"checkpoint-{step:04d}"),
            baseline_checkpoint=str(root / ORIGINALS[original] / "checkpoint-0023")
            if original
            else None,
            schedule_path=str(root / f"textcraft-breadth-p00-w{world}-soriginal-raw-001/PLAN.json"),
        )
        return name

    def contrast(name, family, left, right, same_actor=False):
        contrasts.append(
            dict(
                name=name,
                family=family,
                left=left,
                right=right,
                same_actor=same_actor,
                direction="right minus left",
            )
        )

    for world in (42, 50):
        baseline = {}
        for teacher, native, prefix in (
            ("discovery", "public", ""),
            ("known_recipe", "quantity_corrected_original", "corrected-"),
        ):
            baseline[teacher] = cell(
                f"base_{teacher}_w{world}",
                f"textcraft-breadth-p00-w{world}-soriginal-{prefix}raw-001",
                "qwen",
                "baseline",
                native,
                "raw",
                world,
                23,
                ORIGINALS[teacher],
            )
        for mode in ("stable_visible", "random_visible"):
            root_order = "textcraft-teaching-order-20260928-001"
            right = cell(
                f"order_{mode}_w{world}",
                f"{root_order}/eval-{mode}-s2026092208-p00-w{world}-raw",
                "qwen",
                "order",
                mode,
                "raw",
                world,
                23,
                f"{root_order}/train-{mode}-seed2026092208",
            )
            for reference in ("known_recipe", "discovery"):
                label = "known" if reference == "known_recipe" else reference
                contrast(
                    f"order_{mode}_minus_{label}_w{world}", "order", baseline[reference], right
                )
        for teacher in ORIGINALS:
            for step in (46, 69):
                root_dose = "textcraft-teaching-dose-20260928-001"
                right = cell(
                    f"dose_{teacher}_cp{step}_w{world}",
                    f"{root_dose}/eval-{teacher}-cp{step}-p00-w{world}-raw",
                    "qwen",
                    "dose",
                    teacher,
                    "raw",
                    world,
                    step,
                    f"{root_dose}/train-{teacher}",
                    teacher,
                )
                contrast(
                    f"dose_{teacher}_cp{step}_minus_cp23_w{world}", "dose", baseline[teacher], right
                )
        phi = {}
        for teacher in ("discovery", "known"):
            for mode in ("raw", "binder"):
                phi[teacher, mode] = cell(
                    f"phi_{teacher}_{mode}_w{world}",
                    f"textcraft-phi-{teacher}-{mode}-w{world}-20260928-001",
                    "phi",
                    "phi",
                    teacher,
                    mode,
                    world,
                    23,
                    f"textcraft-phi-{teacher}-sft-20260928-001",
                )
            contrast(
                f"phi_{teacher}_binder_minus_raw_w{world}",
                "phi",
                phi[teacher, "raw"],
                phi[teacher, "binder"],
                True,
            )
        for mode in ("raw", "binder"):
            contrast(
                f"phi_{mode}_discovery_minus_known_w{world}",
                "phi",
                phi["known", mode],
                phi["discovery", mode],
            )
    return cells, contrasts


def keys(jobs: list[dict]) -> set[tuple[str, int]]:
    result = {(job["task_id"], job["seed"]) for job in jobs}
    require(len(result) == len(jobs), "duplicate (task_id, seed) PLAN slot")
    require(all(job["policy"] == "flat" for job in jobs), "only fixed flat-policy slots")
    return result


def load_cell(spec: dict) -> dict:
    receipts = {}

    def read(path):
        raw = Path(path).read_bytes()
        receipts[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    reference = spec.get("expected_inputs") or read(Path(spec["schedule_path"]))
    expected = spec.get("expected_jobs") or reference["jobs"]
    if spec["family"] == "phi":
        expected = [j for j in expected if j["seed"] == 2026092204]
    directory = Path(spec["path"])
    plan_path = directory / "PLAN.json"
    audit_path = directory / AUDIT_NAMES[spec["family"]]
    plan = read(plan_path) if plan_path.exists() else None
    jobs = plan["jobs"] if plan else expected
    slot_keys = keys(jobs)
    values = dict.fromkeys(slot_keys)
    result = dict(
        name=spec["name"],
        path=spec["path"],
        family=spec["family"],
        teacher=spec["teacher"],
        mode=spec["mode"],
        world=spec["world"],
        checkpoint_step=spec["step"],
        expected_checkpoint=spec["checkpoint"],
        planned=len(jobs),
        observed=0,
        successes=0,
        unknown=len(jobs),
        success_rate_bounds=[0, 1],
        status="awaiting_native_audit",
        denominator_source="actual_PLAN" if plan else "frozen_schedule_no_actual_PLAN",
        receipts=receipts,
        terminals={},
        _values=values,
        _plan=plan,
        _jobs=jobs,
    )
    for terminal in sorted(directory.glob("TERMINAL-*.json")):
        result["terminals"][str(terminal)] = read(terminal)
    if not audit_path.exists():
        result["status"] = "awaiting_native_audit" if plan else "not_started"
        return result
    require(plan is not None, "native audit without actual PLAN")
    audit = read(audit_path)
    bound = (
        audit.get("plan_sha256") if spec["family"] == "phi" else audit["sha256"].get(str(plan_path))
    )
    require(bound == receipts[str(plan_path)], "native-audited PLAN hash differs")
    require(slot_keys == keys(expected), "PLAN task/seed schedule differs")
    require(plan.get("planned_episodes", len(jobs)) == len(jobs), "PLAN count differs")
    require(plan["teacher"] == spec["teacher"], "wrong teacher for explicit cell")
    require(plan["world_seed"] == spec["world"], "wrong world for explicit cell")
    require(plan.get("assistance", plan.get("execution_mode")) == spec["mode"], "wrong interface")
    for field in INPUT_FIELDS:
        require(plan[field] == reference[field], "fixed input differs: " + field)
    training_path = Path(spec["training"]) / "PLAN.json"
    training = read(training_path)
    actor = plan.get("adapter")
    require(bool(actor) and actor["path"] == spec["checkpoint"], "wrong fixed actor checkpoint")
    require(actor["state"]["step"] == spec["step"], "wrong fixed checkpoint step")
    require(
        actor["training_plan_sha256"] == receipts[str(training_path)],
        "training PLAN lineage differs",
    )
    require(
        actor["training_rows_sha256"] == training["rows_sha256"], "training row lineage differs"
    )
    require(training["seed"] == 2026092208, "wrong fixed fit seed")
    require(plan["model"] == training["model"], "base model lineage differs")
    if spec["kind"] == "dose":
        original = training["original"]
        require(original["checkpoint"] == spec["baseline_checkpoint"], "wrong original checkpoint")
        original_path = Path(spec["baseline_checkpoint"]).parent / "PLAN.json"
        original_plan = read(original_path)
        require(
            original["plan_sha256"] == receipts[str(original_path)], "original PLAN lineage differs"
        )
        require(
            training["rows_sha256"] == original_plan["rows_sha256"], "dose changed teacher rows"
        )
        require(
            actor["original_training_plan_sha256"] == receipts[str(original_path)],
            "dose actor lineage differs",
        )
        require(plan["cumulative_training_updates"] == spec["step"], "dose label differs")
        result["original_actor"] = dict(
            path=original["checkpoint"],
            commit_sha256=original["checkpoint_commit_sha256"],
            sha256=original["checkpoint_files"]["adapter_model.safetensors"],
        )
    result["actor"] = {
        k: actor[k]
        for k in ("path", "sha256", "commit_sha256", "training_plan_sha256", "training_rows_sha256")
    }
    if spec["family"] == "phi":
        require(
            audit.get("schema") == "phi-textcraft-native-audit-20260928-v1",
            "wrong Phi audit schema",
        )
        require(audit.get("passed") is True, "Phi native audit not passed")
        for field in ("teacher", "assistance", "world_seed"):
            require(audit[field] == plan[field], "Phi audit " + field + " differs")
        require(len(audit["episodes"]) == len(jobs), "Phi native audit count differs")
        checked = audit["episodes"]
        for job, row in zip(jobs, checked, strict=True):
            require(row["task_id"] == job["task_id"], "Phi audit/PLAN task order differs")
    else:
        require(
            not audit["unresolved_starts"] and not audit["calls_without_episode"],
            "unresolved native calls",
        )
        require(
            set(audit["audits"]) <= {j["episode_id"] for j in jobs},
            "unplanned native audit episode",
        )
        checked = [audit["audits"].get(j["episode_id"], {}) for j in jobs]
        require(
            sum(g["planned"] for g in audit["groups"].values()) == len(jobs),
            "native audit planned count differs",
        )
    errors = Counter()
    for job, row in zip(jobs, checked, strict=True):
        observed = row.get("replayed") is True and row.get("observed", True)
        if observed:
            require(row["native_score"] in (0, 1), "nonbinary audited score")
            values[job["task_id"], job["seed"]] = row["native_score"]
        errors.update(row.get("errors", {}))
    observed = sum(value is not None for value in values.values())
    successes = sum(value == 1 for value in values.values())
    reported = (
        audit["observed"]
        if spec["family"] == "phi"
        else sum(g["observed"] for g in audit["groups"].values())
    )
    require(observed == reported, "native audit observed count differs")
    reported_wins = (
        audit["successes"]
        if spec["family"] == "phi"
        else sum(g["won"] for g in audit["groups"].values())
    )
    require(successes == reported_wins, "native audit success count differs")
    result.update(
        observed=observed,
        successes=successes,
        unknown=len(jobs) - observed,
        success_rate_bounds=[successes / len(jobs), (successes + len(jobs) - observed) / len(jobs)],
        status="audited_complete" if observed == len(jobs) else "audited_partial",
        physical_cost=audit["physical_cost"],
        errors=dict(errors),
    )
    return result


def compare_cells(left: dict, right: dict, same_actor: bool = False) -> dict:
    require(set(left["_values"]) == set(right["_values"]), "paired task/seed slots differ")
    if left["_plan"] and right["_plan"]:
        for field in PAIRED_FIELDS:
            require(left["_plan"][field] == right["_plan"][field], "paired field differs: " + field)
    if same_actor and "actor" in left and "actor" in right:
        require(left["actor"] == right["actor"], "binder comparison changed actor")
    if "original_actor" in right and "actor" in left:
        require(
            all(left["actor"][k] == v for k, v in right["original_actor"].items()),
            "dose original actor differs from behavioral baseline",
        )
    jobs, arms = [], [{}, {}]
    for index, key in enumerate(sorted(left["_values"])):
        task, seed = key
        eid = str(index)
        jobs.append(dict(episode_id=eid, task_id=task, repeat=seed))
        for arm, cell in zip(arms, (left, right), strict=True):
            value = cell["_values"][key]
            arm[eid] = dict(observed=value is not None, native_score=value)
    result = paired_math()(jobs, *arms)
    for row in result["rows"]:
        row["seed"] = row.pop("repeat")
    return result


def next_decision(family: str, result: dict) -> str:
    if result["unknown_pairs"]:
        return (
            "Await the fixed audited cells; unknown outcomes do not justify an efficacy decision."
        )
    if family == "order":
        effect = "Repair gain" if result["difference"] > 0 else "No repair gain"
        return (
            effect
            + " in this cell: compare both fixed worlds, ordering variants and dose controls "
            "before replication; query-order causality is not isolated."
        )
    if family == "dose":
        effect = "Continuation gain" if result["difference"] > 0 else "No continuation gain"
        return (
            effect + " at this endpoint: inspect both predetermined 46/69 endpoints and worlds "
            "for each teacher; never select the best checkpoint."
        )
    effect = "Positive" if result["difference"] > 0 else "Nonpositive"
    return (
        effect + " within-Phi contrast: use both fixed worlds to decide replication; this tests "
        "teacher/interface packages, not repair-method transfer."
    )


def analyze(root: Path) -> dict:
    specs, contrasts = inventory(root)
    cells, comparisons = {}, []
    for name, spec in specs.items():
        try:
            cells[name] = load_cell(spec)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            cells[name] = dict(
                name=name, path=spec["path"], status="integrity_error", reason=str(exc)
            )
    for contrast in contrasts:
        row = dict(contrast)
        try:
            left, right = cells[row["left"]], cells[row["right"]]
            require("_values" in left and "_values" in right, "cell unavailable or invalid")
            row["result"] = compare_cells(left, right, row["same_actor"])
            row["next_decision"] = next_decision(row["family"], row["result"])
        except (ValueError, KeyError) as exc:
            row.update(
                result=None,
                reason=str(exc),
                next_decision="Resolve receipt/identity issue before interpreting this contrast.",
            )
        comparisons.append(row)
    return dict(
        schema="fixed-teaching-controls-readout-20260928-v1",
        cutoff_utc=datetime.now(timezone.utc).isoformat(),
        root=str(root),
        cells={
            name: {k: v for k, v in cell.items() if not k.startswith("_")}
            for name, cell in cells.items()
        },
        contrasts=comparisons,
        source_sha256={
            str(Path(__file__).resolve()): sha(Path(__file__)),
            str(PAIR_SOURCE): sha(PAIR_SOURCE),
        },
        scope="Fixed panel 00, worlds 42/50, fit seed 2026092208. Qwen has 16 slots/cell, Phi 8; "
        "counts come from actual PLANs or explicitly labeled frozen schedules for absent cells. "
        "Eight exposed roots per cell share recipes; task-bootstrap uncertainty is descriptive. "
        "Task-cluster intervals condition on each world/fit. No best checkpoint or pooled "
        "independence, "
        "novelty, recursion or confirmatory efficacy claim. Native auditor receipts are reused; "
        "this is not fresh replay or model/trace integrity verification.",
    )


def markdown(report: dict) -> str:
    lines = [
        "# Fixed teaching controls",
        "",
        "Cutoff: " + report["cutoff_utc"],
        "",
        "| Cell | Success / observed / planned | Unknown | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, cell in report["cells"].items():
        counts = "/".join(str(cell.get(k, "?")) for k in ("successes", "observed", "planned"))
        status = cell["status"] + (": " + cell["reason"] if "reason" in cell else "")
        lines.append(f"| {name} | {counts} | {cell.get('unknown', '?')} | {status} |")
    lines += [
        "",
        "| Right minus left | Difference (pp) | Win/loss/tie/unknown "
        "| Bounds (pp) | Task CI (pp) |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for contrast in report["contrasts"]:
        result = contrast["result"]
        if result is None:
            lines.append(f"| {contrast['name']} | unavailable | — | {contrast['reason']} | — |")
            continue
        point = "unknown" if result["difference"] is None else f"{100 * result['difference']:+.2f}"
        counts = "/".join(str(result[k]) for k in ("wins", "losses", "ties", "unknown_pairs"))
        bounds = ", ".join(f"{100 * v:+.2f}" for v in result["difference_bounds"])
        ci = result["task_cluster_95"]
        interval = "unknown" if ci is None else str([round(100 * v, 2) for v in ci])
        lines.append(f"| {contrast['name']} | {point} | {counts} | [{bounds}] | {interval} |")
    lines += ["", "## Conditional decisions", ""]
    lines += [
        "- " + text for text in dict.fromkeys(c["next_decision"] for c in report["contrasts"])
    ]
    lines += ["", report["scope"], ""]
    return "\n".join(lines)


def write(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with output.with_suffix(".md").open("x") as stream:
        stream.write(markdown(report))


def evidence_digest(root: Path, cache: dict) -> str:
    digests = {}
    for spec in inventory(root)[0].values():
        directory = Path(spec["path"])
        for path in [directory / AUDIT_NAMES[spec["family"]], *directory.glob("TERMINAL-*.json")]:
            if not path.exists():
                continue
            stat = path.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if cache.get(path, (None,))[0] != signature:
                cache[path] = (signature, sha(path))
            digests[str(path)] = cache[path][1]
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def watch(root: Path, output: Path, deadline: float) -> None:
    deadline = min(deadline, float(os.environ.get("SLURM_JOB_END_TIME", deadline)))
    require(deadline > time.time(), "future explicit observer deadline required")
    cache = {}
    while time.time() < deadline:
        digest = evidence_digest(root, cache)
        path = output / f"snapshot-{digest[:16]}.json"
        if not path.exists():
            report = analyze(root)
            if digest == evidence_digest(root, cache):
                report["observer"] = dict(evidence_sha256=digest, deadline_epoch=deadline)
                write(report, path)
                print(json.dumps(dict(snapshot=str(path), cutoff=report["cutoff_utc"])), flush=True)
        time.sleep(max(0, min(60, deadline - time.time())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true", help="output becomes a snapshot directory")
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    if args.watch:
        if args.deadline_epoch is None:
            parser.error("--watch requires --deadline-epoch")
        watch(args.root, args.output, args.deadline_epoch)
    else:
        report = analyze(args.root)
        write(report, args.output)
        print(
            json.dumps(
                dict(
                    output=str(args.output),
                    cells=len(report["cells"]),
                    states=dict(Counter(c["status"] for c in report["cells"].values())),
                )
            )
        )
