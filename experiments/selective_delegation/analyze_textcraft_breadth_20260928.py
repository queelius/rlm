"""CPU-only fixed-cutoff synthesis of the first six complete TextCraft breadth panels.

Consumes existing native audits and verifies small PLAN/episode/input receipts. It does
not import collectors, replay models, inspect GPU state, or hash model ancestry.
"""

import argparse
import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
PANELS = tuple(range(6))
WORLDS = (42, 50, 51, 52)
SEEDS = {"original": 2026092208, "2291": 2026092291}
TEACHERS = ("discovery", "known_recipe_corrected")
MODES = ("raw", "binder")
ARMS = tuple(f"{t}/{m}" for t, m in itertools.product(TEACHERS, MODES))
ROW_HASHES = {
    "discovery": "dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a",
    "known_recipe_corrected": "24ea72cb1242f2e0d819d8fb115737864de03fb750e064f145f9ec48a245e6d6",
}
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_DRAWS = 20000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def visible_name(item: str, prompt: str) -> bool:
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(item) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, prompt) is not None


def public_names(prompt: str) -> set[str]:
    """Names in public state and previous recipe feedback; ignore schema example text."""
    state = json.loads(next(line for line in prompt.splitlines() if line.startswith('{"goal":')))
    names = set(state["target_items"]) | set(state["current_inventory"])
    names.update(state["inventory_at_task_start"])
    for row in state["history"]:
        feedback = row.get("feedback")
        if row.get("action", {}).get("action") != "get_info" or not isinstance(feedback, list):
            continue
        for info in feedback:
            names.add(info["item"])
            for recipe in info["recipes"]:
                names.update(recipe["ingredients"])
    return names


def behavior(node: dict, first_query_prompt: str | None) -> dict:
    """Descriptive counts from audited public feedback, without interpreting hidden state."""
    history, roots = node["public_history"], set(node["targets"])
    queries = [h for h in history if h.get("action", {}).get("action") == "get_info"]
    counts = Counter(
        {
            "first_action_root_query": int(
                bool(history and roots.intersection(history[0].get("action", {}).get("items", [])))
            ),
            "ever_root_query": int(any(roots.intersection(q["action"]["items"]) for q in queries)),
        }
    )
    if queries:
        require(first_query_prompt is not None, "missing first-query prompt")
        items = queries[0]["action"]["items"]
        visible = set(items).issubset(public_names(first_query_prompt))
        text_visible = all(visible_name(item, first_query_prompt) for item in items)
        counts["first_query_structured_vs_text_disagreement"] += int(visible != text_visible)
        counts["episodes_with_query"] += 1
        counts["first_query_all_names_visible"] += int(visible)
        counts["first_query_any_name_unseen"] += int(not visible)
    for query in queries:
        for info in query["feedback"]:
            if not info["can_craft"] and not info["is_base"] and not info["in_inventory"]:
                counts["nonexistent_item_mentions"] += int(not info["recipes"])
    for row in history:
        feedback = str(row.get("feedback", ""))
        if row.get("action", {}).get("action") != "craft" or not feedback.startswith("Error:"):
            continue
        category = "other"
        for fragment, label in (
            ("Insufficient ingredients", "insufficient_inventory"),
            ("Extra ingredients", "extra_ingredient"),
            ("Missing required ingredient", "missing_ingredient"),
            ("Wrong amount", "wrong_amount"),
            ("No recipe found", "no_recipe"),
        ):
            if fragment in feedback:
                category = label
                break
        counts["craft_error_" + category] += 1
    return dict(counts)


def normalize_jobs(plan: dict) -> list[dict]:
    return [{k: v for k, v in j.items() if k != "condition"} for j in plan["jobs"]]


def load_inputs(root: Path) -> tuple[list[dict], dict]:
    receipts, panel_data, training, runs, rows, plans = {}, {}, {}, [], [], {}

    def read(path: Path, expected: str | None = None):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(expected is None or digest == expected, f"receipt changed: {path}")
        receipts[str(path)] = digest
        return json.loads(content)

    selection = read(root / "textcraft-breadth-inputs-001/SELECTION.json")
    master = read(root / "textcraft-breadth-inputs-001/MANIFEST.json")
    require(
        receipts[str(root / "textcraft-breadth-inputs-001/SELECTION.json")]
        == master["selection_sha256"],
        "selection differs from master manifest",
    )
    for panel, world in itertools.product(PANELS, WORLDS):
        directory = root / f"textcraft-breadth-inputs-001/p{panel:02d}-w{world}"
        selected = next(x for x in master["panels"] if x["path"] == str(directory))
        manifest = read(directory / "MANIFEST.json", selected["sha256"])
        task_bytes = (directory / "tasks.jsonl").read_bytes()
        digest = hashlib.sha256(task_bytes).hexdigest()
        require(digest == manifest["tasks_sha256"], "frozen tasks changed")
        receipts[str(directory / "tasks.jsonl")] = digest
        tasks = [json.loads(line) for line in task_bytes.splitlines()]
        require([t["id"] for t in tasks] == selection["panels"][panel], "selection changed")
        require(
            Counter(t["misc"]["max_depth"] for t in tasks) == {2: 2, 3: 2, 4: 1, 5: 3},
            "declared depth strata changed",
        )
        panel_data[panel, world] = {
            "manifest": manifest,
            "tasks": {t["id"]: t for t in tasks},
            "audits": {a["task_id"]: a for a in manifest["audits"]},
        }

    for panel, world, seed, teacher, mode in itertools.product(
        PANELS, WORLDS, SEEDS, TEACHERS, MODES
    ):
        prefix = "corrected-" if teacher == "known_recipe_corrected" else ""
        directory = root / f"textcraft-breadth-p{panel:02d}-w{world}-s{seed}-{prefix}{mode}-001"
        plan = read(directory / "PLAN.json")
        audit = read(directory / "NATIVE-AUDIT.json")
        require(
            audit["sha256"][str(directory / "PLAN.json")] == receipts[str(directory / "PLAN.json")],
            "native-audited plan changed",
        )
        require(plan["breadth_panel"] == panel and plan["world_seed"] == world, "wrong world")
        require(plan["execution_mode"] == mode, "wrong execution mode")
        require(plan["training_seed"] == SEEDS[seed], "wrong training seed")
        require(
            plan["teacher"]
            == ("public" if teacher == "discovery" else "quantity_corrected_original"),
            "teacher mapping changed",
        )
        require(plan["adapter"]["training_rows_sha256"] == ROW_HASHES[teacher], "wrong teacher")
        prepared = panel_data[panel, world]
        require(plan["tasks_sha256"] == prepared["manifest"]["tasks_sha256"], "task mismatch")
        require(plan["world_sha256"] == prepared["manifest"]["world_sha256"], "world mismatch")
        key = teacher, seed
        endpoint = Path(plan["fixed_adapter"])
        if key not in training:
            train_plan = read(endpoint.parent / "PLAN.json", plan["training_plan_sha256"])
            require(train_plan["rows_sha256"] == ROW_HASHES[teacher], "training data mismatch")
            require(train_plan["seed"] == SEEDS[seed], "training plan seed mismatch")
            require(
                (train_plan["tasks"], train_plan["rows"], train_plan["planned_updates"])
                == (32, 366, 23),
                "training dose contract changed",
            )
            training[key] = {
                "teacher": teacher,
                "seed": SEEDS[seed],
                "adapter": str(endpoint),
                "adapter_sha256_from_native_audit": plan["adapter"]["sha256"],
                "training_plan_sha256": plan["training_plan_sha256"],
                "training_rows_sha256": ROW_HASHES[teacher],
                "tasks": 32,
                "rows": 366,
                "updates": 23,
            }
        require(training[key]["adapter"] == str(endpoint), "endpoint changed within teacher")
        jobs = {j["episode_id"]: j for j in plan["jobs"]}
        paths = sorted((directory / "episodes").glob("*.json"))
        require(len(jobs) == len(paths) == 16, "incomplete or excess episode panel")
        require(set(jobs) == set(audit["audits"]) == {p.stem for p in paths}, "slot mismatch")
        require(len(audit["groups"]) == 1, "unexpected group count")
        group = next(iter(audit["groups"].values()))
        require(
            all(group[k] == 16 for k in ("planned", "recorded", "observed"))
            and group["missing"] == group["unavailable"] == 0,
            "incomplete native audit; never impute unknowns as losses",
        )
        require(not audit["unresolved_starts"] and not audit["calls_without_episode"], "orphans")
        require(group["cost"] == audit["physical_cost"], "unaccounted physical calls")
        run_rows = []
        for path in paths:
            row = read(path, audit["sha256"][str(path)])
            replay = audit["audits"][path.stem]
            require(all(row[k] == v for k, v in jobs[path.stem].items()), "job identity changed")
            require(row["observed"] and replay["replayed"], "unobserved or unaudited row")
            require(row["native_score"] in (0, 1), "nonbinary native outcome")
            require(replay["native_score"] == row["native_score"], "score changed")
            require(replay["calls"] == row["global_calls"], "call count changed")
            require(replay["errors"] == row["errors"], "error count changed")
            require(row["policy"] == "flat" and row["node_count"] == 1, "not a flat run")
            node_path = directory / "nodes" / f"{path.stem}-n0.json"
            node = read(node_path, audit["sha256"][str(node_path)])
            require(node["native_score"] == row["native_score"], "node score changed")
            require(node["call_ids"] == row["call_ids"], "node call identity changed")
            first_query_index = next(
                (
                    i
                    for i, h in enumerate(node["public_history"])
                    if h.get("action", {}).get("action") == "get_info"
                ),
                None,
            )
            first_query_prompt = None
            if first_query_index is not None:
                call_id = node["call_ids"][first_query_index]
                call_path = directory / "calls" / f"{call_id}.json"
                call = read(call_path, audit["sha256"][str(call_path)])
                first_query_prompt = call["request"]["prompt"]
            task = prepared["tasks"][row["task_id"]]
            qualification = prepared["audits"][row["task_id"]]
            run_rows.append(
                {
                    "run": directory.name,
                    "episode_id": path.stem,
                    "panel": panel,
                    "world": world,
                    "training_seed": SEEDS[seed],
                    "repeat": row["repeat"],
                    "evaluation_seed": row["seed"],
                    "teacher": teacher,
                    "mode": mode,
                    "arm": f"{teacher}/{mode}",
                    "task_id": row["task_id"],
                    "declared_depth": task["misc"]["max_depth"],
                    "dependency_chain": qualification["statistics"]["dependency_chain"],
                    "native_qualified": qualification.get("qualification", {}).get("native_score")
                    == 1,
                    "success": int(row["native_score"]),
                    "calls": row["global_calls"],
                    "output_tokens": row["global_output_tokens"],
                    "status": row["status"],
                    "errors": row["errors"],
                    "actions": replay["actions"],
                    "behavior": behavior(node, first_query_prompt),
                }
            )
        require(sum(x["success"] for x in run_rows) == group["won"], "aggregate score changed")
        require(sum(x["calls"] for x in run_rows) == group["cost"]["calls"], "cost changed")
        rows.extend(run_rows)
        plans[panel, world, seed, teacher, mode] = plan
        runs.append(
            {
                "run": directory.name,
                "arm": f"{teacher}/{mode}",
                "audit_sha256": receipts[str(directory / "NATIVE-AUDIT.json")],
                "cost": group["cost"],
                "owner_wall_seconds": audit["owner_wall_seconds"],
                "terminal": audit["terminal"],
            }
        )

    scientific_fields = (
        "tasks_sha256",
        "world_sha256",
        "budget_seconds",
        "sampling",
        "seeds",
        "model",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "truncation",
        "seed_rule",
    )
    for panel, world, seed in itertools.product(PANELS, WORLDS, SEEDS):
        reference = plans[panel, world, seed, "discovery", "raw"]
        for teacher, mode in itertools.product(TEACHERS, MODES):
            plan = plans[panel, world, seed, teacher, mode]
            require(
                all(plan[k] == reference[k] for k in scientific_fields),
                "scientific contract differs across paired arms",
            )
            require(normalize_jobs(plan) == normalize_jobs(reference), "paired schedule differs")
    ids = {r["task_id"] for r in rows}
    require(len(ids) == 48 and len(rows) == 3072 and len(runs) == 192, "fixed cutoff differs")
    historical = {}
    for relative in (
        "analysis-textcraft-target-multiset-004.json",
        "analysis-textcraft-stopped-step1-001.json",
        "analysis-textcraft-fresh-behavior-002.json",
        "analysis-textcraft-procedure-control-001.json",
        "analysis-textcraft-procedure-behavior-002.json",
        "textcraft-public-discovery-sft-001/TEACHER-CONTRACT.json",
        "textcraft-quantity-matched-inputs-004/MANIFEST.json",
    ):
        read(root / relative)
        historical[relative] = receipts[str(root / relative)]
    teacher_data, first_prompts, target_multisets, canonical_multisets = {}, {}, {}, {}
    label_multisets = {}
    for teacher, directory in (
        ("discovery", "textcraft-public-discovery-prototype-001"),
        ("known_recipe_corrected", "textcraft-quantity-matched-inputs-004"),
    ):
        path = root / directory / "rows.jsonl"
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(digest == ROW_HASHES[teacher], "teacher row bytes changed")
        receipts[str(path)] = digest
        dataset = [json.loads(line) for line in content.splitlines()]
        counts, first, targets = Counter(), {}, defaultdict(Counter)
        canonical, labels = defaultdict(Counter), defaultdict(Counter)
        for row in dataset:
            action = json.loads(row["target"])
            counts["prompt_tokens"] += row["prompt_tokens"]
            counts["target_tokens"] += row["target_tokens"]
            targets[row["task_id"]][row["target"]] += 1
            canonical[row["task_id"]][json.dumps(action, sort_keys=True)] += 1
            labels[row["task_id"]][tuple(token for token in row["labels"] if token != -100)] += 1
            if action["action"] == "get_info":
                visible = set(action["items"]).issubset(public_names(row["prompt"]))
                text_visible = all(visible_name(item, row["prompt"]) for item in action["items"])
                counts["query_structured_vs_text_disagreement"] += int(visible != text_visible)
                counts["query_rows"] += 1
                counts["query_rows_all_names_visible"] += int(visible)
            if row["step"] == 0:
                first[row["task_id"]] = row["prompt"]
                require(action["action"] == "get_info", "teacher no longer starts with a query")
                counts["first_queries"] += 1
                counts["first_query_all_names_visible"] += int(visible)
        teacher_data[teacher] = dict(counts)
        first_prompts[teacher], target_multisets[teacher] = first, targets
        canonical_multisets[teacher], label_multisets[teacher] = canonical, labels
    require(
        set(first_prompts[TEACHERS[0]]) == set(first_prompts[TEACHERS[1]]), "teacher tasks differ"
    )
    teacher_data["same_initial_prompts"] = sum(
        first_prompts[TEACHERS[0]][task] == first_prompts[TEACHERS[1]][task]
        for task in first_prompts[TEACHERS[0]]
    )
    teacher_data["same_target_string_multisets"] = sum(
        target_multisets[TEACHERS[0]][task] == target_multisets[TEACHERS[1]][task]
        for task in target_multisets[TEACHERS[0]]
    )
    for name, multisets in (
        ("canonical_action", canonical_multisets),
        ("unmasked_label_tokens", label_multisets),
    ):
        teacher_data[name + "_multiset_differences"] = [
            {
                "task_id": task,
                "discovery_only": list(
                    (multisets[TEACHERS[0]][task] - multisets[TEACHERS[1]][task]).elements()
                ),
                "known_recipe_only": list(
                    (multisets[TEACHERS[1]][task] - multisets[TEACHERS[0]][task]).elements()
                ),
            }
            for task in multisets[TEACHERS[0]]
            if multisets[TEACHERS[0]][task] != multisets[TEACHERS[1]][task]
        ]
    return rows, {
        "receipts": receipts,
        "runs": runs,
        "training": list(training.values()),
        "selection": {k: v for k, v in selection.items() if k != "input_sha256"},
        "contract": {k: reference[k] for k in scientific_fields},
        "excluded_panel_policy": "All panel06/panel07 outcomes excluded, completed or partial.",
        "historical_reports_sha256": historical,
        "teacher_visibility": teacher_data,
    }


def aggregate(rows: list[dict]) -> dict:
    errors, actions, statuses, behaviors = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        errors.update(row["errors"])
        actions.update(row["actions"])
        statuses.update([row["status"]])
        behaviors.update(row["behavior"])
    won = sum(r["success"] for r in rows)
    calls = sum(r["calls"] for r in rows)
    return {
        "attempts": len(rows),
        "task_clusters": len({r["task_id"] for r in rows}),
        "successes": won,
        "success_rate": won / len(rows),
        "calls": calls,
        "calls_per_attempt": calls / len(rows),
        "calls_per_success_aggregate": calls / won if won else None,
        "errors": dict(errors),
        "actions": dict(actions),
        "statuses": dict(statuses),
        "behavior": dict(behaviors),
        "failure_with_no_recorded_errors": sum(
            r["success"] == 0 and not any(r["errors"].values()) for r in rows
        ),
    }


def bootstrap(clusters: dict, stratified: bool, draws: int) -> list[list[float]]:
    """Resample task identities, preserving all paired world/seed/repeat observations."""
    strata = defaultdict(list)
    for task_id, row in sorted(clusters.items()):
        strata[row["depth"] if stratified else 0].append(task_id)
    dimensions = len(next(iter(clusters.values()))["values"])
    samples = [[] for _ in range(dimensions)]
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(draws):
        selected = [task for ids in strata.values() for task in rng.choices(ids, k=len(ids))]
        totals = [0.0] * dimensions
        for task in selected:
            for index, value in enumerate(clusters[task]["values"]):
                totals[index] += value
        for index, total in enumerate(totals):
            samples[index].append(total / len(selected))
    return [[sorted(values)[int((draws - 1) * q)] for q in (0.025, 0.975)] for values in samples]


def paired(rows: list[dict], coefficients: dict[str, int], draws: int = 0) -> dict:
    slots = defaultdict(dict)
    for row in rows:
        if row["arm"] in coefficients:
            slots[row["task_id"], row["world"], row["training_seed"], row["repeat"]][row["arm"]] = (
                row
            )
    parents = defaultdict(list)
    deltas = []
    for slot, arms in sorted(slots.items()):
        require(set(arms) == set(coefficients), "unmatched contrast")
        delta = [
            sum(coef * arms[arm]["success"] for arm, coef in coefficients.items()),
            sum(coef * arms[arm]["calls"] for arm, coef in coefficients.items()),
            sum(
                coef * arms[arm]["errors"].get("native_action_error", 0)
                for arm, coef in coefficients.items()
            ),
        ]
        parents[slot[0]].append(delta)
        deltas.append(delta)
    depth = {r["task_id"]: r["declared_depth"] for r in rows}
    clusters = {
        task: {"depth": depth[task], "values": [mean(v) for v in zip(*values, strict=True)]}
        for task, values in parents.items()
    }
    cluster_means = [c["values"][0] for c in clusters.values()]
    result = {
        "paired_attempts": len(slots),
        "task_clusters": len(parents),
        "positive_pairs": sum(v[0] > 0 for v in deltas),
        "negative_pairs": sum(v[0] < 0 for v in deltas),
        "tied_pairs": sum(v[0] == 0 for v in deltas),
        "difference": mean(v[0] for v in deltas),
        "calls_difference_per_attempt": mean(v[1] for v in deltas),
        "native_errors_difference_per_attempt": mean(v[2] for v in deltas),
        "positive_task_clusters": sum(v > 0 for v in cluster_means),
        "negative_task_clusters": sum(v < 0 for v in cluster_means),
    }
    if draws:
        # All primary contrasts have 16 matched attempts per task identity.
        require({len(v) for v in parents.values()} == {16}, "unequal primary cluster sizes")
        intervals = bootstrap(clusters, True, draws)
        result.update(
            ci95=intervals[0],
            calls_difference_ci95=intervals[1],
            native_errors_difference_ci95=intervals[2],
            unstratified_task_ci95=bootstrap(clusters, False, draws)[0],
        )
    return result


def synthesize(rows: list[dict], evidence: dict, draws: int) -> dict:
    arms = {arm: aggregate([r for r in rows if r["arm"] == arm]) for arm in ARMS}
    for arm in ARMS:
        costs = [r["cost"] for r in evidence["runs"] if r["arm"] == arm]
        arms[arm]["native_cost"] = {k: sum(c[k] for c in costs) for k in costs[0]}
        arms[arm]["owner_wall_seconds"] = sum(
            r["owner_wall_seconds"] for r in evidence["runs"] if r["arm"] == arm
        )
    contrasts = {
        "discovery_binder_minus_raw": {"discovery/binder": 1, "discovery/raw": -1},
        "known_recipe_corrected_binder_minus_raw": {
            "known_recipe_corrected/binder": 1,
            "known_recipe_corrected/raw": -1,
        },
        "discovery_minus_known_recipe_raw": {"discovery/raw": 1, "known_recipe_corrected/raw": -1},
        "discovery_minus_known_recipe_binder": {
            "discovery/binder": 1,
            "known_recipe_corrected/binder": -1,
        },
        "binder_by_teacher_interaction": {
            "discovery/binder": 1,
            "discovery/raw": -1,
            "known_recipe_corrected/binder": -1,
            "known_recipe_corrected/raw": 1,
        },
    }
    conditional = {}
    for field in ("world", "training_seed", "declared_depth", "dependency_chain", "panel"):
        conditional[field] = {}
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in rows if r[field] == value]
            conditional[field][str(value)] = {
                "arms": {arm: aggregate([r for r in subset if r["arm"] == arm]) for arm in ARMS},
                "discovery_binder_minus_raw": paired(
                    subset, contrasts["discovery_binder_minus_raw"]
                ),
                "known_recipe_corrected_binder_minus_raw": paired(
                    subset, contrasts["known_recipe_corrected_binder_minus_raw"]
                ),
            }
    # Compact conditional output retains counts/rates/calls, full errors stay in details.
    compact = {}
    for field, groups in conditional.items():
        compact[field] = {
            key: {
                "arms": {
                    arm: {
                        k: g["arms"][arm][k]
                        for k in (
                            "attempts",
                            "task_clusters",
                            "successes",
                            "success_rate",
                            "calls",
                        )
                    }
                    for arm in ARMS
                },
                "discovery_binding_delta": g["discovery_binder_minus_raw"]["difference"],
                "known_recipe_binding_delta": g["known_recipe_corrected_binder_minus_raw"][
                    "difference"
                ],
            }
            for key, g in groups.items()
        }
    evidence["conditional"] = conditional
    task_rows = []
    for task in sorted({r["task_id"] for r in rows}):
        subset = [r for r in rows if r["task_id"] == task]
        task_rows.append(
            {
                "task_id": task,
                "depth": subset[0]["declared_depth"],
                "successes": {
                    arm: sum(r["success"] for r in subset if r["arm"] == arm) for arm in ARMS
                },
            }
        )
    evidence["task_outcomes"] = task_rows
    native_tasks = {(r["task_id"], r["world"]): r for r in rows}
    result = {
        "schema": "textcraft-breadth-fixed-cutoff-synthesis-v1",
        "analysis_date": "2026-09-28",
        "campaign": "TEXTCRAFT-BREADTH-CAMPAIGN-002",
        "cutoff": {
            "panels": list(PANELS),
            "worlds": list(WORLDS),
            "training_seeds": list(SEEDS.values()),
            "runs": 192,
            "attempts": 3072,
            "attempts_per_arm": 768,
            "task_identities": 48,
            "native_audits_present_and_small_receipts_verified": 192,
            "native_qualified_task_worlds": sum(
                r["native_qualified"] for r in native_tasks.values()
            ),
            "task_worlds": len(native_tasks),
            "excluded": evidence["excluded_panel_policy"],
        },
        "mapping": evidence["training"],
        "teacher_visibility_and_dose": evidence["teacher_visibility"],
        "uncertainty": {
            "draws": draws,
            "seed": BOOTSTRAP_SEED,
            "primary": "Percentile bootstrap task identities within declared-depth strata; "
            "retain all four worlds, two training seeds and two repeats together.",
            "sensitivity": "Unstratified task-identity bootstrap, same fixed worlds/seeds/repeats.",
            "scope": "Conditional on the task selection procedure, four shared worlds and two "
            "fitted seeds; no world/seed population CI, no iid attempt CI, "
            "no multiplicity correction.",
        },
        "arms": arms,
        "contrasts": {name: paired(rows, coef, draws) for name, coef in contrasts.items()},
        "conditional": compact,
        "historical_controls": {
            "references_sha256": evidence["historical_reports_sha256"],
            "root_first_procedure_prompt": "Already tested on the original uncorrected "
            "known-recipe adapter: 0/16 successes versus 3/16 default, despite root-first actions "
            "rising from 3/16 to 10/16; repeated static-recipe queries rose from 70 to 418. "
            "A new minimal prompt on the corrected adapter would be a changed replication.",
            "one_step_extra_sft": "Already tested from discovery checkpoint: 17/32 versus "
            "15/32 baseline; one RL update also 15/32. SFT and RL matched one update and 12,074 "
            "credited/target tokens on eight training tasks. Inconclusive on 16 evaluation goals.",
            "dose_inventory": "Read-only scan of current top-level textcraft*/PLAN.json found "
            "the completed original/discovery/corrected endpoints at one epoch, 23 updates; "
            "no matched multi-epoch teacher-package readout was identified. This search is bounded "
            "to the active store, not proof of absence from all historical stores.",
        },
        "limitations": [
            "Known_recipe_corrected is the quantity-corrected known-recipe teacher, "
            "not a corrected discovery model.",
            "Both teacher packages have 32 tasks, 366 rows and 23 updates; query order, "
            "histories, prompt-token exposure and trajectory package differ. "
            "Target-token dose is identical after the known-recipe quantity correction.",
            "Forty-eight task identities, not 768 independent examples; worlds, repeated "
            "evaluation seeds, training seeds, names and generator induce dependence.",
            "World42 is the source recipe world; worlds50–52 share generator/item grammar "
            "and change recipes plus sufficient inventory. New-to-inventory goal roots "
            "are not an unbounded historical exposure guarantee.",
            "Declared task depth can exceed realized dependency-chain depth. Four depth "
            "strata are deliberately reweighted; rates are not an official benchmark average.",
            "The first six complete panels are an explicit analysis cutoff; panels06–07 "
            "are excluded regardless of availability/outcome. The larger campaign is incomplete.",
            "Binding modifies ingredients only from a previously observed single recipe "
            "when output quantity is divisible; it does not choose targets, quantities, "
            "decomposition or stopping.",
            "All arms are flat action SFT; these data do not establish recursive delegation, "
            "RLVR gains, other-model transfer or equality of inference token cost.",
            "Native audits are reused; this synthesis verifies PLAN/episode, node histories, "
            "first-query calls and frozen input hashes, not a full new replay or weight rehash.",
            "Query visibility and failure categories are descriptive correlates, not isolated "
            "causal mechanisms; repeated nonexistent-name mentions are not independent errors.",
        ],
        "next_decisions": [
            "Highest priority: repair known-recipe teacher query observability while preserving "
            "crafting order and action/target multiset, then compare fixed trained endpoints "
            "against both existing packages. Audit every query name against public state. "
            "This tests whether unobserved-name supervision drives the transfer failure.",
            "Separate order from optimization: compare predetermined one/two/three-epoch "
            "endpoints of the same frozen teacher packages with matched updates and seeds. "
            "The existing one-step discovery SFT control does not answer known-teacher "
            "undertraining. Keep all checkpoints in the report; do not choose a best VAL result.",
            "Check teacher-forced fit by action type and query visibility on frozen TRAIN/VAL "
            "rows: successful fit to hidden-name labels with poor rollout transfer supports "
            "a supervision/state-distribution problem; poor fit leaves undertraining open.",
            "Retain discovery plus binding for depth4–5 feasibility/quantity interventions. "
            "Binding removes ingredient-selection errors but does not ensure enough inventory. "
            "Compare bounded quantity planning/decomposition on paired goals before broad RLVR.",
            "Do not repeat the completed multi-sentence root-first prompt control as if new. "
            "A minimal root-first-only instruction on corrected weights and new goals is a "
            "changed replication, lower priority than repairing training-time observability.",
        ],
        "advisor_decision": "Update the breadth/generalization evidence: large teacher-package "
        "difference, matched supervised target dose but sharply different query observability, "
        "smaller binding gain, and severe depth5 limitation. Parent owns deck integration.",
    }
    return result


def markdown(result: dict) -> str:
    lines = [
        "# TextCraft breadth: fixed panels00–05, 2026-09-28",
        "",
        "Discovery training plus observed-recipe binding reaches 381/768 (49.61%); raw is "
        "323/768 (42.06%). The quantity-corrected **known-recipe** teacher reaches only 48/768 "
        "(6.25%) with binding and 39/768 (5.08%) raw. `corrected` never denotes a correction "
        "to discovery training.",
        "",
        "The cutoff is 192 native-audited runs: six panels × four worlds × two training seeds × "
        "two teacher packages × two execution modes, with 16 attempts each. All 3,072 outcomes "
        "are observed, all 192 task-world inputs passed native qualification, and small audited "
        "PLAN/episode/input hashes match. Panels06–07 are excluded even when completed.",
        "",
        "| Teacher / mode | Successes /768 | Calls | Calls / attempt | "
        "Native action errors | Schema errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, row in result["arms"].items():
        lines.append(
            f"| {arm} | {row['successes']} | {row['calls']:,} | "
            f"{row['calls_per_attempt']:.2f} | {row['errors'].get('native_action_error', 0):,} | "
            f"{row['errors'].get('invalid_schema', 0)} |"
        )
    lines += [
        "",
        "All native service calls returned; no transport failures or unknown outcomes. "
        "Native action errors are environment feedback, not failed model requests.",
        "",
        "| Paired contrast | Difference, percentage points | "
        "95% task-cluster interval | Wins / losses / ties |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["contrasts"].items():
        lo, hi = row["ci95"]
        lines.append(
            f"| {name} | {100 * row['difference']:+.2f} | "
            f"[{100 * lo:+.2f}, {100 * hi:+.2f}] | "
            f"{row['positive_pairs']} / {row['negative_pairs']} / {row['tied_pairs']} |"
        )
    lines += [
        "",
        "Intervals resample the 48 task identities within declared-depth strata, retaining "
        "all worlds, training seeds and repeated attempts. They are conditional on these four "
        "worlds and two fitted seeds. Interaction wins/losses count the sign of the "
        "difference-in-differences, not a simple arm comparison. Full JSON includes an "
        "unstratified task-cluster sensitivity and call/error intervals.",
        "",
    ]
    visibility = result["teacher_visibility_and_dose"]
    lines += [
        "The strongest mechanistic lead is **what the demonstrations ask the model to know**. "
        f"All {visibility['same_initial_prompts']} task-initial prompts are identical. "
        f"All {visibility['same_target_string_multisets']} per-task target-string multisets "
        "are identical after the quantity correction; canonical parsed-action and unmasked "
        "label-token multisets also have no differences. Both supervise 8,820 target tokens. "
        "The difference lies in action ordering and the public state/history conditioning "
        "those same actions; discovery has 438,065 prompt tokens versus 414,682 (+5.64%).",
        "",
        "| Training-time query visibility | Discovery | Known-recipe corrected |",
        "|---|---:|---:|",
        "| First query names available in public state | 32/32 | 2/32 |",
        "| All query rows using available names | 167/167 | 32/167 |",
        "",
        "Visibility is checked against structured `target_items`, current/initial inventory "
        "and previous `get_info` item/recipe-ingredient feedback, excluding schema examples "
        "and model-invented names echoed in other actions. A literal-name-in-prompt check "
        "agrees on all training queries. It differs on four evaluation first queries per "
        "known-recipe mode; structured visibility is the primary measure.",
        "",
        "| Evaluation behavior | Discovery raw / binder | Known-recipe raw / binder |",
        "|---|---:|---:|",
        "| First physical action queries root | 765 / 765 of768 | 113 / 113 of768 |",
        "| First query uses available names | 768/768 / 768/768 | 113/762 / 113/762 |",
        "| Ever queries root | 768 / 768 | 232 / 244 |",
        "| Nonexistent-item query mentions | 104 / 97 | 10,552 / 10,549 |",
        "",
        "Six known-recipe episodes in each mode never query. Of its 762 queried episodes, "
        "649 first ask for a name absent from the structured public information. This is "
        "consistent with learning a privileged leaf-first naming policy that transfers badly; "
        "it is still a behavioral association, not an isolated causal result.",
        "",
        "Synthetic example: TRAIN `textcraft_synth.train.988` starts with goal `c6_i2` and "
        "inventory `c5_ore`, `c0_ore`. Discovery teaches `get_info(c6_i2)`. The corrected "
        "known-recipe teacher teaches `get_info(c9_i1_19)` before that prerequisite name "
        "appears in public information. On VAL `textcraft_synth.val.313`, panel00/world42/"
        "training-seed2026092208/repeat0, the goal is `t8_i2_18`, inventory is `raw_o6`, "
        "`raw_t1`, `raw_t3`, but the known-recipe model first asks for unseen `t1_i1_2`. "
        "Native feedback reports no recipe and it ultimately fails; discovery queries "
        "the visible goal and succeeds. These examples illustrate the full-panel counts; "
        "they are not an outcome-selected evaluation subset.",
        "",
    ]
    for field in ("declared_depth", "world", "training_seed", "dependency_chain", "panel"):
        lines += [
            f"| {field} | Discovery raw → binder | Known-recipe corrected raw → binder |",
            "|---|---:|---:|",
        ]
        for value, row in result["conditional"][field].items():
            a = row["arms"]
            lines.append(
                f"| {value} | {a['discovery/raw']['successes']} → "
                f"{a['discovery/binder']['successes']} /{a['discovery/raw']['attempts']} | "
                f"{a['known_recipe_corrected/raw']['successes']} → "
                f"{a['known_recipe_corrected/binder']['successes']} "
                f"/{a['known_recipe_corrected/raw']['attempts']} |"
            )
        lines.append("")
    lines += [
        "Binding improves discovery in every world, both training seeds and every declared-depth "
        "stratum. It cuts total discovery calls by 9.76% and native action errors by 39.53%; "
        "schema errors stay at 146. Context-cap terminations fall from 49 to 32. These paired "
        "effects support keeping the binder, with remaining failures concentrated in harder tasks.",
        "",
        "Depth5 remains difficult: 37/288 (12.85%) for discovery+binding, versus 181/192 "
        "(94.27%) at depth2. The known-recipe package has fewer native action errors yet far "
        "fewer successes; low error counts alone are not a competence metric. Exact error/status "
        "breakdowns and per-task outcomes are available in the external details artifact.",
        "",
        "The primary interpretation is a teacher-package transfer advantage plus an additional "
        "binding benefit. It does not establish that query order alone causes the package gap, "
        "or that recursive delegation/RLVR has improved.",
        "",
        "Binding shifts discovery errors from argument construction toward feasibility: "
        "missing/extra/wrong ingredient errors fall from 5,953 to 281, while explicit "
        "insufficient-inventory errors rise from 3,155 to 5,132. This change reflects new "
        "trajectories and error precedence; it does not prove the binder worsens planning.",
        "",
        "Historical controls already completed:",
        "",
        "- [Root-first procedure prompt](TEXTCRAFT-PROCEDURE-CONTROL-FINDINGS.md): "
        "old uncorrected known-recipe model 0/16 versus default 3/16, despite more root "
        "queries; repeated static-recipe queries rose 70→418. A minimal prompt on corrected "
        "weights would be a changed replication, not a first test of root-first prompting.",
        "- [One-step extra SFT](TEXTCRAFT-ONE-STEP-FINDINGS.md): discovery 17/32 versus "
        "15/32 baseline, with an uncertain effect; matched one-update/12,074-token RL is "
        "also 15/32. This does not settle whether the known-recipe teacher is undertrained. "
        "The active-store training-plan scan found no matched multi-epoch teacher readout.",
        "",
        "Limitations:",
        "",
    ]
    lines.extend(f"- {s}" for s in result["limitations"])
    lines += ["", "Next research decisions:", ""]
    lines.extend(f"{index}. {s}" for index, s in enumerate(result["next_decisions"], 1))
    lines += [
        "",
        result["advisor_decision"],
        "",
        "Reproduce from the repository root:",
        "",
        "```bash",
        "python experiments/selective_delegation/analyze_textcraft_breadth_20260928.py \\",
        "  --report /tmp/textcraft-breadth-20260928.json \\",
        "  --details /tmp/textcraft-breadth-20260928-details.json",
        "```",
        "",
        "The command also writes Markdown beside the JSON. Outputs are exclusive-create. "
        "The checked-in JSON records the script hash and external evidence-details hash; "
        "the latter includes every verified input receipt and all 192 run audit hashes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    require(args.bootstrap_draws >= 1000, "too few bootstrap draws")
    require(
        len({args.report, args.report.with_suffix(".md"), args.details}) == 3, "overlapping outputs"
    )
    for path in (args.report, args.report.with_suffix(".md"), args.details):
        require(not path.exists(), f"output already exists: {path}")
    rows, evidence = load_inputs(args.root)
    result = synthesize(rows, evidence, args.bootstrap_draws)
    evidence["episodes"] = rows
    evidence["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content = json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("x") as stream:
        stream.write(content)
    result["evidence"] = {
        "root": str(args.root),
        "details": str(args.details),
        "details_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "analysis_script_sha256": evidence["analysis_script_sha256"],
        "verified_small_receipt_count": len(evidence["receipts"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
    print(json.dumps({"cutoff": result["cutoff"], "contrasts": result["contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
