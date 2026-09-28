"""Bounded diagnostic-only synthetic counterfactual witness study; CPU only."""

import argparse
import ast
import copy
import hashlib
import json
import math
import sys
import time
import typing
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent
sys.path.insert(0, str(LIBRARY))

import audit_textcraft_observational_ambiguity as previous  # noqa: E402
import prepare_textcraft_public as public  # noqa: E402

bridge, encoder = public.bridge, public.reference
ROOT = previous.ROOT
DATA = ROOT / "textcraft-fresh-train-20260928-001/diagnostic"
OUTPUT = ROOT / "analysis-teacher-counterfactual-20260928-001"
TASKS_SHA = "12b0381fa3bcc17aee92e8677a713f8db3ba613cad7faa8229cc63411ed6d0ff"
ROOT_MANIFEST_SHA = "76e387378c9a89b0d8e26180874db61ad40086721c6cfef9e3b78dd77f915e05"
sha, save, digest = encoder.sha, encoder.save, previous.digest


def hybrid_world(first, second, targets):
    """Host-only construction. All replaced product names retain their native tier."""
    if len(targets) != 1:
        raise ValueError("predeclared single-root construction required")
    depth = first.item_depths[next(iter(targets))]
    hybrid, changed = copy.deepcopy(first), []
    for item, recipes in first.recipes.items():
        tier = first.item_depths[item]
        if item in targets or not 0 < tier < depth:
            continue
        alternatives = second.recipes.get(item)
        if (
            not alternatives
            or second.item_depths.get(item) != tier
            or any(r.depth != tier for r in [*recipes, *alternatives])
        ):
            raise ValueError("same-tier completion missing: " + item)
        hybrid.recipes[item] = copy.deepcopy(alternatives)
        if [(r.ingredients, r.result_count) for r in recipes] != [
            (r.ingredients, r.result_count) for r in alternatives
        ]:
            changed.append(item)
    return hybrid, sorted(changed)


def exact_collision(prompts, token_ids):
    if len(prompts) != 2 or prompts[0].encode() != prompts[1].encode():
        raise ValueError("complete public prompt bytes differ")
    if token_ids[0] != token_ids[1]:
        raise ValueError("complete encoded input token IDs differ")


def label_statistics(records):
    groups = defaultdict(Counter)
    for prompt_identity, action_identity in records:
        groups[prompt_identity][action_identity] += 1
    total = len(records)
    entropy = (
        sum(
            -count / total * math.log2(count / sum(group.values()))
            for group in groups.values()
            for count in group.values()
        )
        if total
        else None
    )
    return dict(
        observations=total,
        exact_groups=len(groups),
        conditional_entropy_bits=entropy,
        maximum_empirical_exact_label_accuracy=(
            sum(max(group.values()) for group in groups.values()) / total if total else None
        ),
        conflicting_groups=sum(len(group) > 1 for group in groups.values()),
        singleton_groups=sum(sum(group.values()) == 1 for group in groups.values()),
        interpretation="Balanced two-completion empirical labels, not task-success bounds. "
        "Singleton groups are not evidence of inferability.",
    )


def trusted_solvers():
    reference = ROOT / "analysis-textcraft-observational-ambiguity-001.json"
    if sha(reference) != previous.REFERENCE_SHA:
        raise ValueError("reviewed construction reference changed")
    record = json.loads(reference.read_text())
    source = bridge.PACKAGE / "synth_tasks.py"
    expected = {Path(p).name: h for p, h in record["source_sha256"].items()}
    if sha(source) != expected[source.name]:
        raise ValueError("reviewed official solver source changed")
    names = {"extract_base_materials_synth", "solve_crafting_task_synth"}
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    hashes = {
        n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
        for n in nodes
    }
    if hashes != record["extracted_trusted_AST_sha256"]:
        raise ValueError("reviewed official solver AST changed")
    namespace = {**vars(typing), "SynthRecipeDatabase": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["extract_base_materials_synth"], namespace["solve_crafting_task_synth"], hashes


def update_observed(observed, action, reply):
    if action["action"] == "get_info" and isinstance(reply, list):
        for info in reply:
            observed[info["item"]] = dict(
                is_base=info["is_base"],
                recipes=[
                    dict(ingredients=dict(r["ingredients"]), result_count=r["result_count"])
                    for r in info["recipes"]
                ],
            )


def input_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def step(frame, history, observed, task, tokenizer, action):
    cap = frame.budget.reserve()
    prompt = bridge.public_prompt(frame, history, goal=task["goal"])
    row = encoder.encode_row(prompt, action, tokenizer)
    if row["prompt_tokens"] + cap > 8192:
        raise bridge.BudgetExceeded("context_cap")
    if row["target_tokens"] > cap:
        raise bridge.BudgetExceeded("target_output_cap")
    bridge.parse_action(row["target"])
    before_known = sorted(observed)
    frame.budget.charge(row["target_tokens"])
    reply = frame.apply(action)
    history.append(dict(action=copy.deepcopy(action), feedback=copy.deepcopy(reply)))
    update_observed(observed, action, reply)
    return dict(**row, feedback=reply, public_known_before=before_known)


def prefix(task, world, inventory, tokenizer):
    frame = bridge.Frame(world, dict(inventory), task["misc"]["target_items"], bridge.Budget(), 0)
    history, observed = [], {}
    action = {"action": "get_info", "items": list(frame.targets)}
    row = step(frame, history, observed, task, tokenizer, action)
    prompt = bridge.public_prompt(frame, history, goal=task["goal"])
    return frame, history, observed, row, prompt, input_ids(tokenizer, prompt)


def oracle_schedule(solution, observed):
    """Original official depth-stable solver order; suppress already-observed root query."""
    actions, queried = [], set(observed)
    for craft in solution:
        target = craft["target"][0]
        if target not in queried:
            actions.append(dict(action="get_info", items=[target]))
            queried.add(target)
        actions.append(
            dict(
                action="craft",
                ingredients=craft["ingredients"],
                target_item=target,
                output_count=craft["result_count"],
            )
        )
    actions.append(dict(action="finish", message="done"))
    return actions


def continue_public(task, start, forced, tokenizer):
    original, history, observed, root_row, expected_prompt, expected_ids = start
    frame = bridge.Frame(
        original.world,
        dict(original.initial_inventory),
        original.targets,
        copy.copy(original.budget),
        0,
    )
    frame.inventory.update(original.inventory)
    history, observed = copy.deepcopy(history), copy.deepcopy(observed)
    rows, status, error = [copy.deepcopy(root_row)], "unfinished", None
    forced_applied = False
    try:
        # Same public state as the collision, with prefix costs retained in every arm.
        actual_prompt = bridge.public_prompt(frame, history, goal=task["goal"])
        exact_collision(
            [expected_prompt, actual_prompt], [expected_ids, input_ids(tokenizer, actual_prompt)]
        )
        while not frame.finished:
            action = (
                forced
                if not forced_applied
                else public.next_action(
                    dict(frame.targets),
                    dict(frame.initial_inventory),
                    dict(frame.inventory),
                    copy.deepcopy(observed),
                )
            )
            row = step(frame, history, observed, task, tokenizer, action)
            rows.append(row)
            forced_applied = True
            if isinstance(row["feedback"], str) and row["feedback"].startswith("Error:"):
                status, error = "native_action_error", row["feedback"]
                break
        if frame.finished:
            status = "finished"
    except bridge.BudgetExceeded as exc:
        status, error = "budget_or_context_cap", str(exc)
    except Exception as exc:
        status, error = "public_continuation_failure", f"{type(exc).__name__}: {exc}"
    score, details = frame.score()
    assert frame.budget.calls == len(rows)
    assert frame.budget.output_tokens == sum(row["target_tokens"] for row in rows)
    action_counts = Counter(json.loads(row["target"])["action"] for row in rows)
    result = dict(
        status=status,
        error=error,
        forced_action_applied=forced_applied,
        forced_action_native_valid=(
            forced_applied
            and not (
                isinstance(rows[1]["feedback"], str) and rows[1]["feedback"].startswith("Error:")
            )
        ),
        native_score=score,
        native_details=details,
        finished=frame.finished,
        success=frame.finished and score == 1,
        calls=frame.budget.calls,
        output_tokens=frame.budget.output_tokens,
        action_counts=dict(action_counts),
        max_prompt_plus_cap=max(row["prompt_tokens"] + 256 for row in rows),
        final_inventory=dict(frame.inventory),
        public_recipe_count=len(observed),
        actual_new_native_calls=len(rows) - 1,
    )
    return result, rows


def run(output):
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError("immutable study output already exists")
    started = time.monotonic()
    root_manifest = DATA.parent / "MANIFEST.json"
    if sha(root_manifest) != ROOT_MANIFEST_SHA or sha(DATA / "tasks.jsonl") != TASKS_SHA:
        raise ValueError("frozen diagnostic group identity changed")
    meta = json.loads((DATA / "MANIFEST.json").read_text())
    root_meta = json.loads(root_manifest.read_text())
    if sha(DATA / "MANIFEST.json") != root_meta["groups"]["diagnostic"]["manifest_sha256"]:
        raise ValueError("frozen group manifest changed")
    tasks = [json.loads(line) for line in (DATA / "tasks.jsonl").read_text().splitlines()]
    if len(tasks) != 8 or [task["id"] for task in tasks] != meta["task_ids"]:
        raise ValueError("exact frozen diagnostic-B identities required")
    if any(not task["id"].startswith("textcraft_synth.train.") for task in tasks):
        raise ValueError("only official TRAIN identities allowed")
    first = bridge.load_world()
    module = sys.modules["pinned_textcraft_synth_generator_d9c5857d"]
    second = module.SynthRecipeDatabase()
    second.generate_all_recipes(seed=43, items_per_domain_tier=25)
    extract, solve, ast_hashes = trusted_solvers()
    tokenizer = AutoTokenizer.from_pretrained(
        encoder.BASE,
        local_files_only=True,
        trust_remote_code=False,
    )
    paths = [
        Path(__file__),
        HERE / "test_study.py",
        Path(previous.__file__),
        Path(public.__file__),
        Path(encoder.__file__),
        Path(bridge.__file__),
        bridge.PACKAGE / "synth_tasks.py",
        bridge.GENERATOR,
        bridge.ENV,
        LIBRARY / "TEACHER-AMBIGUITY-PROBE-20260928.md",
        DATA / "tasks.jsonl",
        DATA / "MANIFEST.json",
        root_manifest,
        encoder.BASE / "tokenizer.json",
        encoder.BASE / "tokenizer_config.json",
    ]
    save(
        output / "PROVENANCE.json",
        dict(
            schema="teacher-counterfactual-witness-20260928-v1",
            utc=datetime.now(timezone.utc).isoformat(),
            sources_sha256={str(p): sha(p) for p in paths},
            diagnostic_task_ids=meta["task_ids"],
            upstream=bridge.trusted_provenance(),
            reviewed_solver_AST_sha256=ast_hashes,
            tokenizer=str(encoder.BASE),
            tokenizer_local_only=True,
            model_calls=0,
            GPU_used=False,
            training=False,
            source_world_seeds=[42, 43],
            items_per_domain_tier=25,
            construction="World42 root and metadata fixed; only lower-tier, same-name/same-tier "
            "unqueried recipes replaced from43. Componentwise-max consumed base inventory.",
            oracle_ties="Official stable increasing-native-tier sort; original recursive insertion "
            "order breaks ties. Query each craft target except root already queried.",
            continuation="One host-selected branch suggestion, then original public.next_action; "
            "no world/ID/gold/depth/RNG arguments. No repairs or action retries.",
            budget=dict(
                continuations=48,
                calls_per_episode=96,
                output_tokens_per_episode=8192,
                max_new_tokens=256,
                context=8192,
                prefix_charged=True,
            ),
            scope="Synthetic diagnostic counterfactuals, not original official task outcomes. "
            "No training on B, no task-unsolvability or novelty claim.",
        ),
    )
    for seed, world in ((42, first), (43, second)):
        save(output / f"world{seed}.json", previous.snapshot(world))
    pairs, outcomes, oracle_records, semantic_records, public_records = [], [], [], [], []
    prefix_calls = 0
    for index, task in enumerate(tasks):
        pair = dict(
            task_id=task["id"],
            goal=task["goal"],
            targets=task["misc"]["target_items"],
            declared_depth=task["misc"]["max_depth"],
            status="construction_started",
        )
        try:
            hybrid, changed = hybrid_world(first, second, pair["targets"])
            worlds = [("world42", first), ("world42-root_world43-lower", hybrid)]
            pair["changed_products"] = changed
            pair["world_sha256"] = {
                name: digest(previous.snapshot(world)) for name, world in worlds
            }
            hybrid_snapshot = previous.snapshot(hybrid)
            save(
                output / f"pair{index:02d}-recipe-delta.json",
                {item: hybrid_snapshot[item] for item in changed},
            )
            requirements = []
            for _, world in worlds:
                solution = solve(world, pair["targets"], extract(world, pair["targets"], {}))
                if solution is None:
                    raise ValueError("official conservative-supply solve failed")
                consumed = Counter()
                for craft in solution[0]:
                    for item, count in craft["ingredients"].items():
                        if world.is_base_item(item):
                            consumed[item] += count
                requirements.append(dict(consumed))
            inventory = {
                item: max(req.get(item, 0) for req in requirements)
                for item in sorted(set().union(*requirements))
            }
            pair["required_bases_by_world"] = requirements
            pair["common_initial_inventory"] = inventory
            if set(pair["targets"]) & inventory.keys():
                raise ValueError("root present at start")
            starts = [prefix(task, world, inventory, tokenizer) for _, world in worlds]
            prefix_calls += 2
            exact_collision([start[4] for start in starts], [start[5] for start in starts])
            pair.update(
                exact_prompt_bytes=True,
                exact_encoded_input_ids=True,
                prompt=starts[0][4],
                prompt_sha256=hashlib.sha256(starts[0][4].encode()).hexdigest(),
                encoded_input_ids=starts[0][5],
                encoded_input_ids_sha256=digest(starts[0][5]),
                root_prefix_rows=[start[3] for start in starts],
            )
            solutions, schedules, labels, visibility = [], [], [], []
            for _, world in worlds:
                solved = solve(world, pair["targets"], inventory)
                if solved is None:
                    raise ValueError("common-stock official solve failed")
                solutions.append(solved[0])
                schedule = oracle_schedule(solved[0], starts[0][2])
                schedules.append(schedule)
                labels.append(schedule[0])
                names = schedule[0].get("items", [schedule[0].get("target_item")])
                visibility.append({name: name in starts[0][4] for name in names if name})
            common_action = public.next_action(
                pair["targets"], dict(inventory), dict(inventory), copy.deepcopy(starts[0][2])
            )
            second_public = public.next_action(
                pair["targets"], dict(inventory), dict(inventory), copy.deepcopy(starts[1][2])
            )
            if common_action != second_public:
                raise ValueError("public controller differs at identical public state")
            pair.update(
                status="constructed",
                oracle_solutions=solutions,
                oracle_action_schedules=schedules,
                oracle_next_actions=labels,
                oracle_target_names_visible=visibility,
                public_next_action=common_action,
                public_target_names_visible={
                    name: name in starts[0][4] for name in common_action.get("items", [])
                },
                next_label_conflict=labels[0] != labels[1],
            )
            collision_id = digest([starts[0][4], starts[0][5]])
            for label in labels:
                oracle_records.append((collision_id, json.dumps(label, separators=(",", ":"))))
                semantic_records.append((collision_id, json.dumps(label, sort_keys=True)))
                public_records.append((collision_id, json.dumps(common_action, sort_keys=True)))
            for world_index, (world_name, _) in enumerate(worlds):
                for suggestion, action in zip(
                    ("oracle42", "oracle_hybrid", "public"), [*labels, common_action], strict=True
                ):
                    if len(outcomes) >= 48:
                        raise ValueError("predeclared continuation cap")
                    receipt, rows = continue_public(task, starts[world_index], action, tokenizer)
                    entry = dict(
                        task_id=task["id"],
                        root=next(iter(pair["targets"])),
                        world=world_name,
                        suggestion=suggestion,
                        forced_action=action,
                        **receipt,
                    )
                    trace = output / "traces" / f"pair{index:02d}-{world_index}-{suggestion}.json"
                    save(trace, dict(receipt=entry, rows=rows))
                    entry["trace_path"], entry["trace_sha256"] = str(trace), sha(trace)
                    outcomes.append(entry)
        except Exception as exc:
            pair["status"] = "construction_or_execution_failure"
            pair["error"] = f"{type(exc).__name__}: {exc}"
        save(output / f"pair{index:02d}.json", pair)
        pairs.append(
            {
                k: v
                for k, v in pair.items()
                if k
                not in (
                    "prompt",
                    "encoded_input_ids",
                    "root_prefix_rows",
                    "oracle_solutions",
                    "oracle_action_schedules",
                    "changed_products",
                )
            }
        )
        print(
            json.dumps(dict(task=task["id"], status=pair["status"], continuations=len(outcomes))),
            flush=True,
        )
    report = dict(
        schema="teacher-counterfactual-witness-result-v1",
        pairs=pairs,
        outcomes=outcomes,
        selected_pairs=8,
        constructed_pairs=sum(p["status"] == "constructed" for p in pairs),
        failures_retained=[p["task_id"] for p in pairs if p["status"] != "constructed"],
        oracle_literal_label_statistics=label_statistics(oracle_records),
        oracle_semantic_label_statistics=label_statistics(semantic_records),
        public_label_statistics=label_statistics(public_records),
        native_continuations=len(outcomes),
        successful_continuations=sum(o["success"] for o in outcomes),
        forced_actions_native_valid=sum(o["forced_action_native_valid"] for o in outcomes),
        prefix_native_calls=prefix_calls,
        newly_executed_continuation_native_calls=sum(
            o["actual_new_native_calls"] for o in outcomes
        ),
        charged_episode_calls_including_shared_prefix=sum(o["calls"] for o in outcomes),
        charged_output_tokens=sum(o["output_tokens"] for o in outcomes),
        CPU_seconds=time.monotonic() - started,
        model_calls=0,
        GPU_used=False,
        limitations="Two artificial completions; common-stock overprovisioning; deterministic "
        "teacher/planner; query intervention rather than arbitrary craft intervention; correlated "
        "roots. Successful continuation is a witness, failure not unsalvageability. No training.",
    )
    save(output / "RESULTS.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("pairs", "outcomes")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
