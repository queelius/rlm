"""Post-hoc native-audit-grounded behavior census; no model or native continuation."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import textcraft_bridge as bridge  # noqa: E402

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
MATRIX = ROOT / "crossed-rl-analysis-20260928-001/CROSSED.json"
OUTPUT = ROOT / "analysis-rl-outcomes-20260928-001"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, expected=None):
    raw = path.read_bytes()
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("native-audited input changed: " + str(path))
    return json.loads(raw)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def error_kind(feedback):
    if not isinstance(feedback, str) or not feedback.startswith("Error:"):
        return None
    for text, name in (
        ("not divisible", "nondivisible_output"),
        ("Missing required ingredient", "missing_ingredient"),
        ("Wrong amount", "wrong_amount"),
        ("Extra ingredients", "extra_ingredient"),
        ("Insufficient ingredients", "stock_shortage"),
        ("No recipe found", "unknown_recipe"),
    ):
        if text in feedback:
            return name
    return "other_native_error"


def schema_reason(text):
    try:
        bridge.parse_action(text)
    except ValueError as exc:
        return str(exc)
    return None


def first_difference(first, second):
    for index, (a, b) in enumerate(zip(first, second, strict=False)):
        if a["text"] != b["text"]:
            return dict(
                zero_based_call_index=index,
                exact_input_ids_equal=(
                    a["request"]["input_token_ids"] == b["request"]["input_token_ids"]
                ),
                first_response=a["text"],
                second_response=b["text"],
            )
    return dict(
        same_responses_through_shorter_episode=True,
        first_calls=len(first),
        second_calls=len(second),
    )


def episode_summary(directory, job, audit):
    episode_id = job["episode_id"]
    pins = audit["receipt_sha256"]

    def checked(path):
        return read(path, pins[str(path)])

    episode = checked(directory / "episodes" / f"{episode_id}.json")
    node = checked(directory / "nodes" / f"{episode_id}-n0.json")
    native = audit["audits"][episode_id]
    assert native["replayed"] and native["native_score"] == episode["native_score"]
    assert len(episode["node_ids"]) == 1
    calls = [checked(directory / "calls" / f"{cid}.json") for cid in episode["call_ids"]]
    history = node["public_history"]
    assert len(calls) == len(history) == native["calls"]
    action_counts, native_errors, schema_errors = Counter(), Counter(), Counter()
    query_items, successful_crafts, schema_details = [], [], []
    invalid_cost = Counter()
    root_met_call, queries_repeated, seen_queries = None, 0, set()
    requested_crafts = changed_arguments = 0
    consecutive_bad = maximum_bad = 0
    for index, (call, row) in enumerate(zip(calls, history, strict=True)):
        assert call["available"]
        state = json.loads(call["request"]["prompt"][len(bridge.INSTRUCTION) :])
        assert state["history"] == history[:index]
        met = all(
            state["current_inventory"].get(k, 0) - state["inventory_at_task_start"].get(k, 0) >= v
            for k, v in node["targets"].items()
        )
        if met and root_met_call is None:
            root_met_call = index
        reason = schema_reason(call["text"])
        category = error_kind(row["feedback"])
        if reason:
            assert "action" not in row and row["response"] == call["text"]
            assert row["feedback"] == f"Rejected action: {reason}. Inventory unchanged."
            schema_errors[reason] += 1
            invalid_cost.update(
                dict(
                    calls=1,
                    output_tokens=call["usage"]["completion_tokens"],
                    prompt_tokens=call["usage"]["prompt_tokens"],
                    native_service_seconds=call["ended"] - call["started"],
                )
            )
            schema_details.append(
                dict(
                    call_id=call["call_id"],
                    text=call["text"],
                    reason=reason,
                    finish_reason=call["finish_reason"],
                )
            )
        else:
            requested = bridge.parse_action(call["text"])
            action = row["action"]
            action_counts[action["action"]] += 1
            if action["action"] == "get_info":
                for item in action["items"]:
                    queries_repeated += item in seen_queries
                    seen_queries.add(item)
                    query_items.append(item)
            if action["action"] == "craft":
                requested_crafts += 1
                changed_arguments += requested["ingredients"] != action["ingredients"]
                if not category:
                    successful_crafts.append(
                        dict(
                            call_index=index,
                            target=action["target_item"],
                            output_count=action["output_count"],
                        )
                    )
            if category:
                native_errors[category] += 1
        if reason or category:
            consecutive_bad += 1
            maximum_bad = max(maximum_bad, consecutive_bad)
        else:
            consecutive_bad = 0
    assert sum(native_errors.values()) == native["errors"].get("native_action_error", 0)
    assert sum(schema_errors.values()) == native["errors"].get("invalid_schema", 0)
    assert action_counts == Counter(native["actions"])
    final_counts = {
        k: node["final_inventory"].get(k, 0) - node["initial_inventory"].get(k, 0)
        for k in node["targets"]
    }
    total_cost = dict(
        calls=len(calls),
        output_tokens=sum(c["usage"]["completion_tokens"] for c in calls),
        prompt_tokens=sum(c["usage"]["prompt_tokens"] for c in calls),
        native_service_seconds=sum(c["ended"] - c["started"] for c in calls),
    )
    return dict(
        task_id=job["task_id"],
        repeat=job["repeat"],
        seed=job["seed"],
        episode_id=episode_id,
        score=native["native_score"],
        status=episode["status"],
        targets=node["targets"],
        final_root_quantities=final_counts,
        action_counts=dict(action_counts),
        native_errors=dict(native_errors),
        schema_errors=dict(schema_errors),
        malformed_response_cost=dict(invalid_cost),
        cost=total_cost,
        premature_finish=episode["status"] == "finished" and native["native_score"] == 0,
        first_root_satisfied_call=root_met_call,
        calls_with_root_satisfied=native["root_calls_after_quantity_met"],
        queried_items=query_items,
        repeated_query_items=queries_repeated,
        successful_crafts=successful_crafts,
        schema_details=schema_details,
        changed_argument_calls=changed_arguments,
        requested_crafts=requested_crafts,
        maximum_consecutive_native_or_schema_errors=maximum_bad,
    ), calls


def aggregate(episodes):
    counters = {
        name: Counter()
        for name in (
            "cost",
            "action_counts",
            "native_errors",
            "schema_errors",
            "malformed_response_cost",
        )
    }
    for episode in episodes:
        for name in counters:
            counters[name].update(episode[name])
    result = {name: dict(value) for name, value in counters.items()}
    result.update(
        episodes=len(episodes),
        successes=sum(e["score"] for e in episodes),
        premature_finishes=sum(e["premature_finish"] for e in episodes),
        context_caps=sum(e["status"] == "context_cap" for e in episodes),
        repeated_query_items=sum(e["repeated_query_items"] for e in episodes),
        successful_craft_calls=sum(len(e["successful_crafts"]) for e in episodes),
        changed_argument_calls=sum(e["changed_argument_calls"] for e in episodes),
        calls_with_root_satisfied=sum(e["calls_with_root_satisfied"] for e in episodes),
    )
    return result


def run(output):
    if output.exists():
        raise FileExistsError("immutable analysis output already exists")
    matrix = read(MATRIX)
    cells, raw_calls, pins = {}, {}, {str(MATRIX): sha(MATRIX)}
    identities = None
    for name, cell in matrix["cells"].items():
        directory = Path(cell["path"])
        plan_path, audit_path = directory / "PLAN.json", directory / "NATIVE-AUDIT.json"
        plan, audit = (
            read(plan_path, cell["plan_sha256"]),
            read(audit_path, cell["native_audit_sha256"]),
        )
        summary = read(directory / "SUMMARY.json")
        assert summary["complete"] and not summary["failure"]
        pins.update(
            {str(plan_path): cell["plan_sha256"], str(audit_path): cell["native_audit_sha256"]}
        )
        current = [(j["task_id"], j["repeat"], j["seed"]) for j in plan["jobs"]]
        if identities is None:
            identities = current
        assert current == identities
        episodes = []
        raw_calls[name] = {}
        for job in plan["jobs"]:
            episode, calls = episode_summary(directory, job, audit)
            episodes.append(episode)
            raw_calls[name][episode["episode_id"]] = calls
        totals = aggregate(episodes)
        assert totals["successes"] == audit["successes"]
        assert totals["cost"]["calls"] == audit["physical_cost"]["calls"]
        assert totals["cost"]["output_tokens"] == audit["physical_cost"]["completion_tokens"]
        cells[name] = dict(
            path=str(directory),
            adapter=plan["adapter"],
            total=totals,
            excluding_429_repeat0=aggregate(
                [e for e in episodes if e["episode_id"] != "t04-r0-flat"]
            ),
            episodes=episodes,
        )
    comparisons = {}
    for trained, warm in (
        ("raw_raw", "warm_raw"),
        ("binder_raw", "warm_raw"),
        ("raw_binder", "warm_binder"),
        ("binder_binder", "warm_binder"),
    ):
        changes, first_differences = [], []
        for before, after in zip(cells[warm]["episodes"], cells[trained]["episodes"], strict=True):
            episode_id = before["episode_id"]
            difference = first_difference(
                raw_calls[warm][episode_id], raw_calls[trained][episode_id]
            )
            first_differences.append(dict(episode_id=episode_id, **difference))
            if before["score"] != after["score"]:
                changes.append(
                    dict(
                        task_id=before["task_id"],
                        repeat=before["repeat"],
                        episode_id=episode_id,
                        before_score=before["score"],
                        after_score=after["score"],
                        before_calls=before["cost"]["calls"],
                        after_calls=after["cost"]["calls"],
                        before_native_errors=before["native_errors"],
                        after_native_errors=after["native_errors"],
                        before_final_root_quantities=before["final_root_quantities"],
                        after_final_root_quantities=after["final_root_quantities"],
                        first_response_difference=difference,
                    )
                )
        cost_delta = {
            key: cells[trained]["total"]["cost"][key] - value
            for key, value in cells[warm]["total"]["cost"].items()
        }
        cost_percent = {
            key: 100 * value / cells[warm]["total"]["cost"][key]
            for key, value in cost_delta.items()
        }
        comparisons[trained + "_vs_" + warm] = dict(
            changed_slots=changes,
            cost_delta=cost_delta,
            cost_change_percent=cost_percent,
            earliest_response_differences=first_differences,
        )
    schema_texts = Counter(
        d["text"] for cell in cells.values() for e in cell["episodes"] for d in e["schema_details"]
    )
    schema_by_slot = Counter(
        e["episode_id"]
        for cell in cells.values()
        for e in cell["episodes"]
        for _ in e["schema_details"]
    )
    result = dict(
        schema="completed-first-RL-outcome-census-v1",
        utc=datetime.now(timezone.utc).isoformat(),
        cells=cells,
        comparisons=comparisons,
        malformed_text_counts=dict(schema_texts),
        malformed_slot_counts=dict(schema_by_slot),
        matrix_uncertainty={
            key: matrix[key]
            for key in (
                "learning_gain_vs_own_tool_baseline",
                "binder_trained_minus_raw_trained_same_tool",
                "tool_effect_at_fixed_weights",
            )
        },
        training_doses={
            name: dict(
                credited_tokens=record["update"]["credited_tokens"],
                native_collection_calls=record["admission"]["physical_cost"]["calls"],
                actual_updates=record["actual_new_optimizer_steps"],
            )
            for name, record in matrix["training"].items()
        },
        source_sha256={
            str(Path(__file__)): sha(Path(__file__)),
            str(HERE / "test_analyze.py"): sha(HERE / "test_analyze.py"),
            str(Path(bridge.__file__)): sha(Path(bridge.__file__)),
        },
        inputs_sha256=pins,
        audit="Existing native audit authoritative; each consumed episode/node/call verified "
        "against its native receipt hash while reading. No model/checkpoint ancestry walk, "
        "new native replay, permissive parse, model load or training.",
        interpretation="Post-hoc action/error census. Eight familiar TRAIN goals, two held "
        "sampling seeds, one realized RL update per interface, not repeated-training-seed "
        "or held-out generalization evidence. First differing output compared on exact input IDs; "
        "later divergent state visitation is not a controlled same-state policy comparison.",
    )
    save(output / "OUTCOMES.json", result)
    print(
        json.dumps(
            dict(
                cells={k: v["total"] for k, v in cells.items()}, malformed_texts=dict(schema_texts)
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
