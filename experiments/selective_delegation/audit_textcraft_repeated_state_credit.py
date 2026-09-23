"""Measure exact saved-state reuse in one completed TextCraft RL rollout batch."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action(text: str) -> str:
    try:
        value = json.loads(text)
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, json.JSONDecodeError):
        return "<non-json>"


def markov_key(prompt: str) -> str | None:
    marker = '\n{"goal": '
    if marker not in prompt:
        return None
    try:
        frame = json.loads(prompt[prompt.index(marker) + 1 :])
    except json.JSONDecodeError:
        return None
    summary = {
        key: frame[key]
        for key in (
            "goal",
            "target_items",
            "inventory_at_task_start",
            "current_inventory",
            "global_calls_remaining",
            "global_output_tokens_remaining",
            "agent_depth",
            "max_agent_depth",
            "delegated_context",
            "history",
        )
    }
    # History is retained deliberately: it is the public source of known recipes.
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def public_recipe_notebook_key(prompt: str) -> tuple[str | None, str | None, str | None]:
    """Return a prospective public notebook summary; this is not a current-policy state."""
    marker = '\n{"goal": '
    if marker not in prompt:
        return None, None, None
    try:
        frame = json.loads(prompt[prompt.index(marker) + 1 :])
    except json.JSONDecodeError:
        return None, None, None
    notebook = {}
    for event in frame["history"]:
        action = event.get("action", {})
        if action.get("action") != "get_info":
            continue
        feedbacks = event.get("feedback", [])
        if not isinstance(feedbacks, list):
            continue
        for feedback in feedbacks:
            if not isinstance(feedback, dict):
                continue
            item = feedback.get("item")
            if item not in action.get("items", []):
                continue
            notebook[item] = {
                key: feedback[key]
                for key in ("item", "can_craft", "is_base", "crafting_depth", "recipes")
                if key in feedback
            }
    known = json.dumps(notebook, sort_keys=True, separators=(",", ":"))
    summary = {
        key: frame[key]
        for key in (
            "goal",
            "target_items",
            "inventory_at_task_start",
            "current_inventory",
            "global_calls_remaining",
            "global_output_tokens_remaining",
            "agent_depth",
            "max_agent_depth",
            "delegated_context",
        )
    }
    summary["observed_recipe_notebook"] = notebook
    inventory = {
        key: summary[key]
        for key in ("goal", "target_items", "inventory_at_task_start", "current_inventory")
    }
    encoded_summary = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    encoded_inventory = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return (
        hashlib.sha256(encoded_summary).hexdigest(),
        hashlib.sha256(known.encode()).hexdigest(),
        hashlib.sha256(encoded_inventory).hexdigest(),
    )


def groups(records: list[dict], key: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record[key] is not None:
            grouped[record[key]].append(record)
    repeated = [rows for rows in grouped.values() if len(rows) > 1]
    cross = [rows for rows in repeated if len({row["episode_id"] for row in rows}) > 1]
    within = [rows for rows in repeated if len({row["episode_id"] for row in rows}) == 1]
    different_action = [rows for rows in repeated if len({row["action"] for row in rows}) > 1]
    reward_variable = [rows for rows in cross if len({row["reward"] for row in rows}) > 1]
    both = [rows for rows in reward_variable if len({row["action"] for row in rows}) > 1]
    action_mean_return = [
        rows
        for rows in both
        if len(
            set(
                {
                action: sum(row["reward"] for row in rows if row["action"] == action)
                / sum(row["action"] == action for row in rows)
                for action in {row["action"] for row in rows}
                }.values()
            )
        )
        > 1
    ]
    root = [rows for rows in repeated if any(row["is_root"] for row in rows)]
    non_root = [rows for rows in repeated if any(not row["is_root"] for row in rows)]
    return {
        "unique_states": len(grouped),
        "repeated_groups": len(repeated),
        "repeated_records": sum(len(rows) for rows in repeated),
        "cross_episode_groups": len(cross),
        "cross_episode_records": sum(len(rows) for rows in cross),
        "within_episode_loop_groups": len(within),
        "within_episode_loop_records": sum(len(rows) for rows in within),
        "different_actions_in_repeated_groups": len(different_action),
        "reward_variable_cross_episode_groups": len(reward_variable),
        "reward_variable_and_different_action_groups": len(both),
        "different_action_groups_with_different_per_action_mean_return": len(action_mean_return),
        "reward_variable_same_action_groups": len(reward_variable) - len(both),
        "repeated_groups_with_root_record": len(root),
        "repeated_groups_with_nonroot_record": len(non_root),
        "distinct_tasks_in_repeated_groups": len(
            {row["task_id"] for rows in repeated for row in rows}
        ),
    }


def audit(batch: Path) -> dict:
    data = json.loads(batch.read_text())
    rollout = Path(data["data"])
    rewards = {row["episode_id"]: row["native_score"] for row in data["episodes"]}
    tasks = {row["episode_id"]: row["task_id"] for row in data["episodes"]}
    records = []
    for path in sorted((rollout / "calls").glob("*.json")):
        call = json.loads(path.read_text())
        if not call.get("available"):
            continue
        tokens = call.get("input_token_ids")
        notebook, recipe_knowledge, inventory = public_recipe_notebook_key(
            call["request"]["prompt"]
        )
        records.append(
            {
                "episode_id": call["episode_id"],
                "task_id": tasks[call["episode_id"]],
                "call_id": call["call_id"],
                "is_root": call["call_id"].endswith("-c000"),
                "reward": rewards[call["episode_id"]],
                "action": action(call.get("text")),
                "exact_input": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                "markov_summary": markov_key(call["request"]["prompt"]),
                "prospective_recipe_notebook_summary": notebook,
                "recipe_knowledge": recipe_knowledge,
                "inventory_only": inventory,
            }
        )
    by_task: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for episode_id, reward in rewards.items():
        by_task[tasks[episode_id]].append((episode_id, reward))
    exact = defaultdict(list)
    for record in records:
        exact[record["exact_input"]].append(record)
    inventory = defaultdict(list)
    for record in records:
        inventory[record["inventory_only"]].append(record)
    inventory_recipe_mismatches = sum(
        len(rows) > 1 and len({row["recipe_knowledge"] for row in rows}) > 1
        for rows in inventory.values()
    )
    inventory_other_mismatches = sum(
        len(rows) > 1
        and len({row["recipe_knowledge"] for row in rows}) == 1
        and len({row["prospective_recipe_notebook_summary"] for row in rows}) > 1
        for rows in inventory.values()
    )
    different_credit = nonroot_different = same_action_return = 0
    for rows in exact.values():
        if len(rows) < 2:
            continue
        for row in rows:
            anchor = row["reward"] - sum(
                other["reward"] for other in rows if other is not row
            ) / (len(rows) - 1)
            peers = [
                reward
                for episode_id, reward in by_task[row["task_id"]]
                if episode_id != row["episode_id"]
            ]
            episode = row["reward"] - sum(peers) / len(peers) if peers else 0.0
            if abs(anchor - episode) > 1e-12:
                different_credit += 1
                nonroot_different += not row["is_root"]
            if all(other["action"] == row["action"] for other in rows):
                same_action_return += 1
    return {
        "schema": "textcraft-repeated-state-credit-feasibility-v1",
        "batch": str(batch),
        "batch_sha256": sha(batch),
        "audit_source_sha256": sha(Path(__file__).resolve()),
        "call_directory_sha256": {
            path.name: sha(path) for path in sorted((rollout / "calls").glob("*.json"))
        },
        "episodes": len(data["episodes"]),
        "available_calls": len(records),
        "exact_model_input_tokens_excluding_sampling_seed": groups(records, "exact_input"),
        "public_markov_summary_including_history": groups(records, "markov_summary"),
        "prospective_public_recipe_notebook_summary_without_history": groups(
            records, "prospective_recipe_notebook_summary"
        ),
        "inventory_only_recipe_knowledge_mismatch_groups": inventory_recipe_mismatches,
        "inventory_only_other_state_mismatch_groups": inventory_other_mismatches,
        "exact_anchor_leave_one_out_vs_episode_rloo": {
            "records_with_different_advantage": different_credit,
            "nonroot_records_with_different_advantage": nonroot_different,
            "repeated_records_with_same_action_group": same_action_return,
            "definition": "Anchor subtracts other calls in same exact-input group; episode RLOO "
            "subtracts other sampled episodes for the task. Counts are correlated nested prefixes."
        },
        "scope": "Exact input identity excludes sampling seed. Markov summary includes public "
        "history because it carries known recipes; inventory-only matches are not counted.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("immutable report exists")
    result = audit(args.batch)
    args.report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
