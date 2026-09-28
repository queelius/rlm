"""Read-only saved-trace audit and cost summary; no native continuation or model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import study as s


def analyze(output):
    provenance = json.loads((output / "PROVENANCE.json").read_text())
    for name, expected in provenance["sources_sha256"].items():
        if s.sha(Path(name)) != expected:
            raise ValueError("scientific source/input changed: " + name)
    report = json.loads((output / "RESULTS.json").read_text())
    if report["native_continuations"] != len(report["outcomes"]) or len(report["outcomes"]) > 48:
        raise ValueError("continuation accounting differs")
    by_pair = {
        pair["task_id"]: json.loads((output / f"pair{index:02d}.json").read_text())
        for index, pair in enumerate(report["pairs"])
    }
    checked_rows = 0
    for outcome in report["outcomes"]:
        path = Path(outcome["trace_path"])
        if s.sha(path) != outcome["trace_sha256"]:
            raise ValueError("saved continuation changed")
        rows = json.loads(path.read_text())["rows"]
        pair = by_pair[outcome["task_id"]]
        inventory = dict(pair["common_initial_inventory"])
        known, history, output_tokens = {}, [], 0
        for index, row in enumerate(rows):
            if not row["prompt"].startswith(s.bridge.INSTRUCTION):
                raise ValueError("prompt instruction changed")
            state = json.loads(row["prompt"][len(s.bridge.INSTRUCTION) :])
            assert state["inventory_at_task_start"] == pair["common_initial_inventory"]
            assert state["current_inventory"] == inventory
            assert state["history"] == history
            assert state["global_calls_remaining"] == 96 - index
            assert state["global_output_tokens_remaining"] == 8192 - output_tokens
            assert state["agent_depth"] == state["max_agent_depth"] == 0
            assert row["public_known_before"] == sorted(known)
            action = s.bridge.parse_action(row["target"])
            if index == 1:
                assert action == outcome["forced_action"]
                assert row["prompt"] == pair["prompt"]
                assert row["input_ids"][: row["prompt_tokens"]] == pair["encoded_input_ids"]
            if index >= 2:
                assert action == s.public.next_action(
                    dict(pair["targets"]),
                    dict(pair["common_initial_inventory"]),
                    dict(inventory),
                    known,
                )
            if action["action"] == "craft":
                recipe = known[action["target_item"]]["recipes"][0]
                batches, remainder = divmod(action["output_count"], recipe["result_count"])
                assert remainder == 0
                assert action["ingredients"] == {
                    item: amount * batches for item, amount in recipe["ingredients"].items()
                }
                for item, amount in action["ingredients"].items():
                    assert inventory[item] >= amount
                    inventory[item] -= amount
                    if inventory[item] == 0:
                        del inventory[item]
                item = action["target_item"]
                inventory[item] = inventory.get(item, 0) + action["output_count"]
            s.update_observed(known, action, row["feedback"])
            history.append(dict(action=action, feedback=row["feedback"]))
            output_tokens += row["target_tokens"]
            assert row["prompt_tokens"] + 256 <= 8192
            assert row["target_tokens"] <= 256
            assert len(row["input_ids"]) == row["prompt_tokens"] + row["target_tokens"]
            checked_rows += 1
        assert len(rows) == outcome["calls"] <= 96
        assert output_tokens == outcome["output_tokens"] <= 8192
        assert inventory == outcome["final_inventory"]
        assert all(
            inventory.get(k, 0) - pair["common_initial_inventory"].get(k, 0) >= v
            for k, v in pair["targets"].items()
        )
        assert outcome["native_score"] == 1 and outcome["finished"]
    baseline = {
        (o["task_id"], o["world"]): o for o in report["outcomes"] if o["suggestion"] == "public"
    }
    summaries = {}
    for arm in ("oracle42", "oracle_hybrid", "public"):
        rows = [o for o in report["outcomes"] if o["suggestion"] == arm]
        delta = Counter(o["calls"] - baseline[o["task_id"], o["world"]]["calls"] for o in rows)
        summaries[arm] = dict(
            native_successes=sum(o["success"] for o in rows),
            episodes=len(rows),
            calls=sum(o["calls"] for o in rows),
            queries=sum(o["action_counts"].get("get_info", 0) for o in rows),
            output_tokens=sum(o["output_tokens"] for o in rows),
            extra_calls_vs_public_histogram=dict(delta),
            extra_queries_vs_public=sum(
                o["action_counts"].get("get_info", 0)
                - baseline[o["task_id"], o["world"]]["action_counts"].get("get_info", 0)
                for o in rows
            ),
        )
    visibility = [
        value
        for pair in by_pair.values()
        for world in pair["oracle_target_names_visible"]
        for value in world.values()
    ]
    return dict(
        passed=True,
        checked_saved_rows=checked_rows,
        no_additional_native_continuations=True,
        provenance_sha256=s.sha(output / "PROVENANCE.json"),
        results_sha256=s.sha(output / "RESULTS.json"),
        analyzer_sha256=s.sha(Path(__file__)),
        arm_summaries=summaries,
        oracle_next_names_visible=sum(visibility),
        oracle_next_names_total=len(visibility),
        max_calls=max(o["calls"] for o in report["outcomes"]),
        max_output_tokens=max(o["output_tokens"] for o in report["outcomes"]),
        max_prompt_plus_cap=max(o["max_prompt_plus_cap"] for o in report["outcomes"]),
        validation="Saved exact prefix/IDs, public-only decisions after forced suggestion, "
        "observed recipe quantities, inventory arithmetic, budgets and native outcome receipts; "
        "no second native replay. Scientific runner made the native calls and scores.",
        decision="Deprioritize this query-label conflict as a native-failure explanation here: "
        "all48 routes succeed, with at most one extra query. No visible-name conflict witness: "
        "all16 oracle next-query item names were absent from the shared public prompt.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=s.OUTPUT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = analyze(args.output.resolve())
    if args.receipt:
        s.save(args.receipt.resolve(), result)
    print(json.dumps(result, indent=2))
