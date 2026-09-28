"""Focused public-data and matched-length checks, independent of native world access."""

import copy
import importlib.util
import re
from pathlib import Path


def module():
    path = Path(__file__).with_name("demand_table.py")
    assert path.exists(), "table intervention not implemented"
    spec = importlib.util.spec_from_file_location("public_demand_table_fixture", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return re.findall(r"\w+|[^\w\s]", messages[0]["content"])


def payload():
    return dict(
        target_items={"root": 1},
        inventory_at_task_start={"ore": 3},
        current_inventory={"ore": 3},
        history=[
            dict(
                action=dict(action="get_info", items=["root"]),
                feedback=[
                    dict(item="root", recipes=[dict(ingredients={"mid": 2}, result_count=1)])
                ],
            )
        ],
    )


def test_unobserved_recipe_stays_unknown_and_all_fact_columns_match():
    table = module()
    state = payload()
    before = copy.deepcopy(state)
    demand, masked = table.pair_payloads(state)
    assert state == before
    left = {r["item"]: r for r in demand["public_quantity_table"]}
    right = {r["item"]: r for r in masked["public_quantity_table"]}
    assert sorted(left) == ["mid", "ore", "root"]
    assert left["mid"]["observed_recipes"] is None
    assert left["mid"]["remaining_demand"] == 2
    assert left["mid"]["new_batches"] is None
    assert left["mid"]["frontier_deficit_kind"] == "unqueried_recipe"
    assert left["ore"]["remaining_demand"] is None
    for item, row in left.items():
        for key, value in row.items():
            if key in table.DERIVED:
                assert right[item][key] is None
            else:
                assert right[item][key] == value


def test_both_prompts_keep_complete_history_and_match_encoded_length():
    table = module()
    state = payload()
    prompts, lengths = table.render_pair("instruction\n", state, Tokenizer())
    assert lengths["demand"] == lengths["masked"]
    assert all('"history"' in p and '"get_info"' in p for p in prompts.values())
    assert all('"matched_length_padding"' in p for p in prompts.values())
    assert all("do not recommend an action" in p for p in prompts.values())


def test_query_recipe_is_accepted_only_if_actually_requested():
    table = module()
    state = payload()
    state["history"][0]["feedback"].append(
        dict(item="unrequested_hidden", recipes=[dict(ingredients={"ore": 1}, result_count=1)])
    )
    try:
        table.pair_payloads(state)
    except ValueError as exc:
        assert "unrequested" in str(exc)
    else:
        raise AssertionError("unrequested recipe was accepted")


def test_only_public_base_reply_justifies_known_base_deficit():
    table = module()
    state = payload()
    state["history"][0]["feedback"][0]["recipes"][0]["ingredients"] = {"ore": 5}
    demand, _ = table.pair_payloads(state)
    ore = next(r for r in demand["public_quantity_table"] if r["item"] == "ore")
    assert ore["stock_deficit"] == 2
    assert ore["frontier_deficit_kind"] == "unqueried_initial_stock"
    state["history"].append(
        dict(
            action=dict(action="get_info", items=["ore"]),
            feedback=[dict(item="ore", is_base=True, recipes=[])],
        )
    )
    demand, _ = table.pair_payloads(state)
    ore = next(r for r in demand["public_quantity_table"] if r["item"] == "ore")
    assert ore["frontier_deficit_kind"] == "known_base"
