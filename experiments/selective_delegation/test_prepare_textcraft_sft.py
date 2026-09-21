import json

import pytest


def test_selection_rejects_validation_root_and_never_uses_gold():
    import prepare_textcraft_sft as prep

    rows = [
        {
            "id": f"d{d}-{i}",
            "misc": {
                "max_depth": d,
                "target_items": {f"target-{d}-{i}": 1},
                "gold_trajectory": "not inspected",
            },
        }
        for d in (2, 3, 4)
        for i in range(20)
    ]
    validation = [{"id": "d2-0", "misc": {"target_items": {"target-3-0": 1}}}]
    picked, _ = prep.select(rows, validation)
    assert len(picked) == 32
    assert [sum(x["misc"]["max_depth"] == d for x in picked) for d in (2, 3, 4)] == [11, 11, 10]
    assert not {"d2-0", "d3-0"} & {x["id"] for x in picked}
    for x in rows:
        x["misc"]["gold_trajectory"] = "changed future secret"
    assert [x["id"] for x in prep.select(rows, validation)[0]] == [x["id"] for x in picked]


@pytest.fixture(scope="module")
def native_fixture():
    import prepare_textcraft_sft as prep
    import textcraft_bridge as bridge
    from transformers import AutoTokenizer

    world = bridge.load_world()
    recipe = next(
        r[0]
        for r in world.recipes.values()
        if r[0].result_count > 1 and all(world.is_base_item(i) for i in r[0].ingredients)
    )
    task = {
        "id": "PRIVATE_TASK_ID",
        "goal": "Craft the requested target.",
        "misc": {
            "initial_inventory": dict(recipe.ingredients),
            "target_items": {recipe.result_item: 1},
            "max_depth": "PRIVATE_DEPTH",
            "future_secret": "PRIVATE_FUTURE",
            "gold_trajectory": [
                {
                    "ingredients": dict(recipe.ingredients),
                    "target": [recipe.result_item, 1],
                    "result_count": recipe.result_count,
                }
            ],
        },
    }
    tokenizer = AutoTokenizer.from_pretrained(
        prep.BASE, local_files_only=True, trust_remote_code=False
    )
    return task, world, tokenizer


def test_native_public_info_craft_finish_and_action_only_eos_mask(native_fixture):
    import prepare_textcraft_sft as prep
    import textcraft_bridge as bridge

    task, world, tokenizer = native_fixture
    receipt, rows = prep.trajectory(task, world, tokenizer)
    assert receipt["native_score"] == 1 and receipt["eligible"]
    assert [json.loads(r["target"])["action"] for r in rows] == ["get_info", "craft", "finish"]
    assert rows[0]["prompt"] == bridge.initial_prompt(task, "flat")
    assert '"recipes": []' not in rows[1]["prompt"]
    assert '"feedback": [{"item":' in rows[1]["prompt"]
    for row in rows:
        assert all(
            s not in row["prompt"] for s in ("PRIVATE_TASK_ID", "PRIVATE_DEPTH", "PRIVATE_FUTURE")
        )
        n = row["prompt_tokens"]
        assert row["labels"][:n] == [-100] * n
        assert row["labels"][n:] == row["input_ids"][n:]
        assert row["labels"][-1] == tokenizer.eos_token_id
        assert tokenizer.decode(row["labels"][n:-1], skip_special_tokens=False) == row["target"]
    assert json.loads(rows[1]["target"])["output_count"] > 1


def test_capped_trajectory_stays_failed_not_replaced(native_fixture):
    import prepare_textcraft_sft as prep

    task, world, tokenizer = native_fixture
    receipt, rows = prep.trajectory(task, world, tokenizer, max_calls=1)
    assert not receipt["eligible"] and receipt["native_score"] == 0
    assert receipt["status"] == "global_call_cap" and len(rows) == 1
