import json


def test_actual_world43_changes_recipe_and_constructed_task_replays_with_real_tokenizer():
    import prepare_textcraft_world43 as panel
    from transformers import AutoTokenizer

    old, world = panel.worlds()
    task = dict(
        id="fixture",
        goal="Craft the following items: 1x m0_i2",
        misc=dict(target_items={"m0_i2": 1}, initial_inventory={"m5_ore": 1}),
    )
    changed = panel.construct(task, world)
    tokenizer = AutoTokenizer.from_pretrained(panel.BASE, local_files_only=True)
    result = panel.qualify(changed, world, tokenizer)
    assert result["native_score"] == 1 and result["max_prompt_plus_cap"] <= 8192
    frame = panel.bridge.Frame(
        world,
        dict(changed["misc"]["initial_inventory"]),
        changed["misc"]["target_items"],
        panel.bridge.Budget(),
        max_depth=0,
    )
    recipe = old.get_recipes_for_item("m0_i2")[0]
    rejected = frame.apply(
        dict(
            action="craft",
            ingredients=recipe.ingredients,
            target_item="m0_i2",
            output_count=recipe.result_count,
        )
    )
    assert rejected.startswith("Error:") and "not divisible" in rejected
    prompt = panel.bridge.initial_prompt(changed, "flat")
    assert "gold_trajectory" not in prompt and "recipe_seed" not in prompt
    public = json.loads(prompt.splitlines()[-1])
    assert public["target_items"] == task["misc"]["target_items"]
    assert public["current_inventory"] == changed["misc"]["initial_inventory"]
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
