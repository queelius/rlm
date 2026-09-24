from textcraft_recipe_binder import bind_observed_recipe_arguments


def test_binds_only_previously_observed_single_divisible_recipe() -> None:
    history = [
        {
            "action": {"action": "get_info", "items": ["widget"]},
            "feedback": [
                {
                    "item": "widget",
                    "recipes": [{"ingredients": {"ore": 2}, "result_count": 3}],
                }
            ],
        }
    ]
    action = {"action": "craft", "target_item": "widget", "output_count": 6, "ingredients": {}}
    bound, reason = bind_observed_recipe_arguments(action, history)
    assert bound["ingredients"] == {"ore": 4}
    assert reason == "bound_observed_single_recipe"


def test_does_not_bind_unseen_or_nondivisible_recipe() -> None:
    action = {"action": "craft", "target_item": "widget", "output_count": 5, "ingredients": {}}
    bound, reason = bind_observed_recipe_arguments(action, [])
    assert bound == action
    assert reason == "recipe_not_observed"
