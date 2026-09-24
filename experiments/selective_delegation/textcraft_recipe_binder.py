"""Public-observation craft argument binding; never chooses targets or quantities."""

import copy


def bind_observed_recipe_arguments(action: dict, history: list[dict]) -> tuple[dict, str]:
    if action.get("action") != "craft":
        return action, "not_craft"
    recipes = None
    for row in history:
        queried = row.get("action", {})
        feedback = row.get("feedback")
        if queried.get("action") != "get_info" or not isinstance(feedback, list):
            continue
        for info in feedback:
            if isinstance(info, dict) and info.get("item") == action.get("target_item"):
                recipes = info.get("recipes")
    if not isinstance(recipes, list) or len(recipes) != 1:
        return action, "recipe_not_observed"
    recipe = recipes[0]
    output = action.get("output_count")
    count = recipe.get("result_count")
    ingredients = recipe.get("ingredients")
    if not isinstance(output, int) or not isinstance(count, int) or output % count:
        return action, "quantity_not_divisible"
    valid_ingredients = isinstance(ingredients, dict) and all(
        isinstance(value, int) for value in ingredients.values()
    )
    if not valid_ingredients:
        return action, "recipe_not_bindable"
    bound = copy.deepcopy(action)
    batches = output // count
    bound["ingredients"] = {item: amount * batches for item, amount in ingredients.items()}
    return bound, "bound_observed_single_recipe"
