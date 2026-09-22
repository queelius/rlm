from audit_textcraft_fresh_behavior import craft_error_category, exposure_summary


def test_exposure_summary_keeps_full_panel_primary_and_derives_subset() -> None:
    rows = [
        {"task_id": "seen", "old": 0, "public": 1},
        {"task_id": "unseen", "old": 0, "public": 1},
        {"task_id": "unseen", "old": 1, "public": 0},
    ]
    result = exposure_summary(rows, {"seen"})
    assert result["full_primary"] == {"old_successes": 1, "public_successes": 2, "slots": 3}
    assert result["identifier_absent_supplement"] == {
        "old_successes": 1,
        "public_successes": 1,
        "slots": 2,
    }


def test_craft_error_category_preserves_native_feedback_classes() -> None:
    assert craft_error_category("Error: No recipe found for raw_a2") == "no_recipe"
    assert craft_error_category("Recipe 1: Missing required ingredient raw_a0") == (
        "missing_required"
    )
    assert craft_error_category("Recipe 1: Extra ingredients not required: raw_a0") == (
        "extra_ingredient"
    )
    assert craft_error_category("Recipe 1: Insufficient ingredients in inventory: raw_a0") == (
        "insufficient_inventory"
    )
    assert craft_error_category("Recipe 1: Wrong amount of raw_a0. Need 2, provided 4") == (
        "wrong_amount"
    )
