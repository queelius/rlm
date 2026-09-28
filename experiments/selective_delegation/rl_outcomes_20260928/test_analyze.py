"""Focused post-hoc classification safeguards; no runtime or native changes."""

import analyze
import pytest


def test_native_error_categories_separate_stock_and_recipe_scaling():
    assert analyze.error_kind("Error: Wrong amount of x. Need 2, provided 1") == "wrong_amount"
    assert analyze.error_kind("Error: Insufficient ingredients in inventory: x") == "stock_shortage"
    assert analyze.error_kind("Error: Extra ingredients not required: x") == "extra_ingredient"
    assert analyze.error_kind("Successfully crafted 2 x(s)") is None


def test_duplicate_payload_key_stays_a_rejection_not_a_repaired_action():
    text = (
        '{"action":"craft","ingredients":{"ore":1,"ore":1},'
        '"target_item":"product","output_count":1}'
    )
    assert analyze.schema_reason(text) == "duplicate field"
    with pytest.raises(ValueError, match="duplicate field"):
        analyze.bridge.parse_action(text)


def test_divergence_marks_exact_input_identity_not_just_same_call_index():
    first = [dict(text="a", request=dict(input_token_ids=[1, 2]))]
    second = [dict(text="b", request=dict(input_token_ids=[1, 2]))]
    assert analyze.first_difference(first, second)["exact_input_ids_equal"] is True
    second[0]["request"]["input_token_ids"] = [1, 3]
    assert analyze.first_difference(first, second)["exact_input_ids_equal"] is False
