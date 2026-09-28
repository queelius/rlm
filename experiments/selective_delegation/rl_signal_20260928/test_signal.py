"""Focused checks for exact-prefix diversity and conservative token attribution."""

import pytest
import signal_analysis as s


def test_projected_choices_ignore_ingredients_but_preserve_target_and_quantity():
    left = {"action": "craft", "ingredients": {"raw": 1}, "target_item": "x", "output_count": 2}
    right = {**left, "ingredients": {"raw": 9}, "note": "try this"}
    assert s.action_key(left) != s.action_key(right)
    assert s.action_key(left, projected=True) == s.action_key(right, projected=True)
    assert s.action_key(left, projected=True) != s.action_key(
        {**right, "output_count": 4}, projected=True
    )
    assert s.action_key(left, projected=True) != s.action_key(
        {**right, "target_item": "y"}, projected=True
    )


def test_json_member_boundary_tokens_are_not_silently_ingredient_tokens():
    text = '{"action":"craft","ingredients":{"r":2},"target_item":"x","output_count":2}'
    start = text.index('"ingredients"')
    end = text.index(',"target_item"')
    offsets = [(start, start + 2), (start + 2, end - 1), (end - 1, end + 2), (0, 1)]
    counts = s.span_counts(text, offsets)
    assert counts == {"ingredients": 2, "ingredient_boundary": 1, "structural": 1}


def test_initial_diversity_requires_identical_policy_prefixes():
    calls = [
        {"input_token_ids": [1, 2], "text": '{"action":"get_info","items":["x"]}'},
        {"input_token_ids": [1, 3], "text": '{"action":"get_info","items":["y"]}'},
    ]
    with pytest.raises(ValueError, match="initial prefixes differ"):
        s.choice_diversity(calls, require_identical=True)


def test_text_variation_does_not_become_semantic_initial_exploration():
    calls = [
        {"input_token_ids": [1, 2], "text": '{"action":"finish","message":"done"}'},
        {
            "input_token_ids": [1, 2],
            "text": '{ "message": "finished", "action": "finish", "note": "ok" }',
        },
    ]
    result = s.choice_diversity(calls, require_identical=True)
    assert result["unique_raw_texts"] == 2
    assert result["unique_semantic_actions"] == 1
    assert result["empirical_semantic_entropy_nats"] == 0


def test_credit_totals_preserve_negative_and_zero_trajectories():
    rows = [
        {"episode_id": "a", "task_id": "t", "tokens": 6, "advantage": 1 / 3},
        {"episode_id": "b", "task_id": "t", "tokens": 9, "advantage": -1},
        {"episode_id": "c", "task_id": "u", "tokens": 5, "advantage": 0},
    ]
    result = s.credit_totals(rows)
    assert (result["calls"], result["episodes"], result["tokens"]) == (3, 3, 20)
    assert result["advantage_weighted_tokens"] == -7
    assert result["absolute_advantage_weighted_tokens"] == 11
