from audit_textcraft_repeated_state_credit import groups


def test_equal_per_action_mean_returns_do_not_count_as_action_return_difference() -> None:
    rows = [
        {"state": "x", "episode_id": str(index), "task_id": "t", "reward": reward,
         "action": action, "is_root": False}
        for index, (action, reward) in enumerate((("A", 0), ("A", 1), ("B", 0), ("B", 1)))
    ]
    result = groups(rows, "state")
    assert result["reward_variable_and_different_action_groups"] == 1
    assert result["different_action_groups_with_different_per_action_mean_return"] == 0
