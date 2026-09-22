import analyze_textcraft_credit as a


def test_failed_craft_keeps_positive_trajectory_credit():
    rows = [
        dict(
            action="craft",
            error="native_action_error",
            advantage=1 / 3,
            tokens=30,
            episode_id="success",
        ),
        dict(
            action="get_info",
            error="no_explicit_error",
            advantage=-1.0,
            tokens=10,
            episode_id="failure",
        ),
    ]
    result = a.totals(rows)
    assert result["calls"] == 2 and result["tokens"] == 40
    assert result["advantage_weighted_tokens"] == 0
    assert result["absolute_advantage_weighted_tokens"] == 20
    assert (
        a.error_type({"action": {"action": "craft"}, "feedback": "Error: missing item"})
        == "native_action_error"
    )
    assert (
        a.error_type({"response": "bad", "feedback": "Rejected action: parse"}) == "invalid_schema"
    )
