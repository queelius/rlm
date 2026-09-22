import json


def test_actual_completed_native_receipts_reproduce_immutable001_scientific_fields():
    import audit_textcraft_query_transfer as audit

    expected = json.loads((audit.ROOT / "analysis-textcraft-query-transfer-001.json").read_text())
    actual = audit.analyze(audit.ROOT)
    for key in ("groups", "matched_comparisons", "episode_rows", "native_inventory_sha256"):
        assert actual[key] == expected[key], key
    assert actual["groups"]["base_original"]["unknown"] == 3
    assert (
        actual["matched_comparisons"]["trained_original_vs_base_original"]["trained"][
            "first_physical_action_root_query"
        ]
        == 2
    )


def test_teacher_profile_labels_keep_privileged_and_public_arms_unambiguous():
    import audit_textcraft_query_transfer as audit

    assert [row[0] for row in audit.TEACHER_CONFIGS] == [
        "privileged_original",
        "privileged_reminder",
        "public_original",
        "public_reminder",
    ]
    assert audit.TEACHER_PAIRS == (
        ("privileged_original", "public_original"),
        ("privileged_reminder", "public_reminder"),
    )


def test_static_recipe_repeat_uses_strictly_prior_feedback_and_survives_summary():
    import audit_textcraft_query_transfer as audit

    history = [
        {"feedback": [{"item": "X", "recipes": [["wood"]]}]},
        {"feedback": [{"item": "ghost", "recipes": []}]},
    ]
    queries = [{"index": 0, "items": ["X"]}, {"index": 1, "items": ["X", "ghost"]}]
    assert audit.known_static_repeats(queries, history) == (1, 1)
    row = {
        "observed": True,
        **{key: 0 for key in (*audit.BOOLS, *audit.COUNTS)},
        "repeat_returned_static_recipe_calls": 1,
        "repeat_returned_static_recipe_mentions": 1,
    }
    summary = audit.summarize([row], audit.TEACHER_EXTRA_COUNTS)
    assert summary["repeat_returned_static_recipe_calls"] == 1
    assert summary["repeat_returned_static_recipe_mentions"] == 1
