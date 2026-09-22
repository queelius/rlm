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
