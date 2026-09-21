def test_fixed_strata_selection_is_outcome_independent():
    import prepare_textcraft as prep

    rows = [
        {"id": str(i), "misc": {"max_depth": 2 if i < 8 else 4, "gold_trajectory": [{"secret": i}]}}
        for i in range(16)
    ]
    selected = prep.select(rows)
    assert len(selected) == 8
    assert sum(r["misc"]["max_depth"] == 2 for r in selected) == 4
    for row in rows:
        row["misc"]["gold_trajectory"] = [{"secret": "changed"}]
    assert [r["id"] for r in prep.select(rows)] == [r["id"] for r in selected]
