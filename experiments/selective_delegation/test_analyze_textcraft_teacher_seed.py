def test_single_condition_summary_removes_only_inapplicable_contrasts():
    from analyze_textcraft_teacher_seed import single_condition

    row = dict(
        paired={"invalid": 1},
        depth_strata={"invalid": 2},
        physical_cost={"calls": 3},
        observed=32,
        missing=0,
    )
    assert single_condition(row) == dict(physical_cost={"calls": 3}, observed=32, missing=0)
    assert "paired" in row
