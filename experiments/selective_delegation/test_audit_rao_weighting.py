import audit_rao_weighting as subject


def test_tree_dependent_weight_makes_loo_baseline_nonzero_but_unweighted_zero():
    row = subject.g2_weighting(0.3)
    assert row["loo_baseline_unweighted_sum_g2"] == 0.0
    assert row["loo_baseline_weighted_sum_g2"] == 0.0315
    assert row["loo_baseline_weighted_sum_g2"] == row["weighted_baseline_closed_form_sum_g2"]


def test_policy_induced_child_distribution_has_a_separate_ancestor_term():
    row = subject.child_distribution_semigradient(0.3, 0.4)
    assert (
        row["root_local_score_term_dp"] + row["omitted_child_distribution_term_dp"]
        == row["full_dJ_dp"]
    )
    assert row["child_score_term_dq"] == row["full_dJ_dq"]
