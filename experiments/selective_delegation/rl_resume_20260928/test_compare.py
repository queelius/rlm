from compare import paired


def test_missing_pair_cannot_be_reported_as_zero_or_dropped_from_estimate():
    result = paired({("one", 0): 1, ("one", 1): None, ("two", 0): 0, ("two", 1): 0})
    assert result["known"] == 3 and result["unknown"] == 1
    assert result["difference"] is None and result["task_cluster_ci95"] is None
    assert result["task_differences"] == {"one": None, "two": 0}


def test_cluster_mean_preserves_two_paired_execution_seeds_per_goal():
    result = paired({("one", 0): 1, ("one", 1): 1, ("two", 0): 1, ("two", 1): 1})
    assert result["difference"] == 1
    assert result["task_cluster_ci95"] == [1, 1]
    assert result["wins"] == 4 and result["unknown"] == 0
