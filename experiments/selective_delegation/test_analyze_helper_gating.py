import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).with_name("analyze_helper_gating.py")
    spec = importlib.util.spec_from_file_location("helper_gating", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_bootstrap_point_estimate_weights_all_parent_rows_not_clusters_equally():
    module = load_module()
    rows = [
        {"case_id": "a", "rule": 1, "plan": 0},
        {"case_id": "b", "rule": 1, "plan": 0},
        {"case_id": "c", "rule": 0, "plan": 1},
    ]
    estimate, _, _ = module.bootstrap(rows, [["a", "b"], ["c"]], seed=1)
    assert estimate == 1 / 3


def test_bootstrap_keeps_both_repeats_before_parent_average():
    module = load_module()
    rows = [
        {"case_id": "a", "rule": 1, "plan": 0},
        {"case_id": "a", "rule": 0, "plan": 1},
        {"case_id": "b", "rule": 0, "plan": 0},
        {"case_id": "b", "rule": 0, "plan": 0},
    ]
    estimate, lower, upper = module.bootstrap(rows, [["a"], ["b"]], seed=1)
    assert (estimate, lower, upper) == (0.0, 0.0, 0.0)
