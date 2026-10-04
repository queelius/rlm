"""Completed pair admission and fixed endpoints, without native or weight reads."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def plotting():
    path = Path(__file__).with_name("plot_completed_500k_pairs.py")
    assert path.exists(), "Completed-pair plotter is not implemented"
    spec = importlib.util.spec_from_file_location("completed_pairs_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot():
    path = Path(__file__).parent / "extension-data" / "two-500k-pairs-summary.json"
    return json.loads(path.read_text())


def test_selects_admitted_pairs_and_exact_endpoint_values_without_unfinished_seed():
    selected = plotting().select_pairs(snapshot())
    assert [pair["seed"] for pair in selected] == [123, 456]
    assert [pair["curl"]["endpoint_return"] for pair in selected] == [
        842.7435524938227,
        866.1352708875662,
    ]
    assert [pair["no_curl"]["endpoint_return"] for pair in selected] == [
        866.9394162304535,
        810.9574102627266,
    ]
    assert [[len(pair[a]["curve"]) for a in ("curl", "no_curl")] for pair in selected] == [
        [127, 127],
        [126, 126],
    ]
    assert [
        [pair[a]["physical_training_env_steps_for_completed_chain"] for a in ("curl", "no_curl")]
        for pair in selected
    ] == [[602000, 500000], [500000, 500000]]


@pytest.mark.parametrize("fault", ["duplicate", "incomplete", "endpoint"])
def test_ambiguous_or_unvalidated_fixed_endpoint_is_rejected(tmp_path, fault):
    data = snapshot()
    run = next(r for r in data["runs"] if r["seed"] == 456 and r["arm"] == "curl")
    if fault == "duplicate":
        data["runs"].append(copy.deepcopy(run))
    elif fault == "incomplete":
        run["status"] = "incomplete"
    else:
        run["curve"][-1]["mean_return"] = 999
    with pytest.raises(ValueError):
        plotting().select_pairs(data)


def test_single_admitted_pair_remains_supported():
    data = snapshot()
    data["pairs"] = data["pairs"][:1]
    assert [pair["seed"] for pair in plotting().select_pairs(data)] == [123]
