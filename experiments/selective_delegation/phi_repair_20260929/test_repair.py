"""Focused Phi repair data, saved-plan and runtime-namespace seams."""

import importlib.util
from pathlib import Path

import pytest


def package():
    path = Path(__file__).with_name("common.py")
    assert path.exists(), "Phi repair package not implemented"
    spec = importlib.util.spec_from_file_location("phi_repair_test_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_row_order_and_native_labels_are_matched():
    rows = [dict(task_id="a", step=0, target="first"), dict(task_id="a", step=1, target="last")]
    examples = [dict(target_ids=[1, 2]), dict(target_ids=[3, 2])]
    assert package().match_targets(rows, rows, examples, examples) == 4
    with pytest.raises(ValueError, match="target row"):
        package().match_targets(list(reversed(rows)), rows, examples, examples)
    with pytest.raises(ValueError, match="native target"):
        package().match_targets(rows, rows, [dict(target_ids=[2, 1]), examples[1]], examples)


def test_plan_roundtrip_normalizes_only_json_representations():
    value = dict(native_stop_binding=dict(substitutions=[("old", "new")]), count=8)
    result = package().canonical(value)
    assert result == dict(native_stop_binding=dict(substitutions=[["old", "new"]]), count=8)
    assert result == package().canonical(result)


def test_repaired_native_step_uses_original_action_index_for_target_matching():
    known = [dict(task_id="a", step=0, target="first"), dict(task_id="a", step=1, target="last")]
    repaired = [
        dict(known[0], step=1, source_action_index=0),
        dict(known[1], step=0, source_action_index=1),
    ]
    examples = [dict(target_ids=[1, 2]), dict(target_ids=[3, 2])]
    assert package().match_targets(repaired, known, examples, examples) == 4
    assert repaired[0]["step"] == 1


def test_actual_runtime_namespace_has_prepare_only_false(tmp_path):
    args = package().runtime_args(42, tmp_path)
    assert args.prepare_only is False
    assert args.teacher == "stable_visible"
    assert args.assistance == "raw"
    assert args.profile == "original"
    assert args.hours == 0.5


def test_pending_endpoint_is_never_fabricated(tmp_path):
    with pytest.raises(FileNotFoundError, match="ENDPOINT-AUDIT"):
        package().cached_endpoint(tmp_path)
