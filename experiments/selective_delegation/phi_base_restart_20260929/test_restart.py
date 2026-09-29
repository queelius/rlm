"""Catch metadata-container mismatches without accepting scientific plan changes."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")


def implementation():
    path = Path(__file__).with_name("restart_v2.py")
    assert path.exists(), "bounded Phi base restart is not implemented"
    spec = importlib.util.spec_from_file_location("phi_base_restart", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_plan(world=42):
    return json.loads((ROOT / f"textcraft-phi-base-w{world}-20260928-001/PLAN.json").read_text())


def test_actual_saved_plan_accepts_only_json_equivalent_binding_containers():
    module = implementation()
    original = saved_plan()
    candidate = copy.deepcopy(original)
    for key in ("native_client", "saved_call_auditor"):
        binding = candidate["native_stop_binding"][key]
        binding["substitutions"] = [tuple(row) for row in binding["substitutions"]]
    assert candidate != original
    assert module.canonical_matching_plan(original, candidate) == original


@pytest.mark.parametrize("field,value", [("world_seed", 51), ("seeds", [2026092299])])
def test_restart_rejects_changed_scientific_inputs(field, value):
    module = implementation()
    original = saved_plan()
    candidate = dict(original, **{field: value})
    with pytest.raises(ValueError, match="scientific plan"):
        module.canonical_matching_plan(original, candidate)


def test_restart_retains_recorded_native_stop_ids():
    module = implementation()
    original = saved_plan(50)
    candidate = copy.deepcopy(original)
    candidate["native_stop_binding"]["stop_ids"] = [199999]
    with pytest.raises(ValueError, match="scientific plan"):
        module.canonical_matching_plan(original, candidate)


def test_runtime_namespace_reaches_actual_collectors_existing_attempt_guard(tmp_path):
    module = implementation()
    args = module.scientific_arguments(42, tmp_path)
    collector, _, _ = module.original_module().native_adapter.implementation("raw")
    (tmp_path / "calls").mkdir()
    # This actual collector guard is before torch imports, locking or model loading.
    # It can be reached only after the collector has read args.prepare_only.
    with pytest.raises(ValueError, match="existing scientific attempt"):
        collector.run(args, prepared_run=({"jobs": []}, []))
