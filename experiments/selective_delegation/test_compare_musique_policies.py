"""Policy comparisons permit architecture changes, not unmatched sampling panels."""

import importlib.util
from pathlib import Path

import pytest


def test_contract_permits_direct_architecture_but_rejects_cases_and_final_sampling_changes():
    path = Path(__file__).with_name("compare_musique_policies.py")
    assert path.exists(), "policy comparison missing"
    spec = importlib.util.spec_from_file_location("policy_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    planner = dict(
        cases_sha256="cases",
        case_ids=["a", "b"],
        repeats=2,
        seed=7,
        temperature=0.5,
        top_p=1.0,
        top_k=0,
        model="base",
        model_manifest_sha256="model",
        split="transfer",
        caps={"root": 128, "helper": 384, "final": 128},
        execution="isolated",
        conditions=["base", "sft"],
    )
    direct = {
        **planner,
        "mode": "direct",
        "execution": "direct",
        "caps": {"final": 128},
        "conditions": ["base"],
        "architecture": "different prompt",
    }
    module.validate_panel([planner, direct])
    for change in (
        {"seed": 8},
        {"temperature": 0.8},
        {"case_ids": ["b", "a"]},
        {"cases_sha256": "other"},
        {"caps": {"final": 256}},
    ):
        with pytest.raises(ValueError, match="matched"):
            module.validate_panel([planner, {**direct, **change}])
