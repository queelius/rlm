import copy

import pytest


def test_four_arm_partition_accepts512_without_loosening_old384_default():
    import analyze_sufficiency_rl as shared

    parents = [f"p{i}" for i in range(32)]
    cases = [{"parent_id": p} for p in parents for _ in range(2)]
    clusters = [parents[:2], parents[2:4], parents[4:6], *[[p] for p in parents[6:]]]
    manifest = {"component_clusters": clusters}
    assert shared.frozen_clusters(cases, manifest, 512, expected_calls=512) == clusters
    with pytest.raises(ValueError):
        shared.frozen_clusters(cases, manifest, 512)
    with pytest.raises(ValueError):
        shared.frozen_clusters(cases, manifest, 384, expected_calls=512)


def test_four_conditions_keep_sft_bound_to_original_product_not_additive():
    import analyze_sufficiency_reward_control as report

    plan = {
        "conditions": [
            "warm_joint32",
            "rl_terminal",
            "matched_sft_terminal",
            "additive_rl_terminal",
        ],
        "planned_calls": 512,
        "control_binding": {
            "matched_sft_control_for": "rl_terminal",
            "rl_terminal_reward": "product",
            "additive_rl_terminal_reward": "additive",
            "equal_actual_steps_and_sample_cursors": True,
        },
    }
    report.validate_condition_contract(plan)
    changed = copy.deepcopy(plan)
    changed["control_binding"]["matched_sft_control_for"] = "additive_rl_terminal"
    with pytest.raises(ValueError):
        report.validate_condition_contract(changed)
    with pytest.raises(ValueError):
        report.validate_condition_contract({**plan, "planned_calls": 384})


def test_four_arm_cluster_math_retains_two_repeats_and_component_dependence():
    import analyze_sufficiency_rl as shared

    left = [{"parent_id": p, "seed": s, "joint_em": 0} for p in ("a", "b", "c") for s in (1, 2)]
    right = [{**r, "joint_em": int(r["parent_id"] in ("a", "b"))} for r in left]
    result = shared.cluster_contrast(left, right, "joint_em", [["a", "b"], ["c"]], draws=1000)
    assert result["estimate"] == pytest.approx(2 / 3)
    assert result["ci95"] == [0, 1]


def test_actual_product_endpoint_and_changed_dose_not_an_additive_training_audit():
    import json
    from pathlib import Path

    import analyze_sufficiency_reward_control as report

    root = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
    identities = json.loads((root / "sufficiency-compositional-readout-001/PLAN.json").read_text())[
        "adapters"
    ]
    product = identities["rl_terminal"]
    result = report.audit_endpoint(product, report.shared.Audit())
    assert result["mode"] == "rl" and result["seed"] == 2026092194
    with pytest.raises(ValueError, match="terminal endpoint"):
        report.audit_endpoint({**product, "step": 7}, report.shared.Audit())
