import copy
import json
from pathlib import Path

import eval_sufficiency_heldout as shared
import pytest

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")


def test_additive_control_checks_actual_product_plan_and_rejects_dose_or_seed_changes():
    plan = json.loads((ROOT / "sufficiency-rl-001/PLAN.json").read_text())
    product = {"training_plan": plan, "step": 8, "sample_cursor": 8}
    additive = copy.deepcopy(product)
    additive["training_plan"].update(reward_objective="additive", estimator="diagonal")
    shared.validate_additive_control(product, additive)
    for key, value in [("seed", 0), ("learning_rate", 0.001), ("reward_objective", "product")]:
        changed = copy.deepcopy(additive)
        changed["training_plan"][key] = value
        with pytest.raises(ValueError):
            shared.validate_additive_control(product, changed)
    with pytest.raises(ValueError, match="dose"):
        shared.validate_additive_control(product, {**additive, "step": 7})
    changed = copy.deepcopy(additive)
    changed["training_plan"]["estimator"] = "pairing_mean"
    with pytest.raises(ValueError):
        shared.validate_additive_control(product, changed)


def test_actual050_panel_profile_and_four_arm_condition_contract():
    import eval_sufficiency_reward_control as reader

    cases = ROOT / "sufficiency-reward-control-inputs-001/cases.jsonl"
    manifest = shared.read(cases.with_name("MANIFEST.json"))
    profile = reader.profile()
    assert shared.validate_panel(cases, manifest, profile) == (
        "256e9a38fbdb9ddce46f061372096f046cbc8c68063a7e710dca9c1442f97d7c"
    )
    assert profile["component_cluster_count"] == 29
    assert len(shared.jobs(shared.panel.read_jsonl(cases))) == 128
    changed = {**profile, "component_cluster_count": 30}
    with pytest.raises(ValueError, match="component"):
        shared.validate_panel(cases, manifest, changed)


def test_four_arm_missing_summary_preserves512_planned_calls(tmp_path):
    cases = shared.panel.read_jsonl(ROOT / "sufficiency-reward-control-inputs-001/cases.jsonl")
    conditions = [*shared.CONDITIONS, "additive_rl_terminal"]
    plan = {"conditions": conditions, "planned_calls": 512, "jobs": shared.jobs(cases)}
    for condition in conditions:
        shared.save_plan(tmp_path / condition / "PLAN.json", plan)
    summary = shared.summaries(tmp_path, plan, cases)
    assert summary["planned_calls"] == 512
    assert list(summary["conditions"]) == conditions
    assert summary["physical_cost"]["calls"] == 0
    assert all(
        arm["missing_or_inference_unavailable"] == 128 and arm["returned_valid"] == 0
        for arm in summary["conditions"].values()
    )


def test_incomplete_additive_endpoint_writes_explicit_skip_without_tokenizer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    identities = shared.read(ROOT / "sufficiency-compositional-readout-001/PLAN.json")["adapters"]
    additive = copy.deepcopy(identities["rl_terminal"])
    additive.update(step=7, sample_cursor=7)
    additive["training_plan"] = shared.read(ROOT / "sufficiency-additive-rl-001/PLAN.json")
    endpoints = {
        "product": identities["rl_terminal"],
        "sft": identities["matched_sft_terminal"],
        "additive": additive,
    }
    monkeypatch.setattr(
        shared.adapter_runtime, "adapter_identity", lambda *a: identities["warm_joint32"]
    )
    monkeypatch.setattr(shared, "endpoint_identity", lambda path, *a: endpoints[path.name])
    args = SimpleNamespace(
        warm_adapter=tmp_path / "warm",
        rl_output=tmp_path / "product",
        sft_output=tmp_path / "sft",
        additive_rl_output=tmp_path / "additive",
        output=tmp_path / "out",
    )
    assert shared.prepare(args, profile={}) == (None, None, None)
    skip = shared.read(args.output / "SKIPPED.json")
    assert skip["scientific_readout"] is False and skip["chain_resolved"] is True
    assert skip["actual_additive_steps"] == 7 and skip["required_steps"] == 8
    assert not (args.output / "PLAN.json").exists()
