import argparse
import json

import eval_textcraft_trained as trained
import pytest

ROOT = trained.collector.ROOT


def test_actual048_endpoint_explicit_identity_retains_default_and_rejects_wrong_plan():
    adapter = ROOT / "textcraft-action-sft-001/checkpoint-0023"
    binding = trained.endpoint(
        adapter,
        training_plan_sha256=trained.TRAINING_PLAN_SHA256,
        rows_sha256="caa78390f9d4ac28e600674b26e56375203b72d8cdad1c3f9471da3fb25776a9",
    )
    assert binding["state"]["step"] == 23
    assert len(binding["step_receipts_sha256"]) == 23
    with pytest.raises(ValueError, match="training PLAN"):
        trained.endpoint(adapter, training_plan_sha256="incorrect")


def test_public_pending_input_contract_preserves_actual052_schedule(tmp_path):
    import eval_textcraft_public as public

    args = argparse.Namespace(
        prepared=ROOT / "textcraft-inputs-001",
        adapter=ROOT / "textcraft-public-discovery-sft-001/checkpoint-0023",
        output=tmp_path / "readout",
        hours=1.5,
    )
    plan, tasks, binding = public.prepare(args, require_endpoint=False)
    previous = json.loads((ROOT / "textcraft-trained-readout-001/PLAN.json").read_text())
    assert binding is None and len(tasks) == 8
    assert not (args.output / "PLAN.json").exists()
    for key in (
        "jobs",
        "sampling",
        "seeds",
        "seed_rule",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "budget_seconds",
        "initial_token_audit_by_profile",
        "tasks_sha256",
    ):
        assert plan[key] == previous[key], key
    assert plan["training_plan_sha256"] != previous["training_plan_sha256"]
    assert plan["teaching_policy"] == "public_observation_prerequisite_discovery"
    assert plan["teacher_contract"]["rows"] == 366
    assert plan["primary_comparison"] == "public_vs_privileged_original_prompt"


def test_public_contract_rejects_wrong_teacher_and_partial_actual_state():
    import eval_textcraft_public as public

    directory = ROOT / "textcraft-public-discovery-sft-001"
    contract = json.loads((directory / "TEACHER-CONTRACT.json").read_text())
    plan = json.loads((directory / "PLAN.json").read_text())
    public.validate_teacher(plan, contract)
    with pytest.raises(ValueError, match="teacher"):
        public.validate_teacher(plan, {**contract, "rows_sha256": "privileged"})
    old = ROOT / "textcraft-action-sft-001"
    state = json.loads((old / "checkpoint-0023/STATE.json").read_text())
    steps = [json.loads(p.read_text()) for p in sorted((old / "steps").glob("*.json"))]
    trained.validate_dose(plan, state, steps)
    with pytest.raises(ValueError, match="checkpoint23"):
        trained.validate_dose(plan, {**state, "step": 22}, steps)
