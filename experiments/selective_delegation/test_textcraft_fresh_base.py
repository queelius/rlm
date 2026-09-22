import argparse
import copy
import importlib.util
import json

import pytest


def test_extended_collection_cap_is_declared_but_scientific_caps_still_match(tmp_path):
    import analyze_textcraft_fresh_base as analysis
    import eval_textcraft_fresh_base as reader

    plan, _ = reader.prepare(argparse.Namespace(output=tmp_path, hours=2, prepare_only=True))
    teacher = json.loads(reader.TEMPLATE.read_text())
    assert plan["budget_seconds"] == 7200 and teacher["budget_seconds"] == 3600
    assert plan["collection_wall_cap_contract"]["per_episode_reasoning_budget_changed"] is False
    analysis.validate_base(plan, teacher)
    for field in (
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
    ):
        changed = copy.deepcopy(plan)
        changed[field] += 1
        with pytest.raises(ValueError, match="contract"):
            analysis.validate_base(changed, teacher)
    changed = copy.deepcopy(plan)
    changed.pop("collection_wall_cap_contract")
    with pytest.raises(ValueError, match="collection"):
        analysis.validate_base(changed, teacher)
    rows = {("one", 0): dict(observed=False, native_score=None)}
    other = {("one", 0): dict(observed=True, native_score=1)}
    result = analysis.profiles.compare([dict(task_id="one", repeat=0)], rows, other)
    assert result["unknown_pairs"] == 1 and result["wins"] == result["losses"] == 0
    assert result["ci95"] is None and result["complete_panel_difference"] is None


def test_base_plan_matches_frozen_fresh_slots_and_removes_all_adapter_metadata(tmp_path):
    assert importlib.util.find_spec("eval_textcraft_fresh_base") is not None
    import analyze_textcraft_fresh_base as analysis
    import eval_textcraft_fresh_base as reader

    args = argparse.Namespace(output=tmp_path, hours=1, prepare_only=True)
    plan, tasks = reader.prepare(args)
    original = json.loads(reader.TEMPLATE.read_text())
    assert len(tasks) == 16 and len(plan["jobs"]) == 32
    assert plan.get("adapter") is None and plan.get("fixed_adapter") is None
    assert "training_plan_sha256" not in plan
    analysis.validate_base(plan, original)
    plan["max_new_tokens"] += 1
    with pytest.raises(ValueError, match="contract"):
        analysis.validate_base(plan, original)


def test_actual_base_native_receipt_rejects_adapter_claim_and_keeps_unknown_pairs():
    assert importlib.util.find_spec("analyze_textcraft_fresh_base") is not None
    import analyze_textcraft_fresh_base as a
    from transformers import AutoTokenizer

    output = a.reader.c.ROOT / "textcraft-pilot-001"
    plan = a.audit.read(output / "PLAN.json")
    call = a.audit.read(sorted((output / "calls").glob("*flat*.json"))[0])
    spec = {
        k: call["request"][k]
        for k in (
            "call_id",
            "prompt",
            "cap",
            "policy",
            "role",
            "node_id",
            "parent_node_id",
            "depth",
            "episode_id",
            "seed",
        )
    }
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
    a.audit.audit_call(call, spec, plan, a.reader.c.inputs.sha(output / "PLAN.json"), tokenizer)
    call["request"]["adapter_enabled"] = True
    with pytest.raises(ValueError):
        a.audit.audit_call(call, spec, plan, a.reader.c.inputs.sha(output / "PLAN.json"), tokenizer)
    jobs = [dict(task_id="one", repeat=0), dict(task_id="one", repeat=1)]
    result = a.profiles.compare(
        jobs,
        {("one", 0): dict(observed=True, native_score=0)},
        {("one", 0): dict(observed=True, native_score=1)},
    )
    assert result["wins"] == 1 and result["unknown_pairs"] == 1
    assert result["complete_panel_difference"] is None and result["ci95"] is None
