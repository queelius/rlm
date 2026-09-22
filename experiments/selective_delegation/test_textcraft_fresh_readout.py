import argparse
import importlib
import json

import pytest


@pytest.mark.parametrize("parents,repeats,expected", [(8, 2, 16), (16, 2, 32), (8, 4, 32)])
def test_planned_unknown_counts_follow_jobs(tmp_path, parents, repeats, expected):
    import eval_textcraft as c

    jobs = [
        dict(episode_id=f"{p}-{r}", task_id=str(p), repeat=r, policy="flat")
        for p in range(parents)
        for r in range(repeats)
    ]
    group = c.summarize(tmp_path, {"jobs": jobs})["groups"]["flat"]
    assert group["planned"] == expected
    assert group["missing_or_unknown"] == expected
    assert group["won"] == 0


def test_native_audit_inventory_preserves_default_eight_and_explicit_fresh_sixteen():
    import analyze_textcraft as audit

    tasks = {str(i): {} for i in range(16)}
    jobs = [
        dict(episode_id=f"{p}-{r}", task_id=p, repeat=r, policy="flat")
        for p in tasks
        for r in range(2)
    ]
    plan = {"jobs": jobs, "planned_episodes": 32}
    with pytest.raises(ValueError, match="inventory"):
        audit.validate_inventory(tasks, plan)
    audit.validate_inventory(tasks, plan, expected_task_count=16)
    with pytest.raises(ValueError, match="inventory"):
        audit.validate_inventory(tasks, {**plan, "jobs": jobs[:-1]}, expected_task_count=16)


def test_fresh_fixed_endpoint_plans_match_jobs_and_unknown_pair_math(tmp_path):
    reader = importlib.import_module("eval_textcraft_fresh")
    compare = importlib.import_module("analyze_textcraft_fresh")
    plans = []
    for teacher in ("privileged", "public"):
        args = argparse.Namespace(
            prepared=reader.c.ROOT / "textcraft-fresh-inputs-001",
            output=tmp_path / teacher,
            teacher=teacher,
            hours=1.0,
            prepare_only=True,
        )
        plan, tasks, binding = reader.prepare(args)
        assert len(tasks) == 16 and len(plan["jobs"]) == 32
        assert plan["world_seed"] == 42 and binding["state"]["step"] == 23
        assert plan["budget_seconds"] == 3600 and plan["planned_per_condition"] == 32
        plans.append(plan)
    compare.validate_pair(plans)
    jobs = plans[0]["jobs"]
    left = {(j["task_id"], j["repeat"]): dict(observed=True, native_score=0) for j in jobs}
    right = {k: dict(observed=True, native_score=1) for k in left}
    result = compare.profiles.compare(jobs, left, right)
    assert result["planned_pairs"] == 32 and result["complete_panel_difference"] == 1
    assert result["ci95"] == [1, 1]
    del right[next(iter(right))]
    result = compare.profiles.compare(jobs, left, right)
    assert result["unknown_pairs"] == 1 and result["complete_panel_difference"] is None
    assert result["ci95"] is None


def test_actual_saved_trained_call_ids_decode_adapter_and_sampling():
    import analyze_textcraft as audit
    from transformers import AutoTokenizer

    output = audit.collector.ROOT / "textcraft-trained-readout-001"
    plan = json.loads((output / "PLAN.json").read_text())
    call = json.loads(sorted((output / "calls").glob("*.json"))[0].read_text())
    req = call["request"]
    spec = {
        k: req[k]
        for k in (
            "call_id",
            "prompt",
            "cap",
            "policy",
            "condition",
            "role",
            "node_id",
            "parent_node_id",
            "depth",
            "episode_id",
            "seed",
        )
    }
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
    audit.audit_call(call, spec, plan, audit.collector.inputs.sha(output / "PLAN.json"), tokenizer)
    call["request"]["adapter_sha256"] = "wrong"
    with pytest.raises(ValueError):
        audit.audit_call(
            call, spec, plan, audit.collector.inputs.sha(output / "PLAN.json"), tokenizer
        )
