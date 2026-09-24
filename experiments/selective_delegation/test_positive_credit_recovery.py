import copy
import json
from pathlib import Path

import analyze_textcraft_positive_partial as audit
import pytest
from transformers import AutoTokenizer


def test_explicit_subset_inventory_requires_exact_frozen_jobs():
    tasks = {str(i): {} for i in range(16)}
    jobs = [
        dict(task_id="14", episode_id="a"),
        dict(task_id="14", episode_id="b"),
        dict(task_id="15", episode_id="c"),
        dict(task_id="15", episode_id="d"),
    ]
    plan = dict(jobs=jobs, planned_episodes=4)
    audit.validate_inventory(tasks, plan, 16, expected_jobs=jobs)
    with pytest.raises(ValueError):
        audit.validate_inventory(tasks, plan, 16)
    with pytest.raises(ValueError):
        audit.validate_inventory(tasks, plan, 16, expected_jobs=jobs[:3])


def test_actual_interrupted_episode_condition_audits_as_unknown_and_rejects_wrong_role():
    root = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
    output = root / "textcraft-fresh-positive-credit-001"
    plan = audit.read(output / "PLAN.json")
    row = audit.read(output / "episodes/t14-r0-flat-original.json")
    job = next(j for j in plan["jobs"] if j["episode_id"] == row["episode_id"])
    task = next(
        t
        for t in map(json.loads, (Path(plan["prepared"]) / "tasks.jsonl").read_text().splitlines())
        if t["id"] == job["task_id"]
    )
    calls = {cid: audit.read(output / "calls" / (cid + ".json")) for cid in row["call_ids"]}
    nodes = {
        nid: audit.read(output / "nodes" / (row["episode_id"] + "-" + nid + ".json"))
        for nid in row["node_ids"]
    }
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    result = audit.audit_episode(
        task,
        job,
        row,
        calls,
        nodes,
        plan,
        audit.collector.inputs.sha(output / "PLAN.json"),
        tokenizer,
        audit.bridge.load_world(),
    )
    assert result["replayed"] is False and result["native_score"] is None
    assert row["observed"] is False and row["native_score"] is None
    wrong = copy.deepcopy(calls)
    wrong[row["call_ids"][0]]["role"] = "child"
    with pytest.raises(ValueError, match="role/tree identity"):
        audit.audit_episode(
            task,
            job,
            row,
            wrong,
            nodes,
            plan,
            audit.collector.inputs.sha(output / "PLAN.json"),
            tokenizer,
            audit.bridge.load_world(),
        )


def test_whole_subset_audit_retains_four_unobserved_slots(tmp_path):
    root = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
    plan = audit.read(root / "textcraft-fresh-positive-credit-001/PLAN.json")
    jobs = plan["jobs"][-4:]
    plan.update(jobs=jobs, planned_episodes=4, budget_seconds=1800)
    audit.collector.save(tmp_path / "PLAN.json", plan)
    original_owner = audit.read(
        root / "textcraft-fresh-positive-credit-001/OWNER-9c8217f74532.json"
    )
    audit.collector.save(
        tmp_path / "OWNER-fixture.json",
        {"pid": 1073741823, "create_time": 0, "source": original_owner["source"]},
    )
    audit.collector.save(tmp_path / "TERMINAL-fixture.json", {"elapsed_seconds": 0})
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    result = audit.analyze(tmp_path, tokenizer, expected_task_count=16, expected_jobs=jobs)
    group = next(iter(result["groups"].values()))
    assert group["planned"] == 4 and group["observed"] == 0 and group["missing"] == 4
    assert group["success_rate_bounds"] == [0, 1]
