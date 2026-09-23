"""Actual saved native request/replay fixture, without model generation."""

import json

import textcraft_seed_world43 as reader


def test_actual_saved_world43_native_episode():
    from transformers import AutoTokenizer

    c, audit = reader.c, reader.profiles.audit
    output = c.ROOT / "textcraft-world43-privileged-001"
    plan = json.loads((output / "PLAN.json").read_text())
    job = plan["jobs"][0]
    row = json.loads((output / "episodes" / (job["episode_id"] + ".json")).read_text())
    tasks = [
        json.loads(line)
        for line in (c.ROOT / "textcraft-world43-inputs-001/tasks.jsonl").read_text().splitlines()
    ]
    task = next(t for t in tasks if t["id"] == job["task_id"])
    calls = {
        cid: json.loads((output / "calls" / (cid + ".json")).read_text()) for cid in row["call_ids"]
    }
    nodes = {
        nid: json.loads((output / "nodes" / (job["episode_id"] + "-" + nid + ".json")).read_text())
        for nid in row["node_ids"]
    }
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    audit.audit_episode(
        task,
        job,
        row,
        calls,
        nodes,
        plan,
        c.inputs.sha(output / "PLAN.json"),
        tokenizer,
        reader.panel.worlds()[1],
    )
