import json
import time

import pytest


def test_saved052_prompt_only_adds_frozen_procedure_and_keeps_public_projection():
    import eval_textcraft_procedure as control
    from transformers import AutoTokenizer

    c = control.collector
    root = c.ROOT
    case = json.loads((root / "textcraft-inputs-001/tasks.jsonl").read_text().splitlines()[0])
    saved = json.loads(
        (root / "textcraft-trained-readout-001/calls/t00-r0-flat-original-c000.json").read_text()
    )
    frame = c.bridge.Frame(
        {},
        dict(case["misc"]["initial_inventory"]),
        case["misc"]["target_items"],
        c.bridge.Budget(),
        max_depth=0,
    )
    plain = c.bridge.public_prompt(frame, [], goal=case["goal"])
    assert plain == saved["request"]["prompt"]
    prompt = c.render_prompt(frame, [], goal=case["goal"], profile="procedure_control")
    assert prompt == plain + c.PROCEDURAL_INSTRUCTION
    assert "gold_trajectory" not in prompt and "max_depth" not in prompt
    tokenizer = AutoTokenizer.from_pretrained(c.BASE, local_files_only=True)
    kwargs = dict(
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
    )
    assert (
        tokenizer.apply_chat_template([{"role": "user", "content": plain}], **kwargs)
        == saved["input_token_ids"]
    )
    assert (
        len(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)) + 256
        < 8192
    )


def test_procedure_native_receipts_replay_and_changed_suffix_fails(tmp_path):
    import analyze_textcraft as audit
    import eval_textcraft as c
    import torch

    class Tokenizer:
        eos_token_id = pad_token_id = 99

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return '{"action":"finish","message":"done"}'

    class Model:
        device = torch.device("cpu")

        def parameters(self):
            return []

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 99]])

    task = dict(id="task", goal="craft", misc=dict(target_items={"m0_i1": 1}, initial_inventory={}))
    job = dict(episode_id="ep", task_id="task", repeat=0, seed=2026092204, policy="flat")
    client = c.NativeClient(Model(), Tokenizer(), tmp_path, time.time() + 60, "model", "plan")
    row = c.episode(
        task,
        job,
        client,
        c.bridge.load_world(),
        tmp_path,
        time.time() + 60,
        profile="procedure_control",
    )
    calls = {p.stem: json.loads(p.read_text()) for p in (tmp_path / "calls").glob("*.json")}
    nodes = {"n0": json.loads((tmp_path / "nodes/ep-n0.json").read_text())}
    plan = dict(
        model=str(c.BASE),
        model_manifest_sha256="model",
        profile="procedure_control",
        procedural_instruction=c.PROCEDURAL_INSTRUCTION,
    )
    result = audit.audit_episode(
        task, job, row, calls, nodes, plan, "plan", Tokenizer(), c.bridge.load_world()
    )
    assert result["replayed"] and result["native_score"] == 0
    plan["procedural_instruction"] += "CHANGED"
    with pytest.raises(ValueError, match="procedure"):
        audit.audit_episode(
            task, job, row, calls, nodes, plan, "plan", Tokenizer(), c.bridge.load_world()
        )


def test_procedure_jobs_pair_old_original_slots_without_extra_calls():
    import eval_textcraft_procedure as control

    base = [
        dict(
            episode_id=f"t{i:02}-r{r}-flat",
            task_id=str(i),
            repeat=r,
            seed=control.collector.SEEDS[r],
            policy="flat",
        )
        for i in range(8)
        for r in range(2)
    ]
    jobs = control.jobs(base)
    assert len(jobs) == len({j["episode_id"] for j in jobs}) == 16
    assert all(j["prompt_profile"] == "procedure_control" and j["policy"] == "flat" for j in jobs)
    assert [(j["task_id"], j["repeat"], j["seed"]) for j in jobs] == [
        (j["task_id"], j["repeat"], j["seed"]) for j in base
    ]


def test_pair_uses_only_original_baseline_and_preserves_missing():
    import analyze_textcraft_procedure as analysis

    jobs = [dict(task_id=str(i), repeat=r) for i in range(8) for r in range(2)]
    old = {
        str(i): dict(j, condition="trained_original", observed=True, native_score=0)
        for i, j in enumerate(jobs)
    }
    old["reminder"] = dict(
        jobs[0], condition="trained_instruction_control", observed=True, native_score=1
    )
    new = {str(i): dict(j, observed=True, native_score=1) for i, j in enumerate(jobs[:-1])}
    result = analysis.pair(jobs, old, new)
    assert result["planned_pairs"] == 16 and result["wins"] == 15
    assert result["unknown_pairs"] == 1 and result["ci95"] is None
