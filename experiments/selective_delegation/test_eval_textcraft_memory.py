import copy
import json
import time
from pathlib import Path

import eval_textcraft_memory as m
import pytest
import torch
from transformers import AutoTokenizer


def test_matched_memory_plans_keep_slots_caps_and_bind_actual_collector():
    template, tasks = m.fixed.template_inputs()
    binding = template["adapter"]
    plans = [m.build_plan(template, "public1", mode, binding) for mode in m.memory.MODES]
    m.comparison.match_slots(plans)
    assert len(tasks) == 16 and all(len(p["jobs"]) == 32 for p in plans)
    assert all(p["budget_seconds"] == 7200 for p in plans)
    assert all(p["max_global_calls"] == 96 and p["max_global_output_tokens"] == 8192 for p in plans)
    assert all(
        p["source_sha256"][str(Path(m.c.__file__).resolve())] == m.sha(Path(m.c.__file__))
        for p in plans
    )
    wrong = copy.deepcopy(plans)
    wrong[1]["budget_seconds"] = 3600
    with pytest.raises(ValueError):
        m.comparison.match_slots(wrong)


def test_actual_five_call_output_layout_replays_native_with_memory_hook(tmp_path):
    template, tasks = m.fixed.template_inputs()
    old = m.c.ROOT / "textcraft-fresh-public-001"
    saved = m.read(old / "episodes/t00-r0-flat-original.json")
    tokens = [
        m.read(old / "calls" / (cid + ".json"))["output_token_ids"] for cid in saved["call_ids"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        m.c.BASE, local_files_only=True, trust_remote_code=False
    )

    class SavedOutputs(torch.nn.Module):
        device = torch.device("cpu")
        peft_config = {"textcraft_action": {}}
        active_adapters = ["textcraft_action"]

        def __init__(self):
            super().__init__()
            self.index = 0

        def generate(self, *, input_ids, attention_mask, generation_config):
            emitted = tokens[self.index]
            self.index += 1
            assert len(emitted) <= generation_config.max_new_tokens
            return torch.cat([input_ids, torch.tensor([emitted])], dim=1)

    plan = m.build_plan(template, "public1", "ledger_recent4", template["adapter"])
    job, task = plan["jobs"][0], tasks[0]
    m.c.save(tmp_path / "PLAN.json", plan)
    plan_sha = m.sha(tmp_path / "PLAN.json")
    client = m.c.NativeClient(
        SavedOutputs(),
        tokenizer,
        tmp_path,
        time.time() + 60,
        plan["model_manifest_sha256"],
        plan_sha,
        adapter=plan["adapter"],
    )
    world = m.c.bridge.load_world()
    with m.memory.installed("ledger_recent4"):
        row = m.c.episode(task, job, client, world, tmp_path, time.time() + 60)
        calls = {p.stem: m.read(p) for p in (tmp_path / "calls").glob("*.json")}
        nodes = {"n0": m.read(tmp_path / "nodes" / (job["episode_id"] + "-n0.json"))}
        result = m.audit.audit_episode(
            task, job, row, calls, nodes, plan, plan_sha, tokenizer, world
        )
    assert row["observed"] and row["native_score"] == 1 and row["global_calls"] == 5
    assert result["native_score"] == 1
    last = calls[row["call_ids"][-1]]
    frame = json.loads(last["request"]["prompt"][len(m.c.bridge.INSTRUCTION) :])
    assert frame["observed_recipe_notebook"] and len(frame["history"]) == 4
    with pytest.raises(ValueError, match="native public request mismatch"):
        m.audit.audit_episode(task, job, row, calls, nodes, plan, plan_sha, tokenizer, world)
