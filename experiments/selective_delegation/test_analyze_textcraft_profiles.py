import json
import time


def test_reminder_native_episode_replays_exact_suffix_and_rejects_changed_prompt(tmp_path):
    import analyze_textcraft as audit
    import eval_textcraft as collector
    import pytest
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
    client = collector.NativeClient(
        Model(), Tokenizer(), tmp_path, time.time() + 60, "model", "plan"
    )
    row = collector.episode(
        task,
        job,
        client,
        collector.bridge.load_world(),
        tmp_path,
        time.time() + 60,
        profile="instruction_control",
    )
    calls = {p.stem: json.loads(p.read_text()) for p in (tmp_path / "calls").glob("*.json")}
    nodes = {"n0": json.loads((tmp_path / "nodes/ep-n0.json").read_text())}
    plan = dict(
        model=str(collector.BASE),
        model_manifest_sha256="model",
        profile="instruction_control",
        instruction_reminder=collector.INSTRUCTION_REMINDER,
    )
    result = audit.audit_episode(
        task, job, row, calls, nodes, plan, "plan", Tokenizer(), collector.bridge.load_world()
    )
    assert result["replayed"] and result["native_score"] == 0
    calls["ep-c000"]["request"]["prompt"] += "ALTERED"
    with pytest.raises(ValueError, match="request"):
        audit.audit_episode(
            task, job, row, calls, nodes, plan, "plan", Tokenizer(), collector.bridge.load_world()
        )


def test_comparison_keeps_missing_base_slots_unknown():
    import analyze_textcraft_profiles as analysis

    jobs = [dict(task_id=str(i), repeat=r, seed=r) for i in range(8) for r in range(2)]
    base = {(j["task_id"], j["repeat"]): dict(observed=True, native_score=0) for j in jobs[:13]}
    new = {(j["task_id"], j["repeat"]): dict(observed=True, native_score=1) for j in jobs}
    result = analysis.compare(jobs, base, new)
    assert result["planned_pairs"] == 16 and result["unknown_pairs"] == 3
    assert result["wins"] == 13 and result["losses"] == 0
    assert result["complete_panel_difference"] is None and result["ci95"] is None
