import json
import time

import pytest


def test_paired_parent_denominator_and_unknown_are_distinct():
    import analyze_textcraft as analysis

    jobs = [
        dict(episode_id=f"{p}-{r}-{a}", task_id=str(p), repeat=r, policy=a)
        for p in range(8)
        for r in range(2)
        for a in ("flat", "recursive")
    ]
    rows = {
        j["episode_id"]: dict(j, observed=True, native_score=int(j["policy"] == "recursive"))
        for j in jobs
    }
    result = analysis.paired(jobs, rows, draws=100)
    assert result["parents"] == 8 and result["paired_attempts"] == 16
    assert result["difference"] == 1 and result["ci95"] == [1, 1]
    rows[jobs[0]["episode_id"]]["observed"] = False
    result = analysis.paired(jobs, rows, draws=100)
    assert result["unknown_pairs"] == 1 and result["difference"] is None
    assert result["complete_parents"] == 7


def test_native_saved_episode_replay_checks_nested_inventory_and_ids(tmp_path):
    import analyze_textcraft as analysis
    import eval_textcraft as collector
    import torch

    world = collector.bridge.load_world()
    recipe = next(
        r[0] for r in world.recipes.values() if all(world.is_base_item(i) for i in r[0].ingredients)
    )
    texts = [
        "invalid",
        json.dumps(dict(action="delegate", targets={recipe.result_item: 1}, context="craft")),
        json.dumps(
            dict(
                action="craft",
                ingredients=recipe.ingredients,
                target_item=recipe.result_item,
                output_count=recipe.result_count,
            )
        ),
        '{"action":"finish","message":"child done"}',
        '{"action":"finish","message":"root done"}',
    ]

    class Tokenizer:
        eos_token_id = pad_token_id = 99

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return texts[ids[0] - 10]

    class Model:
        device = torch.device("cpu")
        count = 0

        def parameters(self):
            return []

        def generate(self, **kwargs):
            answer = torch.tensor([[1, 2, 10 + self.count, 99]])
            self.count += 1
            return answer

    task = dict(
        id="task",
        goal="craft",
        misc=dict(target_items={recipe.result_item: 1}, initial_inventory=dict(recipe.ingredients)),
    )
    job = dict(episode_id="ep", task_id="task", repeat=0, seed=2026092204, policy="recursive")
    tokenizer = Tokenizer()
    client = collector.NativeClient(Model(), tokenizer, tmp_path, time.time() + 60, "model", "plan")
    row = collector.episode(task, job, client, world, tmp_path, time.time() + 60)
    calls = {p.stem: json.loads(p.read_text()) for p in (tmp_path / "calls").glob("*.json")}
    nodes = {
        v["node_id"]: v
        for v in (json.loads(p.read_text()) for p in (tmp_path / "nodes").glob("*.json"))
    }
    plan = dict(model=str(collector.BASE), model_manifest_sha256="model")
    result = analysis.audit_episode(task, job, row, calls, nodes, plan, "plan", tokenizer, world)
    assert result["replayed"] and result["native_score"] == 1
    assert result["calls"] == 5 and result["output_tokens"] == 10
    assert result["errors"] == {"invalid_schema": 1} and result["child_successes"] == 1
    calls[row["call_ids"][2]]["input_token_ids"] = [999]
    with pytest.raises(ValueError, match="token"):
        analysis.audit_episode(task, job, row, calls, nodes, plan, "plan", tokenizer, world)
