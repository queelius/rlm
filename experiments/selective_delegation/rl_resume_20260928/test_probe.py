"""Focused native-transition and original-token checks for the exploratory probe."""

import importlib.util
import json
import time

import pytest


def probe():
    assert importlib.util.find_spec("probe_common") is not None
    import probe_common

    return probe_common


def test_train_schedule_rejects_val_and_pairs_modes():
    p = probe()
    tasks = [{"id": f"textcraft_synth.train.{i}"} for i in range(8)]
    raw = p.jobs(tasks, "raw", "collect", 1)
    bound = p.jobs(tasks, "binder", "collect", 1)
    assert len(raw) == 32
    assert [(j["task_id"], j["seed"], j["repeat"]) for j in raw] == [
        (j["task_id"], j["seed"], j["repeat"]) for j in bound
    ]
    assert not {j["seed"] for j in raw} & {j["seed"] for j in p.jobs(tasks, "raw", "readout", 1)}
    tasks[0]["id"] = "textcraft_synth.val.0"
    with pytest.raises(ValueError, match="TRAIN"):
        p.jobs(tasks, "binder", "collect", 1)


def test_native_binder_execution_does_not_rewrite_sampled_likelihood_targets(tmp_path):
    p = probe()
    import torch
    from transformers import AutoTokenizer

    world = p.c.bridge.load_world()
    recipe = next(
        rs[0]
        for rs in world.recipes.values()
        if all(world.is_base_item(item) for item in rs[0].ingredients)
    )
    wrong = {item: count + 1 for item, count in recipe.ingredients.items()}
    requested = dict(
        action="craft",
        target_item=recipe.result_item,
        output_count=recipe.result_count,
        ingredients=wrong,
    )
    answers = iter(
        [
            json.dumps(dict(action="get_info", items=[recipe.result_item])),
            json.dumps(requested),
            '{"action":"finish","message":"done"}',
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained(p.c.BASE, local_files_only=True)

    class Model:
        device = torch.device("cpu")

        def parameters(self):
            return []

        def generate(self, **kwargs):
            tokens = tokenizer.encode(next(answers), add_special_tokens=False)
            return torch.cat(
                [kwargs["input_ids"], torch.tensor([tokens + [tokenizer.eos_token_id]])], dim=1
            )

    task = dict(
        id="textcraft_synth.train.fixture",
        goal="Craft the target",
        misc=dict(
            target_items={recipe.result_item: recipe.result_count},
            initial_inventory=dict(recipe.ingredients),
        ),
    )
    job = p.jobs(
        [task] + [dict(task, id=f"textcraft_synth.train.{i}") for i in range(7)],
        "binder",
        "collect",
        1,
    )[0]
    plan = dict(
        model=str(p.c.BASE),
        model_manifest_sha256="fixture",
        adapter=None,
        profile="original",
        jobs=[job],
        planned_episodes=1,
    )
    p.c.save(tmp_path / "PLAN.json", plan)
    client = p.c.NativeClient(
        Model(),
        tokenizer,
        tmp_path,
        time.time() + 60,
        "fixture",
        p.sha(tmp_path / "PLAN.json"),
    )
    collector, _ = p.implementation("binder")
    result = collector.episode(task, job, client, world, tmp_path, time.time() + 60)
    assert result["observed"] and result["native_score"] == 1
    report = p.audit_collection(tmp_path, plan, [task], tokenizer, world, "binder")
    assert report["successes"] == 1
    call = p.read(tmp_path / "calls" / f"{job['episode_id']}-c001.json")
    node = p.read(tmp_path / "nodes" / f"{job['episode_id']}-n0.json")
    assert node["execution_assists"][1]["requested_action"] == requested
    assert node["execution_assists"][1]["executed_action"]["ingredients"] == recipe.ingredients
    _, targets = p.rl.loss_math.causal_inputs(call)
    assert tokenizer.decode(targets, skip_special_tokens=True) == json.dumps(requested)
    node["execution_assists"][1]["requested_action"]["ingredients"] = recipe.ingredients
    p.c.probe.campaign.snapshot(tmp_path / "nodes" / f"{job['episode_id']}-n0.json", node)
    with pytest.raises(ValueError, match="binding mismatch"):
        p.audit_collection(tmp_path, plan, [task], tokenizer, world, "binder")


def test_native_generated_score_and_entropy_match_original_token_replay(tmp_path):
    p = probe()
    import torch
    from collect import MetricsClient
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class Tokenizer:
        eos_token_id = 15
        pad_token_id = 15

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return str(ids)

    client = MetricsClient(model, Tokenizer(), tmp_path, time.time() + 60, "fixture", "plan")
    saved = p.read(p.ROOT / "textcraft-train-readiness-001/calls/t00-r0-flat-c000.json")
    spec = {
        key: saved["request"][key]
        for key in (
            "call_id",
            "prompt",
            "policy",
            "role",
            "depth",
            "node_id",
            "parent_node_id",
            "episode_id",
            "task_id",
            "global_call_index",
            "cap",
            "seed",
            "condition",
            "prompt_profile",
        )
    }
    spec["cap"] = 8
    call = client.call(spec)
    assert call["available"]
    metrics = p.read(tmp_path / "generation-logps" / (spec["call_id"] + ".json"))
    replay = p.rl.action_logps(model, call).flatten().tolist()
    assert replay == pytest.approx(metrics["logps"], abs=1e-5)
    assert len(metrics["token_entropies"]) == len(call["output_token_ids"])
    assert all(0 < h <= 2.773 for h in metrics["token_entropies"])
