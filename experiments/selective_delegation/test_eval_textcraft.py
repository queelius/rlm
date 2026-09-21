import json
import time


def test_nested_returns_share_counts_and_do_not_expose_native_reward(tmp_path):
    import eval_textcraft as run

    class Client:
        def __init__(self):
            self.requests = []
            self.responses = iter(
                [
                    {"action": "delegate", "targets": {"m0_i1": 1}, "context": "root context"},
                    {"action": "delegate", "targets": {"m0_i1": 1}, "context": "child context"},
                    {"action": "finish", "message": "grandchild actual text"},
                    {"action": "finish", "message": "child actual text"},
                    {"action": "finish", "message": "root actual text"},
                ]
            )

        def ids(self, prompt):
            return [1]

        def call(self, request):
            self.requests.append(request)
            return {
                "call_id": request["call_id"],
                "available": True,
                "text": json.dumps(next(self.responses)),
                "output_token_ids": [1, 2],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

    task = {
        "id": "textcraft-test",
        "goal": "Craft one",
        "misc": {"target_items": {"m0_i1": 1}, "initial_inventory": {}},
    }
    job = {
        "episode_id": "one",
        "task_id": task["id"],
        "policy": "recursive",
        "repeat": 0,
        "seed": 2026092204,
    }
    client = Client()
    result = run.episode(task, job, client, run.bridge.load_world(), tmp_path, time.time() + 60)
    assert result["observed"] and result["native_score"] == 0
    assert result["global_calls"] == 5 and result["global_output_tokens"] == 10
    assert [r["depth"] for r in client.requests] == [0, 1, 2, 1, 0]
    assert [r["global_call_index"] for r in client.requests] == list(range(5))
    assert "grandchild actual text" in client.requests[3]["prompt"]
    assert "native_score" not in client.requests[3]["prompt"]
    assert "child actual text" in client.requests[4]["prompt"]
    client = Client()
    result = run.episode(
        task,
        {**job, "episode_id": "capped"},
        client,
        run.bridge.load_world(),
        tmp_path,
        time.time() + 60,
        budget=run.bridge.Budget(3, 20),
    )
    assert result["observed"] and result["status"] == "global_call_cap"
    assert result["global_calls"] == 3 and len(client.requests) == 3


def test_native_frozen_client_records_actual_generation_and_node_request(tmp_path):
    import eval_textcraft as run
    import torch
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
    for p in model.parameters():
        p.requires_grad_(False)

    class Tokenizer:
        eos_token_id = 15
        pad_token_id = 15

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return "native:" + str(ids)

    client = run.NativeClient(model, Tokenizer(), tmp_path, time.time() + 60, "manifest", "plan")
    call = client.call(
        {
            "call_id": "one",
            "prompt": "public-only",
            "policy": "recursive",
            "role": "child",
            "depth": 1,
            "node_id": "n1",
            "parent_node_id": "n0",
            "episode_id": "ep",
            "task_id": "task",
            "global_call_index": 0,
            "seed": 7,
            "cap": 8,
        }
    )
    assert call["available"] and 1 <= len(call["output_token_ids"]) <= 8
    assert not call["request"]["adapter_enabled"]
    assert call["request"]["depth"] == 1 and call["request"]["parent_node_id"] == "n0"
    assert call["usage"]["completion_tokens"] == len(call["output_token_ids"])
    assert call["request"]["sampling"]["temperature"] == 0.5
    assert all(not p.requires_grad for p in model.parameters())


def test_rejected_json_charged_and_child_crafting_counts_for_original_root(tmp_path):
    import eval_textcraft as run

    world = run.bridge.load_world()
    recipe = next(
        r[0] for r in world.recipes.values() if all(world.is_base_item(i) for i in r[0].ingredients)
    )
    responses = iter(
        [
            "not JSON",
            json.dumps(
                {"action": "delegate", "targets": {recipe.result_item: 1}, "context": "craft this"}
            ),
            json.dumps(
                {
                    "action": "craft",
                    "ingredients": recipe.ingredients,
                    "target_item": recipe.result_item,
                    "output_count": recipe.result_count,
                }
            ),
            '{"action":"finish","message":"child done"}',
            '{"action":"finish","message":"root done"}',
        ]
    )

    class Client:
        def ids(self, prompt):
            return [1]

        def call(self, request):
            if request["global_call_index"] == 1:
                assert "Rejected action" in request["prompt"]
                assert '"global_calls_remaining": 95' in request["prompt"]
            return {
                "call_id": request["call_id"],
                "available": True,
                "text": next(responses),
                "output_token_ids": [1, 2],
                "usage": {"completion_tokens": 2},
            }

    task = {
        "id": "textcraft-test",
        "goal": "Craft one",
        "misc": {
            "target_items": {recipe.result_item: 1},
            "initial_inventory": dict(recipe.ingredients),
        },
    }
    job = {
        "episode_id": "crafted",
        "task_id": task["id"],
        "policy": "recursive",
        "repeat": 0,
        "seed": 2026092204,
    }
    result = run.episode(task, job, Client(), world, tmp_path, time.time() + 60)
    assert result["native_score"] == 1 and result["global_calls"] == 5
    assert result["global_output_tokens"] == 10 and result["errors"] == {"invalid_schema": 1}
    assert recipe.result_item not in result["root_initial_inventory"]
