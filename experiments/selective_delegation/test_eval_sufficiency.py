import json
import time

import pytest


def test_native_enabled_adapter_request_preserves_public_contract(tmp_path):
    import eval_sufficiency as evaluation
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
            )
        ),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    enabled = []
    layer = next(module for module in model.modules() if hasattr(module, "lora_A"))
    layer.register_forward_pre_hook(
        lambda module, args: enabled.append(not module.disable_adapters)
    )

    class Tokenizer:
        eos_token_id = 31
        pad_token_id = 31

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert "SECRET" not in messages[0]["content"]
            return [1, 2, 3]

        def decode(self, ids, **kwargs):
            return '{"answerable":false,"answer":""}'

    case = {
        "answer": "SECRET",
        "public": {
            "question": "Where?",
            "documents": [
                {"docid": "d0", "title": "Public", "text": "Public text", "answer": "SECRET"}
            ],
        },
    }
    identity = {"files": {"adapter_model.safetensors": "fixture"}, "arm": "joint", "step": 32}
    client = evaluation.AdapterClient(model, Tokenizer(), tmp_path, time.time() + 90, identity)
    row = client.call(
        "fixture",
        evaluation.prompt(case),
        "joint",
        "sufficiency",
        evaluation.baseline.SEEDS[0],
        2,
        {"truncation": False},
    )
    assert row["available"] and enabled and all(enabled)
    request = row["request"]
    assert request["adapter_enabled"] and request["adapter"] == identity
    assert request["seed"] == 2026092181 and request["sampling"]["temperature"] == 0.5
    assert request["role"] == "sufficiency" and request["condition"] == "joint"
    assert row["request_digest"] == evaluation.native.probe.runtime.digest(request)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert len(row["output_token_ids"]) <= 2
    assert torch.cuda.is_initialized() is False


def test_incomplete_checkpoint_never_substituted(tmp_path):
    import eval_sufficiency as evaluation

    adapter = tmp_path / "checkpoint-0031"
    adapter.mkdir()
    (adapter / "COMMIT.json").write_text(json.dumps({"step": 31, "files": {}}))
    (adapter / "STATE.json").write_text(json.dumps({"step": 31, "epoch": 0, "cursor": 496}))
    (tmp_path / "PLAN.json").write_text(json.dumps({"role": "sufficiency"}))
    with pytest.raises(ValueError, match="step32"):
        evaluation.adapter_identity(adapter, "joint")
