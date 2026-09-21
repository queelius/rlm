"""CPU-only train-fit request replay, paired math, and frozen native routing."""

import inspect
import json
import time

import pytest


def test_pair_math_keeps_missing_and_protocol_separate():
    import analyze_rl_trainfit as analysis

    def row(reward, valid=True):
        return {"reward": reward, "score": {"valid": valid, "f1": float(reward or 0)}}

    result = analysis.paired_changes(
        [row(0, False), row(1), row(0), row(1)],
        [row(1), row(0), row(0), None],
    )
    assert result["planned"] == 4 and result["missing_after"] == 1
    assert result["wins_protocol"] == result["losses_both_valid"] == 1
    assert result["em_difference"] == -0.25


def test_source_replay_rejects_seed_change(tmp_path):
    import eval_rl_trainfit as evaluation
    import probe
    from test_eval_planner import CASE

    source = tmp_path / "source"

    class Fake:
        output = source
        helper_contract = {
            "mode": "trained_helper",
            "adapter_binding": {"adapter_model.safetensors": "helper"},
        }

        def call(self, identity, prompt, role, seed, cap):
            request = evaluation.expected_request(
                prompt, role, seed, cap, "root", self.helper_contract
            )
            request["input_token_ids"] = [1, 2]
            record = {
                "call_id": identity,
                "request": request,
                "request_digest": probe.runtime.digest(request),
                "available": True,
                "role": role,
                "adapter_enabled": request["adapter_enabled"],
                "adapter_sha256": request["adapter_sha256"],
                "model_instance": request["model_instance"],
                "model": request["model"],
                "input_token_ids": [1, 2],
                "output_token_ids": [3],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                "text": '{"subquestions":["Question?"]}' if role == "root" else '{"answer":"Gold"}',
            }
            probe.runtime.save(source / "calls" / (identity + ".json"), record)
            probe.runtime.save(source / "starts" / (identity + ".json"), record)
            return record

    native = Fake()
    original, _ = evaluation.rl.rollout(native, CASE, 1, 0, 0)
    replay, _ = evaluation.replay_episode(source, CASE, 0, 0, "root", native.helper_contract)
    assert original == replay
    root_path = source / "calls" / (original["episode_id"] + "-root.json")
    record = json.loads(root_path.read_text())
    record["request"]["seed"] += 1
    root_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="request"):
        evaluation.replay_episode(source, CASE, 0, 0, "root", native.helper_contract)


def test_native_frozen_roles_have_no_optimizer_and_preserve_temperatures(tmp_path):
    import eval_rl_trainfit as evaluation
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)

    def model():
        return get_peft_model(
            Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=16,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    eos_token_id=2,
                    pad_token_id=2,
                )
            ),
            LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        )

    root, helper = model(), model()
    observed = []
    handles = []
    for name, instance in (("root", root), ("helper", helper)):
        layer = instance.base_model.model.model.layers[0].self_attn.q_proj

        def hook(*_, name=name, layer=layer, instance=instance):
            observed.append(
                (name, layer.disable_adapters, any(p.requires_grad for p in instance.parameters()))
            )

        handles.append(layer.register_forward_pre_hook(hook))

    class Tokenizer:
        eos_token_id = pad_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return [3, 4]

        def decode(self, *args, **kwargs):
            return '{"answer":"a"}'

    contract = {
        "mode": "trained_helper",
        "adapter_binding": {"adapter_model.safetensors": "helper"},
    }
    client = evaluation.FrozenClient(
        root,
        Tokenizer(),
        tmp_path,
        time.time() + 120,
        "root",
        helper_contract=contract,
        helper_model=helper,
    )
    records = [client.call(role, "prompt", role, 50, 1) for role in ("root", "helper", "final")]
    assert [
        (r["request"]["temperature"], r["model_instance"], r["adapter_enabled"]) for r in records
    ] == [(0.8, "root", True), (0.5, "helper", True), (0.5, "root", False)]
    assert ("root", False, False) in observed and ("helper", False, False) in observed
    assert ("root", True, False) in observed and all(not row[2] for row in observed)
    assert not any(
        p.requires_grad or p.grad is not None for m in (root, helper) for p in m.parameters()
    )
    assert "torch.optim" not in inspect.getsource(evaluation)
    for handle in handles:
        handle.remove()
