"""Frozen-root helper adaptation contracts, entirely CPU-only."""

import importlib.util
import json
import time
from pathlib import Path

import pytest
from test_eval_planner import CASE, PLAN


def implementation():
    path = Path(__file__).with_name("eval_helper.py")
    assert path.exists(), "helper evaluation implementation missing"
    spec = importlib.util.spec_from_file_location("eval_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job(plan=PLAN):
    return {
        "case_id": CASE["id"],
        "repeat": 0,
        "plan": plan,
        "seed": 123,
        "source_episode_id": "source-sft",
        "source_hashes": {},
        "root": {
            "call_id": "source-root",
            "available": True,
            "request_digest": "frozen-root-digest",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    }


def test_same_frozen_plan_binds_each_arms_actual_predictions_without_gold(tmp_path):
    m = implementation()

    class Client:
        output = tmp_path

        def __init__(self):
            self.requests = []

        def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
            self.requests.append((condition, role, prompt, seed, max_new_tokens))
            answer = "BASE_PERSON" if condition == "base_helper" else "TRAINED_PERSON"
            return {
                "call_id": identity,
                "available": True,
                "role": role,
                "text": json.dumps({"answer": answer}),
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }

    client = Client()
    rows = [m.collect_episode(client, CASE, job(), c) for c in m.CONDITIONS[:2]]
    assert [r[1] for r in client.requests] == ["helper", "helper", "final"] * 2
    assert [r[3:] for r in client.requests[:3]] == [r[3:] for r in client.requests[3:]]
    assert [r[4] for r in client.requests[:3]] == [192, 192, 128]
    assert "Did BASE_PERSON build it?" in client.requests[1][2]
    assert "Did TRAINED_PERSON build it?" in client.requests[4][2]
    assert all(row["plan"] == PLAN for row in rows)
    assert all(row["new_physical_cost"]["calls"] == 3 for row in rows)
    assert all(row["deployed_cost"]["calls"] == 4 for row in rows)
    prompts = "".join(r[2] for r in client.requests)
    assert all(
        secret not in prompts
        for secret in ("GOLD_SECRET", "STEP_SECRET", "REFERENCE_SECRET", "SOURCE_SECRET")
    )


def test_invalid_frozen_root_or_helper_remains_zero_without_fallback(tmp_path):
    m = implementation()

    class Client:
        output = tmp_path
        calls = 0

        def call(self, identity, *args, **kwargs):
            self.calls += 1
            return {
                "call_id": identity,
                "role": "helper",
                "available": True,
                "text": "not JSON",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    client = Client()
    bad_root = m.collect_episode(client, CASE, job(None), "base_helper")
    assert bad_root["status"] == "invalid_plan" and client.calls == 0
    bad_helper = m.collect_episode(client, CASE, job(), "trained_helper")
    assert bad_helper["status"] == "invalid_helper" and client.calls == 1
    assert not bad_root["correct"] and not bad_helper["correct"]


def test_named_adapter_routing_with_real_tiny_cpu_model_and_native_receipts(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    m = implementation()
    torch.set_num_threads(1)
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
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
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
        adapter_name="helper_sft",
    )
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    observed = []
    handle = layer.register_forward_pre_hook(lambda *_: observed.append(layer.disable_adapters))

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return [10, 11]

        def decode(self, tokens, **kwargs):
            return '{"answer":"actual"}'

    client = m.HelperClient(model, Tokenizer(), tmp_path, time.time() + 1000, "helper-hash")
    for index, (condition, role, enabled) in enumerate(
        [
            ("trained_helper", "helper", True),
            ("trained_helper", "final", False),
            ("base_helper", "helper", False),
            ("base_helper", "final", False),
            ("format_reminder", "helper", False),
            ("format_reminder", "final", False),
        ]
    ):
        observed.clear()
        receipt = client.call(str(index), "public", condition, role, 12, max_new_tokens=2)
        assert receipt["available"] and receipt["adapter_enabled"] is enabled
        assert receipt["request"]["adapter_name"] == ("helper_sft" if enabled else None)
        assert observed and all(disabled is not enabled for disabled in observed)
        assert receipt["usage"]["completion_tokens"] == len(receipt["output_token_ids"])
        assert not any(parameter.requires_grad for parameter in model.parameters())
    handle.remove()
    with pytest.raises(ValueError, match="role"):
        client.call("root", "forbidden", "trained_helper", "root", 12, max_new_tokens=2)


def test_saved_root_request_fixture_reconstructs_public_prompt_and_rejects_changes(tmp_path):
    m = implementation()
    root = {
        "call_id": "saved-root",
        "role": "root",
        "condition": "sft",
        "available": True,
        "text": json.dumps(PLAN),
        "adapter_enabled": True,
        "adapter_sha256": "root-hash",
        "request": {
            "prompt": m.evaluation.planner_prompt(CASE),
            "role": "root",
            "condition": "sft",
            "model": "model",
            "adapter_enabled": True,
            "adapter_sha256": "root-hash",
            "seed": 123,
        },
    }
    root["request_digest"] = m.probe.runtime.digest(root["request"])
    assert m.validate_root(root, CASE, "model", "root-hash") == PLAN
    root["request"]["prompt"] += "secret annotation"
    root["request_digest"] = m.probe.runtime.digest(root["request"])
    with pytest.raises(ValueError, match="prompt"):
        m.validate_root(root, CASE, "model", "root-hash")


def test_format_control_only_appends_explicit_json_reminder_and_never_enables_adapter():
    m = implementation()
    original = m.evaluation.isolated_helper_prompt(CASE, "Who made it?")
    reminder = "\nReturn ONLY a JSON object with one string field named answer."
    assert m.helper_prompt(CASE, "Who made it?", "format_reminder") == original + reminder
    for condition in ("base_helper", "trained_helper"):
        assert m.helper_prompt(CASE, "Who made it?", condition) == original
    assert not m.enabled_for("format_reminder", "helper")
    assert not m.enabled_for("format_reminder", "final")


def test_three_arm_summary_counts_shared_root_once_and_keeps_invalid_denominators(tmp_path):
    m = implementation()

    class Client:
        output = tmp_path

        def call(self, *args, **kwargs):
            raise AssertionError("invalid root cannot trigger inference")

    frozen = job(None)
    for condition in m.CONDITIONS:
        m.collect_episode(Client(), CASE, frozen, condition)
    report = m.summarize(tmp_path, {"case_ids": [CASE["id"]], "repeats": 1}, [frozen])
    assert report["all_planned_complete"]
    assert all(row["planned"] == 1 and row["em"] == 0 for row in report["conditions"].values())
    assert report["new_physical_cost"]["calls"] == 0
    assert report["shared_root_acquisition_cost"]["calls"] == 1
    assert report["physical_including_shared_roots_once"]["calls"] == 1
    assert report["paired_f1_trained_minus_base"]["estimate"] == 0
    start = {"call_id": "interrupted-helper", "available": False, "usage": {}}
    m.probe.runtime.save(tmp_path / "starts/interrupted-helper.json", start)
    interrupted = m.summarize(tmp_path, {"case_ids": [CASE["id"]], "repeats": 1}, [frozen])
    assert interrupted["unresolved_start_receipts"] == 1
    assert interrupted["new_physical_cost"]["unknown_usage_calls"] == 1


def test_training_manifest_rejects_wrong_role_or_base_model():
    m = implementation()
    m.validate_helper_training({"role": "helper", "model": "expected-base"}, "expected-base")
    for plan in (
        {"role": "planner", "model": "expected-base"},
        {"role": "helper", "model": "different-base"},
        {"model": "expected-base"},
    ):
        with pytest.raises(ValueError, match="helper training role/model"):
            m.validate_helper_training(plan, "expected-base")
