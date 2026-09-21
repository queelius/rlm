"""CPU-only direct adapter control: native routing, exact prompt/seed and denominator."""

import importlib.util
import time
from pathlib import Path

import pytest
from test_eval_planner import CASE


def implementation():
    path = Path(__file__).with_name("eval_direct_adapted.py")
    assert path.exists(), "standalone direct readout missing"
    spec = importlib.util.spec_from_file_location("direct_adapted", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed_base", [None, 2026092112])
def test_native_direct_calls_use_only_helper_adapter_and_match_baseline_request(
    tmp_path, seed_base
):
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
    ).eval()
    observed, prompts = [], []
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    handle = layer.register_forward_pre_hook(lambda *_: observed.append(layer.disable_adapters))

    class Tokenizer:
        eos_token_id = pad_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return [10, 11]

        def decode(self, *args, **kwargs):
            return '{"answer":"GOLD_SECRET"}'

    extra = {} if seed_base is None else {"seed_base": seed_base}
    client = m.DirectClient(model, Tokenizer(), tmp_path, time.time() + 1000, "helper-sha", **extra)
    for condition, enabled in (("base_direct", False), ("helper_sft_direct", True)):
        observed.clear()
        row = client.call(CASE, condition, 1)
        assert row["available"] and row["role"] == "direct_answer"
        assert row["adapter_enabled"] == enabled
        assert row["adapter_sha256"] == ("helper-sha" if enabled else None)
        assert observed and all(value is not enabled for value in observed)
        assert not any(p.requires_grad for p in model.parameters())
        request = row["request"]
        assert request["prompt"] == m.evaluation.direct_prompt(CASE)
        assert (
            request["seed"]
            == (m.evaluation.SEED if seed_base is None else seed_base)
            + int(m.probe.runtime.digest(CASE["id"])[:6], 16)
            + 102
        )
        assert request["sampling"] == {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 128,
            "max_time": 90.0,
        }
        assert row["usage"]["completion_tokens"] == len(row["output_token_ids"])
        assert row["request_digest"] == m.probe.runtime.digest(request)
    handle.remove()
    assert len(prompts) == 2 and all("GOLD_SECRET" not in prompt for prompt in prompts)
    cached = client.call(CASE, "base_direct", 1)
    assert cached["available"] and client.returned == 2
    reference = m.evaluation.HFClient(
        model, Tokenizer(), tmp_path / "old-direct", time.time() + 1000, "unused"
    ).call(
        "saved-direct", m.evaluation.direct_prompt(CASE), "base", "final", cached["request"]["seed"]
    )
    for field in (
        "prompt",
        "input_token_ids",
        "seed",
        "sampling",
        "model",
        "adapter_enabled",
        "adapter_sha256",
    ):
        assert reference["request"][field] == cached["request"][field]
    assert reference["output_token_ids"] == cached["output_token_ids"]
    with pytest.raises(ValueError, match="cached request"):
        client.call({**CASE, "question": "Changed?"}, "base_direct", 1)


def test_direct_summary_scores_official_aliases_and_keeps_all_planned_attempts(tmp_path):
    m = implementation()
    case = {**CASE, "metadata": {"answer_aliases": ["ALIAS"]}}
    call = {
        "call_id": "a",
        "condition": "helper_sft_direct",
        "role": "direct_answer",
        "available": True,
        "text": '{"answer":"ALIAS"}',
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    m.probe.runtime.save(tmp_path / "calls/a.json", call)
    m.probe.runtime.save(
        tmp_path / "episodes/a.json",
        {
            "episode_id": "a",
            "case_id": case["id"],
            "condition": "helper_sft_direct",
            "repeat": 0,
            "call_ids": ["a"],
            "status": "scored",
            **m.probe.grade(call["text"], case),
        },
    )
    result = m.summarize(tmp_path, {"case_ids": [case["id"]], "repeats": 2})
    assert result["conditions"]["helper_sft_direct"]["em"] == 0.5
    assert result["conditions"]["helper_sft_direct"]["missing"] == 1
    assert result["conditions"]["base_direct"]["planned"] == 2
    assert result["physical_cost"]["calls"] == 1
    assert not result["all_planned_complete"]


def test_panel_requires_all_64_unique_development_parents():
    m = implementation()
    cases = [{**CASE, "id": str(i), "split": "development"} for i in range(64)]
    m.validate_panel(cases)
    for bad in (
        cases[:32],
        cases[:-1] + [cases[0]],
        [{**case, "split": "train"} for case in cases],
    ):
        with pytest.raises(ValueError, match="64"):
            m.validate_panel(bad)


def test_hotpot_panel_and_actual_yes_no_grading(tmp_path):
    m = implementation()
    cases = [
        {**CASE, "id": str(i), "split": "transfer", "dataset": "hotpotqa", "answer": "yes"}
        for i in range(32)
    ]
    m.validate_panel(cases, "hotpot_explorer32")
    for bad in (
        cases[:31],
        [{**c, "split": "train"} for c in cases],
        [{**c, "dataset": "musique"} for c in cases],
    ):
        with pytest.raises(ValueError):
            m.validate_panel(bad, "hotpot_explorer32")

    class Client:
        output = tmp_path

        def call(self, case, condition, repeat):
            return {
                "call_id": f"{case['id']}-{repeat}",
                "available": True,
                "text": '{"answer":"yes indeed"}',
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }

    row = m.collect_episode(Client(), cases[0], "base_direct", 0)
    assert row["valid"] and not row["correct"] and row["f1"] == 0
    assert row["metric"] == "official_hotpotqa_em_f1"
    assert m.panel_spec()["seed"] == m.evaluation.SEED
    assert m.panel_spec("hotpot_explorer32")["seed"] == 2026092112
