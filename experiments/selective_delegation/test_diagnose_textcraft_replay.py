import diagnose_textcraft_replay as d
import pytest


def test_selection_is_outcome_blind_and_keeps_all_last_calls():
    calls = {str(i): dict(input_token_ids=[0] * (i + 1), output_token_ids=[1]) for i in range(60)}
    episodes = [dict(call_ids=[str(i)]) for i in range(32)]
    selected = d.select_calls(calls, episodes)
    assert len(selected) == 44 and set(map(str, range(32))) <= set(selected)
    for call in calls.values():
        call["reward"] = 99
    assert d.select_calls(calls, episodes) == selected


def test_gap_metrics_record_token_identity_and_signed_gaps():
    result = d.compare([-1.0, -2.0], [-1.2, -1.7], [10, 11])
    assert result["max_abs"] == pytest.approx(0.3)
    assert result["mean_abs"] == pytest.approx(0.25)
    assert result["sequence_sum_gap"] == pytest.approx(0.1)
    assert result["top_tokens"][0]["token_id"] == 11
    with pytest.raises(ValueError):
        d.compare([-1.0], [-1.0, -2.0], [10])


def test_full_scan_thresholds_and_posthoc_calls_preserve_fixed_selection():
    records = {
        cid: dict(original_cached_vs_full=d.compare([0.0], [gap], [1]))
        for cid, gap in (("fixed", 0.0), ("z", 0.3), ("a", 0.3), ("low", 0.01))
    }
    result = d.aggregate(records)
    assert not result["complete"] and result["completed_calls"] == 4
    assert result["threshold_branches"] == dict(max_exceeds_025=True, mean_exceeds_0025=True)
    assert result["mean_abs"] == pytest.approx(0.1525)
    assert d.cached_selection(["fixed"], records) == ["fixed", "a", "z", "low"]


def test_tiny_native_generation_capture_uses_same_temperature_as_full_replay():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            )
        ),
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0, target_modules=["q_proj"]),
    )
    d.rl.set_mode(model, training=True)
    d.rl.set_mode(model, training=False)
    capture = d.rl.GenerationCapture(model)
    with torch.no_grad():
        capture.generate(
            input_ids=torch.tensor([[3, 4]]),
            attention_mask=torch.ones(1, 2),
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=0.5,
                top_p=1.0,
                top_k=0,
                max_new_tokens=3,
                use_cache=True,
                pad_token_id=0,
                eos_token_id=None,
            ),
        )
        record = dict(input_token_ids=[3, 4], output_token_ids=capture.tokens)
        replay = d.rl.action_logps(model, record).flatten().tolist()
        forced = d.cached_logps(model, record)
    assert max(abs(a - b) for a, b in zip(replay, capture.logps, strict=True)) < 1e-5
    assert forced == pytest.approx(replay, abs=1e-5)
    assert not any(parameter.requires_grad for parameter in model.parameters())
