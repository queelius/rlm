import math

import pytest


def test_fresh_comparison_scores_actual_new_ids_not_old_bf16_ids():
    import diagnose_textcraft_precision as p
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
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()
    call = dict(input_token_ids=[3, 4], output_token_ids=[5, 6])
    with torch.no_grad():
        logits = model(torch.tensor([[3, 4, 5]]), use_cache=False).logits
        expected = torch.log_softmax(logits[0, -2:].float() / 0.5, -1)
        captured = [expected[0, 5].item(), expected[1, 6].item()]
        result = p.score_fresh(model, call, captured, old_ids=[7])
    assert result["output_token_ids"] == [5, 6]
    assert result["ids_equal_bf16"] is False
    assert result["own_cached_vs_full"]["max_abs"] == pytest.approx(0, abs=1e-6)
    assert result["full_probabilities"] == pytest.approx([math.exp(x) for x in captured])
    assert result["all_finite"]
