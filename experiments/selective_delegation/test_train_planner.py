import pytest
import torch
from train_planner import epoch_order, target_loss, tokenize_rows


def test_target_loss_scores_every_target_once_with_global_denominator():
    logits = torch.tensor([[[3.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    targets = torch.tensor([[0, 1]])
    loss = target_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 2), targets.flatten(), reduction="sum"
    )
    assert torch.allclose(loss, expected)
    (loss / 4).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_epoch_order_is_complete_deterministic_and_changes_by_epoch():
    assert epoch_order(32, 10, 0) == epoch_order(32, 10, 0)
    assert sorted(epoch_order(32, 10, 0)) == list(range(32))
    assert epoch_order(32, 10, 0) != epoch_order(32, 10, 1)


def test_last_target_positions_start_at_final_prompt_token():
    class Tokenizer:
        eos_token_id = 22

        def apply_chat_template(self, messages, **kwargs):
            return [10, 11]

        def encode(self, text, **kwargs):
            return [20, 21]

    row = tokenize_rows([{"id": "a", "split": "train", "prompt": "p", "target": "t"}], Tokenizer())[
        0
    ]
    assert row["input_ids"] == [10, 11, 20, 21]
    assert row["target_ids"] == [20, 21, 22]
    assert row["input_ids"][-len(row["target_ids"])] == 11


def test_installed_qwen_selected_logits_match_full_causal_target_loss():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    ).eval()
    prefix_and_target = torch.tensor([[10, 11, 20, 21, 22]])
    target = torch.tensor([[20, 21, 22]])
    with torch.no_grad():
        full = model(input_ids=prefix_and_target, use_cache=False).logits[:, 1:4]
        selected = model(
            input_ids=prefix_and_target[:, :-1], use_cache=False, logits_to_keep=3
        ).logits
    assert torch.allclose(target_loss(full, target), target_loss(selected, target), atol=1e-6)


def test_predeclared_prompt_and_target_budgets_are_not_silently_relaxed():
    class Tokenizer:
        eos_token_id = 22

        def apply_chat_template(self, messages, **kwargs):
            return [10] * 513

        def encode(self, text, **kwargs):
            return [20]

    with pytest.raises(ValueError, match="prompt"):
        tokenize_rows([{"id": "a", "split": "train", "prompt": "p", "target": "t"}], Tokenizer())
