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


def test_helper_caps_allow_full_source_without_relaxing_planner_or_target_alignment():
    class Tokenizer:
        eos_token_id = 22

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "PUBLIC_PROMPT"}]
            return [10] * 4819

        def encode(self, text, **kwargs):
            assert text == '{"answer":"target"}'
            return [20, 21]

    rows = [
        {
            "id": "a-step01",
            "parent_id": "a",
            "split": "train",
            "prompt": "PUBLIC_PROMPT",
            "target": '{"answer":"target"}',
            "host_secret": "SECRET",
        }
    ]
    result = tokenize_rows(rows, Tokenizer(), role="helper")[0]
    assert len(result["input_ids"]) == 4821
    assert result["target_ids"] == [20, 21, 22]
    assert result["input_ids"][-3] == 10 and result["parent_id"] == "a"
    with pytest.raises(ValueError, match="prompt"):
        tokenize_rows(rows, Tokenizer())

    class LongTarget(Tokenizer):
        def encode(self, text, **kwargs):
            return [20] * 48

    with pytest.raises(ValueError, match="target"):
        tokenize_rows(rows, LongTarget(), role="helper")

    class LongPrompt(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [10] * 6144

    with pytest.raises(ValueError, match="context"):
        tokenize_rows(rows, LongPrompt(), role="helper")


def test_helper_manifest_role_parent_vs_step_inventory_and_fresh_fixed_dose():
    from argparse import Namespace

    from train_planner import resolve_options, validate_input_role

    rows = [
        {
            "id": f"p{parent}-s{step}",
            "parent_id": f"p{parent}",
            "step_index": step,
            "split": "train",
            "prompt": "p",
            "target": '{"answer":"a"}',
        }
        for parent in range(256)
        for step in range(2 if parent < 198 else 3)
    ]
    manifest = {"role": "helper", "examples": 570, "training_parents": 256}
    assert validate_input_role(rows, manifest, "helper") == 256
    with pytest.raises(ValueError, match="role"):
        validate_input_role(rows, manifest, "planner")
    with pytest.raises(ValueError, match="570"):
        validate_input_role(rows[:-1], manifest, "helper")
    wrong = [dict(row) for row in rows]
    wrong[0]["split"] = "validation"
    with pytest.raises(ValueError, match="train"):
        validate_input_role(wrong, manifest, "helper")
    args = Namespace(role="helper", epochs=None, seed=None, hours=None)
    resolve_options(args)
    assert (args.epochs, args.seed, args.hours) == (1, 2026092111, 0.75)
    old = Namespace(role="planner", epochs=None, seed=None, hours=None)
    resolve_options(old)
    assert (old.epochs, old.seed, old.hours) == (3, 20260921, 2)
    with pytest.raises(ValueError, match="one epoch"):
        resolve_options(Namespace(role="helper", epochs=3, seed=None, hours=None))
