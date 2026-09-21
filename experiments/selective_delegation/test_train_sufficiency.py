import json
from argparse import Namespace

import pytest


def test_native_qwen_lora_target_only_forward_backward():
    import torch
    import torch.nn.functional as functional
    import train_sufficiency as trainer
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
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
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0,
        ),
    )
    # Three prompt IDs, two answer IDs; the final target is emitted EOS=9.
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    targets = torch.tensor([[4, 5, 9]])
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=3).logits
    full = model(input_ids=ids, use_cache=False).logits
    actual = trainer.target_loss(logits, targets)
    expected = functional.cross_entropy(
        full[:, 2:, :].reshape(-1, 32), targets.reshape(-1), reduction="sum"
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.named_parameters()
        if "lora_" in name
    )
    assert all(p.grad is None for name, p in model.named_parameters() if "lora_" not in name)


def test_native_target_alignment_includes_eos_without_prompt_loss():
    import train_sufficiency as trainer

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["enable_thinking"] is False
            return [1, 2, 3]

        def encode(self, text, **kwargs):
            return [4, 5]

    example = dict(id="p-0", parent_id="p", split="train", prompt="public", target="target")
    row = trainer.tokenize_rows([example], Tokenizer())[0]
    assert row["input_ids"] == [1, 2, 3, 4, 5]
    assert row["target_ids"] == [4, 5, 99]
    assert row["prompt_tokens"] == 3
    example["split"] = "validation"
    with pytest.raises(ValueError, match="train"):
        trainer.tokenize_rows([example], Tokenizer())


def test_fixed_recipe_and_validation_of_both_prepared_arms():
    import train_sufficiency as trainer

    for arm in ("joint", "positive_only"):
        rows = []
        for parent in range(256):
            for slot in range(2):
                positive = arm == "positive_only" or slot == 0
                rows.append(
                    dict(
                        id=f"{parent}-{slot}",
                        parent_id=str(parent),
                        split="train",
                        prompt=trainer.sufficiency.INSTRUCTION + "public",
                        target=json.dumps(
                            {"answerable": positive, "answer": "answer" if positive else ""}
                        ),
                    )
                )
        manifest = dict(
            role="sufficiency",
            arm=arm,
            examples=512,
            training_parents=256,
            selected_parents=[str(i) for i in range(256)],
        )
        assert trainer.validate_input_role(rows, manifest, "sufficiency") == 256
        rows[0]["split"] = "test"
        with pytest.raises(ValueError):
            trainer.validate_input_role(rows, manifest, "sufficiency")
    args = Namespace(hours=0.5, learning_rate=1e-4)
    trainer.resolve_options(args)
    assert (args.epochs, args.seed, args.role) == (1, 2026092189, "sufficiency")
