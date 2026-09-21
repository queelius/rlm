import hashlib
import json

import train_alfworld_sft as trainer


class Tokenizer:
    eos_token_id = 9

    def apply_chat_template(self, *_args, **_kwargs):
        return [1, 2, 3]

    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]


def test_public_target_json_eos_shift_and_exact_update_schedule():
    rows = [
        {
            "id": str(index),
            "prompt": "public_context",
            "target_json": json.dumps({"action_index": index}),
            "target_with_eos": json.dumps({"action_index": index}) + "<eos>",
        }
        for index in range(524)
    ]
    examples = trainer.tokenize_rows(rows, Tokenizer())
    assert examples[0]["input_ids"][-1] == ord("}")
    assert examples[0]["target_ids"][-1] == 9
    assert trainer.batch_sizes(len(examples)) == [16] * 32 + [12]
    assert trainer.validate_rows(rows, {"examples": 524, "successful_games_only": True}) == 524


def test_tiny_lora_target_loss_backpropagates_only_adapter_parameters():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    base = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=1))
    model = get_peft_model(
        base,
        LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"),
        autocast_adapter_dtype=True,
    )
    ids, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3]])
    logits = model(input_ids=ids, logits_to_keep=targets.shape[1]).logits
    trainer.target_loss(logits, targets).backward()
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable and all(
        "lora_" in name and parameter.dtype == torch.float32 for name, parameter in trainable
    )
    assert any(parameter.grad is not None for _, parameter in trainable)


def test_completed_checkpoint_is_not_replayed_and_commit_is_verified(tmp_path):
    checkpoint = tmp_path / "checkpoint-0033"
    checkpoint.mkdir()
    state = {"step": 33, "epoch": 1, "cursor": 0, "training_seconds": 1.0}
    state_path = checkpoint / "STATE.json"
    state_path.write_text(json.dumps(state))
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    (checkpoint / "COMMIT.json").write_text(json.dumps({"files": {"STATE.json": digest}}))
    restored, restored_state = trainer.committed_checkpoint(tmp_path, resume=True)
    assert restored == checkpoint
    assert trainer.completed_epoch(restored_state)
    state_path.write_text("corrupted")
    try:
        trainer.committed_checkpoint(tmp_path, resume=True)
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("corrupt committed checkpoint was accepted")


def test_parameter_delta_detects_a_real_first_update():
    import torch

    before, after = torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.25])
    assert trainer.parameter_l1_delta([after], [before]) == 0.25
