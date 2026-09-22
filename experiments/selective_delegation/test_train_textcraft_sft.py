import hashlib
import json

import pytest


class Tokenizer:
    eos_token_id = 9

    def apply_chat_template(self, *_args, **_kwargs):
        return [1, 2, 3]

    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]


def row(index):
    return {
        "task_id": f"textcraft.train.{index % 32}",
        "step": index // 32,
        "prompt": "You control a crafting inventory. Public state only.",
        "target": '{"action":"finish","message":"done"}',
    }


def test_public_rows_are_json_eos_target_only_and_schedule_is_366_rows():
    import train_textcraft_sft as trainer

    rows = [row(index) for index in range(366)]
    examples = trainer.tokenize_rows(rows, Tokenizer())
    assert examples[0]["input_ids"][-1] == ord("}")
    assert examples[0]["target_ids"][-1] == 9
    assert trainer.batch_sizes(len(examples)) == [16] * 22 + [14]
    assert trainer.validate_rows(rows, {"rows": 366, "eligible_task_count": 32}) == 32


def test_rejects_nonpublic_or_nonaction_target_rows():
    import train_textcraft_sft as trainer

    public = row(0)
    bad_prompt = {**public, "prompt": "gold trajectory goes here"}
    bad_target = {**public, "target": '{"action":"craft","hidden":true}'}
    rows = [row(index) for index in range(366)]
    with pytest.raises(ValueError, match="public"):
        trainer.validate_rows([bad_prompt, *rows[1:]], {"rows": 366, "eligible_task_count": 32})
    with pytest.raises(ValueError, match="strict"):
        trainer.validate_rows([bad_target, *rows[1:]], {"rows": 366, "eligible_task_count": 32})


def test_committed_checkpoint_resumes_remaining_cursor_and_rejects_corruption(tmp_path):
    import train_textcraft_sft as trainer

    checkpoint = tmp_path / "checkpoint-0008"
    checkpoint.mkdir()
    state = {"step": 8, "epoch": 0, "cursor": 128, "training_seconds": 1.0}
    path = checkpoint / "STATE.json"
    path.write_text(json.dumps(state))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (checkpoint / "COMMIT.json").write_text(json.dumps({"files": {"STATE.json": digest}}))
    restored, restored_state = trainer.committed_checkpoint(tmp_path, resume=True)
    assert restored == checkpoint
    assert restored_state["cursor"] == 128
    assert trainer.batch_sizes(366, start=restored_state["cursor"]) == [16] * 14 + [14]
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        trainer.committed_checkpoint(tmp_path, resume=True)
