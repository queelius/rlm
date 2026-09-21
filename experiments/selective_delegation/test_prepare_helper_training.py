"""Train-only helper supervision, teacher binding, immutable export and cap tests."""

import importlib.util
import json
from pathlib import Path

import pytest


def implementation():
    path = Path(__file__).with_name("prepare_helper_training.py")
    assert path.exists(), "helper preparation missing"
    spec = importlib.util.spec_from_file_location("helper_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(split="train"):
    return {
        "id": "host-" + split,
        "split": split,
        "question": "COMPOSED_SECRET",
        "answer": "FINAL_SECRET",
        "documents": [{"id": "p0", "title": "Public", "text": "SOURCE_TEXT", "support": "SECRET"}],
        "metadata": {
            "source_id": "SOURCE_ID_SECRET",
            "hops": 2,
            "question_decomposition": [
                {
                    "question": "Machine >> inventor",
                    "answer": "EARLIER_GOLD",
                    "id": 12,
                    "paragraph_support_idx": 0,
                },
                {
                    "question": "Where was #1 born?",
                    "answer": "CURRENT_GOLD",
                    "id": 13,
                    "paragraph_support_idx": 0,
                },
            ],
        },
    }


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False and kwargs["add_generation_prompt"]
        return [1] * 600

    def encode(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return [3, 4, 5]


def test_train_only_raw_question_teacher_binding_and_no_host_label_projection():
    m = implementation()
    rows = m.build_examples(
        [case(), case("validation"), case("transfer")], expected_parents=1, expected_steps=2
    )
    assert len(rows) == 2 and all(r["split"] == "train" for r in rows)
    assert rows[0]["step_index"] == 0 and rows[1]["step_index"] == 1
    assert "Machine >> inventor" in rows[0]["prompt"]
    assert "Where was EARLIER_GOLD born?" in rows[1]["prompt"]
    assert json.loads(rows[1]["target"]) == {"answer": "CURRENT_GOLD"}
    for row in rows:
        for secret in (
            "COMPOSED_SECRET",
            "FINAL_SECRET",
            "SOURCE_ID_SECRET",
            "paragraph_support_idx",
            "host-train",
            '"hops"',
            '"support"',
        ):
            assert secret not in row["prompt"] + row["target"]
        assert "SOURCE_TEXT" in row["prompt"]
    assert "CURRENT_GOLD" not in rows[1]["prompt"]
    assert "EARLIER_GOLD" not in rows[0]["prompt"]


def test_immutable_export_seals_sources_and_audits_real_target_eos(tmp_path):
    m = implementation()
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(case()) + "\n")
    output = tmp_path / "prepared"
    manifest = m.prepare(cases, output, tokenizer=Tokenizer(), expected_parents=1, expected_steps=2)
    assert manifest["examples"] == 2 and manifest["training_parents"] == 1
    assert manifest["token_audit"]["maximum_target_tokens_including_eos"] == 4
    assert manifest["token_audit"]["maximum_total_tokens"] == 604
    assert manifest["token_audit"]["exceed_total_6144"] == 0
    assert manifest["teacher_binding"] == "earlier annotated train answers only"
    assert (output / "source" / "prepare_helper_training.py").exists()
    assert (output / "MANIFEST.json").exists()
    for path, digest in manifest["artifact_sha256"].items():
        assert m.sha(output / path) == digest
    with pytest.raises(FileExistsError):
        m.prepare(cases, output, tokenizer=Tokenizer(), expected_parents=1, expected_steps=2)


def test_forward_reference_or_context_overflow_is_rejected_without_truncation(tmp_path):
    m = implementation()
    bad = case()
    bad["metadata"]["question_decomposition"][0]["question"] = "Who is #2?"
    with pytest.raises(ValueError, match="reference"):
        m.build_examples([bad], expected_parents=1, expected_steps=2)
    rows = m.build_examples([case()], expected_parents=1, expected_steps=2)

    class LongTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [1] * 6144

    with pytest.raises(ValueError, match="6144"):
        m.token_audit(rows, LongTokenizer())
