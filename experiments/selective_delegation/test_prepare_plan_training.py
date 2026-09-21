"""Planner supervision stays separate from host labels and evaluation parents."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("prepare_plan_training.py")
    assert path.exists(), "planner preparation implementation is missing"
    spec = importlib.util.spec_from_file_location("planner_preparation", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def case(identity="parent", split="train"):
    return {
        "id": identity,
        "split": split,
        "question": "What country is its inventor from?",
        "answer": "GOLD_ANSWER",
        "source_id": "HOST_SOURCE",
        "secret": "HOST_SECRET",
        "documents": [
            {
                "id": "p7",
                "title": "Machine",
                "text": "BODY_SECRET",
                "is_supporting": True,
                "secret": "DOC_SECRET",
            }
        ],
        "metadata": {
            "source_id": "HOP_SOURCE_SECRET",
            "hops": 2,
            "supporting_paragraph_ids": ["SUPPORT_SECRET"],
            "question_decomposition": [
                {"question": "Who invented it?", "answer": "STEP_ANSWER", "id": 91},
                {
                    "question": "What country is #1 from?",
                    "answer": "GOLD_ANSWER",
                    "paragraph_support_idx": 17,
                },
            ],
        },
    }


def test_prompt_allowlist_and_question_only_privileged_target():
    m = module()
    original = case()
    prompt = m.planner_prompt(original)
    target = m.reference_target(original)
    assert "Machine" in prompt and "p7" in prompt and original["question"] in prompt
    for forbidden in (
        "BODY_SECRET",
        "GOLD_ANSWER",
        "HOST_SOURCE",
        "HOST_SECRET",
        "DOC_SECRET",
        "HOP_SOURCE_SECRET",
        "SUPPORT_SECRET",
        "STEP_ANSWER",
    ):
        assert forbidden not in prompt + target
    assert json.loads(target) == {"subquestions": ["Who invented it?", "What country is #1 from?"]}
    assert "answer" not in json.loads(target)
    changed = json.loads(json.dumps(original))
    changed["metadata"]["question_decomposition"][0]["answer"] = "NEW_SECRET"
    changed["documents"][0]["text"] = "NEW_TEXT_SECRET"
    assert m.planner_prompt(changed) == prompt
    assert m.reference_target(changed) == target


def test_strict_plan_contract_allows_variable_length_and_rejects_extra_fields():
    m = module()
    for length in (1, 2, 3, 8):
        value = {"subquestions": ["Question?"] * length}
        assert m.parse_plan(json.dumps(value)) == value
    for text in (
        '{"subquestions": []}',
        '{"subquestions": [""]}',
        '{"subquestions": ["  "]}',
        '{"subquestions": [1]}',
        '{"subquestions": ["Q?"], "answer": "secret"}',
        '{"subquestions": ["Q?"], "subquestions": ["Other?"]}',
        '```json\n{"subquestions": ["Q?"]}\n```',
        json.dumps({"subquestions": ["Q?"] * 9}),
    ):
        with pytest.raises(ValueError):
            m.parse_plan(text)


def test_training_order_is_deterministic_and_never_exports_heldout_targets():
    m = module()
    train = [case(str(index)) for index in range(5)]
    held = case("held", "validation")
    transfer = case("transfer", "transfer")
    held["metadata"] = transfer["metadata"] = {"question_decomposition": None}
    examples = m.build_examples(train + [held, transfer], expected_count=5)
    reverse = m.build_examples(list(reversed(train)) + [transfer, held], expected_count=5)
    assert examples == reverse
    assert {row["id"] for row in examples} == {"0", "1", "2", "3", "4"}
    assert all(set(row) == {"id", "split", "prompt", "target"} for row in examples)
    with pytest.raises(ValueError, match="train"):
        m.reference_target(held)
    with pytest.raises(ValueError, match="count"):
        m.build_examples(train, expected_count=256)


class Tokenizer:
    """Deliberate token-length fixture; actual tokenizer audited separately on CPU."""

    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return [1] * 1800

    def encode(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return [2] * 300


def test_token_audit_reports_overflow_without_truncation():
    m = module()
    examples = m.build_examples([case()], expected_count=1)
    audit = m.token_audit(examples, Tokenizer())
    assert audit["exceed_prompt_512"] == 1
    assert audit["exceed_target_256"] == 1
    assert audit["exceed_total_2048"] == 1
    assert audit["rows"][0]["total_tokens"] == 2101
    assert not audit["truncation_used"]


def test_preparation_writes_immutable_train_only_examples(tmp_path):
    m = module()
    inputs = tmp_path / "cases.jsonl"
    inputs.write_text(json.dumps(case()) + "\n" + json.dumps(case("held", "validation")) + "\n")
    output = tmp_path / "prepared"
    manifest = m.prepare(inputs, output, tokenizer=Tokenizer(), expected_count=1)
    assert manifest["supervision"]["annotation_privileged"]
    assert manifest["counts"] == {"train": 1, "validation": 1}
    assert len((output / "examples.jsonl").read_text().splitlines()) == 1
    original = (output / "MANIFEST.json").read_bytes()
    with pytest.raises(FileExistsError):
        m.prepare(inputs, output, tokenizer=Tokenizer(), expected_count=1)
    assert (output / "MANIFEST.json").read_bytes() == original
