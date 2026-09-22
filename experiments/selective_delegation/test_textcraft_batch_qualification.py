import importlib
import json

import pytest


def test_fixed_length_strata_ignore_outputs_and_preserve_native_prompt_ids():
    q = importlib.import_module("textcraft_batch_qualification")
    rows = [
        dict(
            call_id=str(i),
            request={"input_token_ids": list(range(i + 1)), "seed": i, "cap": 64},
            available=False,
            text="wrong",
        )
        for i in range(40)
    ]
    selected = q.select(rows)
    assert len(selected) == 32
    assert sum(len(r["request"]["input_token_ids"]) <= 20 for r in selected) == 16
    changed = [dict(r, text="correct", available=True) for r in reversed(rows)]
    assert [r["call_id"] for r in q.select(changed)] == [r["call_id"] for r in selected]
    body = q.request_body(selected[0])
    assert body["prompt"] == selected[0]["request"]["input_token_ids"]
    assert body["return_token_ids"] and body["temperature"] == 0.5
    assert body["seed"] == selected[0]["request"]["seed"]
    assert body["max_tokens"] == 64 and body["top_k"] == -1


def test_missing_or_mismatched_native_echo_is_not_silently_accepted():
    q = importlib.import_module("textcraft_batch_qualification")

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return '{"action":"finish","message":"done"}'

    row = dict(request={"input_token_ids": [1, 2], "cap": 8}, output_token_ids=[3, 4])
    response = dict(
        id="one",
        model=q.ALIAS,
        choices=[dict(prompt_token_ids=[1, 2], token_ids=[3, 4], finish_reason="stop")],
        usage=dict(prompt_tokens=2, completion_tokens=2),
    )
    result = q.normalize(row, response, Tokenizer())
    assert result["valid_action"] and result["hf_output_ids_equal"]
    response["choices"][0]["prompt_token_ids"] = [9]
    with pytest.raises(ValueError, match="input"):
        q.normalize(row, response, Tokenizer())


def test_actual_saved057_requests_freeze_with_embedded_tokenizer_template(tmp_path):
    q = importlib.import_module("textcraft_batch_qualification")
    q.freeze(tmp_path / "inputs")
    plan = json.loads((tmp_path / "inputs/PLAN.json").read_text())
    rows = json.loads((tmp_path / "inputs/REQUESTS.json").read_text())
    warmup = json.loads((tmp_path / "inputs/WARMUP.json").read_text())
    assert len(rows) == 32 and plan["max_model_calls"] == 69
    assert len(warmup) == 5
    assert not {row["call_id"] for row in rows} & {row["call_id"] for row in warmup}
    assert len(plan["template_sha256"]) == 64
    assert all(q.request_body(row)["prompt"] == row["input_token_ids"] for row in rows)
