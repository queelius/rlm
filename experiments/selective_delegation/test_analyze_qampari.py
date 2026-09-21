import copy
import json

import pytest


def gold():
    return [{"answer_text": "Alpha", "aliases": ["Alpha"]}]


def test_native_regrade_zeroes_whole_invalid_map_but_marks_missing_unobserved():
    import analyze_qampari as a

    calls = [{"available": True, "text": '{"answers":["Alpha"]}'} for _ in range(4)]
    result = a.score_native(calls, 4, gold())
    assert result["metrics"]["f1"] == 1
    assert result["answers"] == ["Alpha"]
    calls[1]["text"] = "not JSON"
    result = a.score_native(calls, 4, gold())
    assert result["observed"] and not result["valid"]
    assert result["metrics"]["f1"] == 0 and result["answers"] is None
    calls[1] = None
    result = a.score_native(calls, 4, gold())
    assert not result["observed"] and result["status"] == "missing_or_unavailable"


def test_official_regrade_retains_raw_alias_duplicate_penalty_and_empty_zero():
    import analyze_qampari as a

    row = a.score_native([{"available": True, "text": '{"answers":["Alpha","alpha"]}'}], 1, gold())
    assert row["metrics"]["precision"] == 0.5
    assert row["metrics"]["f1"] == pytest.approx(2 / 3)
    assert (
        a.score_native([{"available": True, "text": '{"answers":[]}'}], 1, gold())["metrics"]["f1"]
        == 0
    )


def test_bootstrap_keeps_two_repeats_within_parent():
    import analyze_qampari as a

    result = a.parent_interval({"a": [1, -1], "b": [0, 0]}, draws=100, seed=3)
    assert result == {"estimate": 0, "ci95": [0, 0], "parents": 2, "repeats": 2}


def test_literal_alias_coverage_counts_blocks_without_asserting_support():
    import analyze_qampari as a

    docs = [{"title": "", "text": "unrelated"} for _ in range(200)]
    docs[0]["text"] = "ALPHA is explicitly not an answer to this question."
    docs[51]["title"] = "Alpha"
    docs[100]["text"], docs[101]["text"] = "New", "York"
    case = {
        "id": "p",
        "question_type": "composition",
        "public": {"documents": docs},
        "answer_list": gold() + [{"answer_text": "New York", "aliases": ["New York"]}],
    }
    result = a.literal_coverage([case])
    assert result["total_literal_covered_entities"] == 1
    assert result["rows"][0]["covered_entities_by_block"] == [1, 1, 0, 0]
    assert "NOT proof" in result["warning"]


def test_audit_rejects_changed_prompt_seed_cap_and_decoding():
    import analyze_qampari as a
    import qampari_probe as q

    spec = {
        "call_id": "c",
        "condition": "map50",
        "role": "map",
        "seed": 8,
        "cap": 256,
        "prompt": "public",
    }
    plan = {"model": "model", "model_manifest_sha256": "modelhash"}
    request = {
        "prompt": "public",
        "input_token_ids": [1, 2],
        "model": "model",
        "model_manifest_sha256": "modelhash",
        "adapter_enabled": False,
        "adapter_sha256": None,
        "condition": "map50",
        "role": "map",
        "seed": 8,
        "context_limit": 40960,
        "truncation": False,
        "sampling": {
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 256,
            "max_time": 90.0,
            "do_sample": True,
        },
    }
    row = {
        "call_id": "c",
        "request": request,
        "request_digest": q.probe.runtime.digest(request),
        "plan_sha256": "planhash",
        "condition": "map50",
        "role": "map",
        "available": True,
        "input_token_ids": [1, 2],
        "output_token_ids": [3],
        "text": '{"answers":[]}',
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        "started": 1,
        "ended": 2,
        "finish_reason": "eos",
        "effective_max_time": 90,
    }

    class Tokenizer:
        eos_token_id = 3

        def decode(self, *args, **kwargs):
            return '{"answers":[]}'

    assert a.audit_call(row, spec, plan, "planhash", [1, 2], Tokenizer())["stop"] == "eos"
    for field, value in [("prompt", "gold"), ("seed", 9)]:
        wrong = copy.deepcopy(row)
        wrong["request"][field] = value
        wrong["request_digest"] = q.probe.runtime.digest(wrong["request"])
        with pytest.raises(ValueError):
            a.audit_call(wrong, spec, plan, "planhash", [1, 2], Tokenizer())
    wrong = copy.deepcopy(row)
    wrong["request"]["sampling"]["max_new_tokens"] = 128
    wrong["request_digest"] = q.probe.runtime.digest(wrong["request"])
    with pytest.raises(ValueError, match="contract"):
        a.audit_call(wrong, spec, plan, "planhash", [1, 2], Tokenizer())
    wrong = copy.deepcopy(row)
    wrong["text"] = json.dumps({"answers": ["changed"]})
    with pytest.raises(ValueError, match="decode"):
        a.audit_call(wrong, spec, plan, "planhash", [1, 2], Tokenizer())
