import copy
import json

import pytest


def test_official_pair_dependence_missing_and_parent_averaging():
    import analyze_sufficiency as analysis

    cases, records = [], {}
    for parent in ("p", "q"):
        for label in (True, False):
            case = {
                "id": parent + str(label),
                "parent_id": parent,
                "answerable": label,
                "answer": "London",
                "answer_aliases": [],
            }
            cases.append(case)
            for seed in analysis.baseline.SEEDS:
                records[case["id"], seed] = {
                    "status": "valid",
                    "prediction": {"answerable": label, "answer": "London" if label else ""},
                }
    # One negative overanswer must zero that complete pair despite a correct positive.
    records["pFalse", analysis.baseline.SEEDS[0]]["prediction"] = {
        "answerable": True,
        "answer": "London",
    }
    # Missing negative is unknown, never counted as an observed correct abstention.
    records["qFalse", analysis.baseline.SEEDS[1]] = {"status": "missing", "prediction": None}
    rows = analysis.pair_metrics(cases, records)
    assert [r["joint_em"] for r in rows] == [0, 1, 1, 0]
    assert sum(r["negative_overanswer"] for r in rows) == 1
    assert sum(r["both_observed"] for r in rows) == 3
    assert all(r["positive_em"] == 1 for r in rows)
    means = analysis.parent_values(rows, "joint_em")
    assert means == {"p": 0.5, "q": 0.5}
    interval = analysis.interval(means, draws=100, seed=7)
    assert interval == {"estimate": 0.5, "ci95": [0.5, 0.5]}


def test_native_receipt_adapter_and_response_binding():
    import analyze_sufficiency as analysis

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return '{"answerable":true,"answer":"London"}'

    case = {"id": "p", "public": {"question": "Where?", "documents": []}}
    job = {"case_id": "p", "episode_id": "p-1", "seed": 1}
    adapter = {"files": {"adapter_model.safetensors": "adapter-hash"}, "step": 32}
    plan = {"model": "base", "adapters": adapter}
    req = {
        "prompt": analysis.baseline.prompt(case),
        "seed": 1,
        "input_token_ids": [1, 2],
        "model": "base",
        "condition": "joint",
        "role": "sufficiency",
        "adapter_enabled": True,
        "adapter": adapter,
        "adapter_sha256": "adapter-hash",
        "sampling": analysis.SAMPLING,
    }
    call = {
        "call_id": "p-1",
        "request": req,
        "condition": "joint",
        "role": "sufficiency",
        "request_digest": analysis.baseline.native.probe.runtime.digest(req),
        "available": True,
        "input_token_ids": [1, 2],
        "output_token_ids": [3, 4],
        "text": Tokenizer().decode([]),
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }
    assert analysis.audit_call(call, job, case, plan, "joint", Tokenizer())["status"] == "valid"
    changed = copy.deepcopy(call)
    changed["request"]["adapter_enabled"] = False
    changed["request_digest"] = analysis.baseline.native.probe.runtime.digest(changed["request"])
    with pytest.raises(ValueError, match="adapter"):
        analysis.audit_call(changed, job, case, plan, "joint", Tokenizer())
    changed = copy.deepcopy(call)
    changed["text"] = json.dumps({"answerable": False, "answer": ""})
    with pytest.raises(ValueError, match="decode"):
        analysis.audit_call(changed, job, case, plan, "joint", Tokenizer())
    changed = copy.deepcopy(call)
    changed.update(available=False, text=None)
    del changed["output_token_ids"]
    assert (
        analysis.audit_call(changed, job, case, plan, "joint", Tokenizer())["status"]
        == "unavailable"
    )
