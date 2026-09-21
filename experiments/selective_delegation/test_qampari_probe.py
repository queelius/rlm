import json
import time
from pathlib import Path

import pytest


def fixture_case():
    return {
        "id": "opaque",
        "source_id": "SECRET",
        "question_type": "SECRET",
        "answer_list": [{"answer_text": "Alpha", "aliases": ["Alpha"]}],
        "public": {
            "question": "Public question?",
            "documents": [
                {"docid": f"d{i}", "title": "Title", "text": f"Passage{i}"} for i in range(200)
            ],
        },
    }


def test_requests_preserve_exact_evidence_budget_and_hide_gold():
    import qampari_probe as q

    case = fixture_case()
    direct = q.requests(case, {"episode_id": "d", "condition": "direct200", "seed": 7})
    maps = q.requests(case, {"episode_id": "m", "condition": "map50", "seed": 7})
    assert len(direct) == 1 and len(maps) == 4
    assert sum(r["cap"] for r in direct) == sum(r["cap"] for r in maps) == 1024
    assert [r["seed"] for r in maps] == [7, 8, 9, 10]
    docs = [json.loads(r["prompt"].split("\n", 1)[1])["documents"] for r in maps]
    assert sum(docs, []) == case["public"]["documents"]
    assert all(len(d) == 50 for d in docs)
    assert all("SECRET" not in r["prompt"] and "Alpha" not in r["prompt"] for r in direct + maps)


def test_strict_parser_official_nonempty_metric_and_explicit_empty_zero():
    import qampari_probe as q

    for text in [
        '{"answers":[1]}',
        '{"answers":[],"gold":1}',
        '{"answers":[],"answers":["x"]}',
        '```{"answers":[]}```',
    ]:
        with pytest.raises(ValueError):
            q.parse_answers(text)
    assert q.parse_answers('{"answers":[]}') == []
    gold = fixture_case()["answer_list"]
    assert q.grade([], gold) == {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "rec_above": 0,
        "f1_above": 0,
    }
    result = q.grade(["Alpha", "alpha", "Alpha"], gold)
    assert result["precision"] == 0.5 and result["recall"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_invalid_map_block_is_whole_policy_zero_missing_is_unknown(tmp_path):
    import qampari_probe as q

    class Client:
        def __init__(self, missing=False):
            self.calls = 0
            self.missing = missing

        def call(self, request):
            self.calls += 1
            return {
                "call_id": request["call_id"],
                "available": not self.missing,
                "text": '{"answers":["Alpha"]}' if self.calls != 2 else "invalid",
                "request_digest": str(self.calls),
            }

    job = {"episode_id": "test", "condition": "map50", "seed": 7, "case_id": "opaque"}
    client = Client()
    result = q.execute(client, fixture_case(), job, tmp_path)
    assert client.calls == 4
    assert result["status"] == "protocol_invalid" and result["observed"]
    assert result["answers"] is None and result["metrics"]["f1"] == 0
    result = q.execute(
        Client(missing=True), fixture_case(), {**job, "episode_id": "missing"}, tmp_path
    )
    assert result["status"] == "inference_unavailable" and not result["observed"]
    assert len(result["call_ids"]) == 1


def test_real_cpu_native_call_records_frozen_roles_caps_tokens_and_no_adapter(tmp_path):
    import qampari_probe as q
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=4,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            eos_token_id=3,
            pad_token_id=0,
        )
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class Tokenizer:
        eos_token_id, pad_token_id = 3, 0

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return str(ids)

    client = q.NativeClient(model, Tokenizer(), tmp_path, time.time() + 90)
    result = client.call(
        {
            "call_id": "tiny",
            "condition": "map50",
            "role": "map",
            "prompt": "public",
            "seed": 2,
            "cap": 256,
        }
    )
    assert result["available"]
    assert result["request"]["adapter_enabled"] is False
    assert result["request"]["sampling"]["max_new_tokens"] == 256
    assert result["usage"]["prompt_tokens"] == 2
    assert result["usage"]["completion_tokens"] == len(result["output_token_ids"])
    assert not any(p.requires_grad for p in model.parameters())
    assert (tmp_path / "calls/tiny.json").is_file()


def test_actual_frozen_public_fixture_contains_200_ranked_documents():
    import qampari_probe as q

    path = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/") / (
        "selective-delegation-20260921/qampari-inputs-001/cases.jsonl"
    )
    case = json.loads(path.read_text().splitlines()[0])
    assert case["public"]["question"] == "What companies are producing modern armament in Pakistan?"
    reqs = q.requests(case, {"episode_id": "saved", "condition": "map50", "seed": 2026092187})
    assert len(reqs) == 4
    assert case["source_id"] not in "".join(r["prompt"] for r in reqs)


def test_summary_preserves_unobserved_slots_and_actual_native_cost(tmp_path):
    import qampari_probe as q

    jobs = [
        {"episode_id": condition, "condition": condition, "case_id": "opaque", "repeat": 0}
        for condition in ("direct200", "map50")
    ]
    plan = {"jobs": jobs, "planned_calls": 5, "planned_episodes": 2}
    q.save(tmp_path / "PLAN.json", plan)
    q.save(
        tmp_path / "episodes/direct200.json",
        {
            **jobs[0],
            "question_type": "simple",
            "observed": True,
            "status": "protocol_invalid",
            "metrics": {"f1": 0, "precision": 0, "recall": 0, "rec_above": 0, "f1_above": 0},
        },
    )
    q.save(
        tmp_path / "calls/direct200.json",
        {
            "call_id": "direct200",
            "condition": "direct200",
            "available": True,
            "started": 1,
            "ended": 4,
            "usage": {"prompt_tokens": 30000, "completion_tokens": 5},
        },
    )
    report = q.summarize(tmp_path, plan)
    assert report["conditions"]["direct200"]["returned_protocol_invalid"] == 1
    assert report["conditions"]["map50"]["missing_or_unavailable"] == 1
    assert report["conditions"]["map50"]["returned_protocol_invalid"] == 0
    assert report["paired_observed"] == []
    assert report["physical_cost"]["calls"] == 1
    assert report["physical_cost"]["total_tokens"] == 30005
    assert report["physical_cost"]["native_call_seconds"] == 3
