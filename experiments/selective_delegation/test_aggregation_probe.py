"""Contracts for a frozen-trace final evidence ablation."""

import importlib.util
import json
from pathlib import Path

import eval_planner as evaluation
import pytest
from test_plan_probe import case_and_state


def implementation():
    path = Path(__file__).with_name("aggregation_probe.py")
    assert path.exists(), "aggregation probe implementation missing"
    spec = importlib.util.spec_from_file_location("aggregation_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conditions_change_only_documents_and_never_project_host_fields():
    mod = implementation()
    case, _ = case_and_state()
    plan = {"subquestions": ["Who made it?", "When did #1 end?"]}
    report = {"execution": "isolated", "steps": [{"answer": "actual prediction"}]}
    full = mod.final_prompt(case, plan, report, "full_source")
    trace = mod.final_prompt(case, plan, report, "trace_only")
    prefix, payload = full.rsplit("\n", 1)
    trace_prefix, trace_payload = trace.rsplit("\n", 1)
    body = json.loads(payload)
    del body["documents"]
    assert prefix == trace_prefix
    assert body == json.loads(trace_payload)
    assert body["helper_report"] == report
    assert body["model_plan"] == plan
    assert all(
        s not in full + trace
        for s in (
            "SECRET_GOLD",
            "SECRET_ALIAS",
            "SECRET_SOURCE",
            "SECRET_STEP_ONE",
            "initial_attempt",
        )
    )


def test_actual_helper_binding_and_tampered_trace_rejected(tmp_path):
    mod = implementation()
    case, _ = case_and_state()
    plan = {"subquestions": ["Who made it?", "When did #1 end?"]}
    trace = []
    records = []
    for i, (question, resolved, answer) in enumerate(
        [
            (plan["subquestions"][0], "Who made it?", "ACTUAL_COMPANY"),
            (plan["subquestions"][1], "When did ACTUAL_COMPANY end?", "1954"),
        ]
    ):
        trace.append(
            {"step": i + 1, "question": question, "resolved_question": resolved, "answer": answer}
        )
        records.append(
            {
                "request": {"prompt": evaluation.isolated_helper_prompt(case, resolved)},
                "text": json.dumps({"answer": answer}),
            }
        )
    assert mod.reconstruct_trace(case, plan, records) == trace
    records[1]["request"]["prompt"] = evaluation.isolated_helper_prompt(case, "When did GOLD end?")
    with pytest.raises(ValueError, match="helper prompt"):
        mod.reconstruct_trace(case, plan, records)


def test_invalid_source_zero_has_no_calls_and_all_planned_denominator(tmp_path):
    mod = implementation()
    case, _ = case_and_state()

    class NeverCall:
        output = tmp_path

        def call(self, *args, **kwargs):
            raise AssertionError("invalid source must not call model")

    job = {
        "episode": {
            "episode_id": "e",
            "case_id": case["id"],
            "candidate": 0,
            "status": "invalid_helper",
        },
        "eligible": False,
        "records": [],
        "source_hashes": {},
        "source_request_digests": {},
    }
    row = mod.collect(NeverCall(), job, case, "trace_only", 0, 123)
    assert row["score"]["correct"] is False
    assert row["new_physical_cost"]["calls"] == 0
    summary = mod.summarize(tmp_path, {"planned_candidates": 64, "eligible_candidates": 0})
    assert summary["conditions"]["trace_only"]["planned_denominator"] == 128
    assert summary["conditions"]["trace_only"]["explicit_source_zeros"] == 1
    assert summary["conditions"]["trace_only"]["pending"] == 127


def test_new_calls_are_paired_and_old_final_not_in_deployed_cost(tmp_path):
    mod = implementation()
    case, _ = case_and_state()
    seen = []

    class Client:
        output = tmp_path

        def call(self, identity, prompt, seed, cap, temperature):
            seen.append((seed, cap, temperature))
            return {
                "call_id": identity,
                "available": True,
                "text": '{"answer":"1954"}',
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            }

    root = {
        "call_id": "r",
        "role": "root",
        "available": True,
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    old = {
        "call_id": "f",
        "role": "final",
        "available": True,
        "usage": {"prompt_tokens": 999, "completion_tokens": 999},
    }
    job = {
        "episode": {"episode_id": "e", "case_id": case["id"], "candidate": 0, "status": "scored"},
        "eligible": True,
        "plan": {"subquestions": ["Who?"]},
        "report": {"steps": []},
        "records": [root, old],
        "source_hashes": {},
        "source_request_digests": {},
    }
    rows = [mod.collect(Client(), job, case, c, 0, 123) for c in mod.CONDITIONS]
    assert seen == [(123, 128, 0.5), (123, 128, 0.5)]
    assert all(r["new_physical_cost"]["calls"] == 1 for r in rows)
    assert all(r["hypothetical_deployed_cost"]["prompt_tokens"] == 17 for r in rows)


def test_repeat_noise_not_mislabeled_between_plan_variation():
    mod = implementation()
    rows = []
    for candidate, rewards in enumerate(((0, 1), (0, 1), (0, 1), (0, 1))):
        for repeat, reward in enumerate(rewards):
            rows.append(
                {
                    "case_id": "p",
                    "candidate": candidate,
                    "repeat": repeat,
                    "condition": "full_source",
                    "eligible": True,
                    "available": True,
                    "score": {"correct": bool(reward)},
                }
            )
    stats = mod.variation(rows, "full_source")[0]
    assert stats["between_candidate_mean_variance"] == 0
    assert stats["within_candidate_repeat_variance"] == 0.25
    assert stats["repeat_disagreement_candidates"] == 4
