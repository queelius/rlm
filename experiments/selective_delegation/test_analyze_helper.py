"""Focused saved-receipt tests; no model loading."""

import importlib.util
import json
from pathlib import Path

import eval_helper
import eval_planner
import probe
import pytest


def implementation():
    path = Path(__file__).with_name("analyze_helper.py")
    assert path.exists(), "helper analyzer missing"
    spec = importlib.util.spec_from_file_location("helper_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    output, cases_path = tmp_path / "run", tmp_path / "cases.jsonl"
    cases = [
        {
            "id": p,
            "question": "Where?",
            "documents": [],
            "answer": "Paris",
            "metadata": {"answer_aliases": ["City of Paris"], "component_ids": ["shared"]},
        }
        for p in ("a", "b")
    ]
    cases_path.write_text("\n".join(json.dumps(c) for c in cases))
    save(
        output / "PLAN.json",
        {
            "schema": "matched-frozen-root-helper-evaluation-v1",
            "case_ids": ["a", "b"],
            "repeats": 1,
            "conditions": list(eval_helper.CONDITIONS),
            "cases_sha256": probe.campaign.sha(cases_path),
            "helper_adapter_binding": {"adapter_model.safetensors": "helper-hash"},
            "root_adapter_binding": {"adapter_model.safetensors": "root-hash"},
            "format_reminder": eval_helper.FORMAT_REMINDER,
        },
    )
    plan = {"subquestions": ["Who?", "Where is #1?"]}
    for case in cases:
        for condition in eval_helper.CONDITIONS:
            if case["id"] == "b" and condition == "format_reminder":
                continue
            eid = f"{case['id']}-r0-{condition}"
            trace, ids = [], []
            invalid = case["id"] == "a" and condition == "base_helper"
            for index, question in enumerate(plan["subquestions"]):
                resolved = eval_planner.bind_question(question, [s["answer"] for s in trace])
                answer = "Alice" if condition == "base_helper" else "Bob"
                text = "not JSON" if invalid else json.dumps({"answer": answer})
                cid = eid + f"-helper-{index + 1}"
                request = {"prompt": eval_helper.helper_prompt(case, resolved, condition)}
                save(
                    output / "calls" / (cid + ".json"),
                    {
                        "call_id": cid,
                        "condition": condition,
                        "role": "helper",
                        "request": request,
                        "request_digest": probe.runtime.digest(request),
                        "available": True,
                        "text": text,
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                        "started": 10,
                        "ended": 12,
                    },
                )
                ids.append(cid)
                if invalid:
                    break
                trace.append(
                    {
                        "step": index + 1,
                        "question": question,
                        "resolved_question": resolved,
                        "answer": answer,
                    }
                )
            if not invalid:
                cid = eid + "-final"
                request = {
                    "prompt": eval_planner.final_prompt(
                        case, plan, {"execution": "isolated", "steps": trace}
                    )
                }
                save(
                    output / "calls" / (cid + ".json"),
                    {
                        "call_id": cid,
                        "condition": condition,
                        "role": "final",
                        "request": request,
                        "request_digest": probe.runtime.digest(request),
                        "available": True,
                        "text": json.dumps(
                            {"answer": "Rome" if condition == "base_helper" else "City of Paris"}
                        ),
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                        "started": 12,
                        "ended": 15,
                    },
                )
                ids.append(cid)
            save(
                output / "episodes" / (eid + ".json"),
                {
                    "episode_id": eid,
                    "case_id": case["id"],
                    "repeat": 0,
                    "condition": condition,
                    "seed": 100,
                    "plan": plan,
                    "plan_valid": True,
                    "reused_root_call_id": case["id"] + "-old-root",
                    "root_request_digest": "root",
                    "source_episode_id": case["id"] + "-source",
                    "source_hashes": {},
                    "call_ids": ids,
                    "helper_trace": trace,
                    "status": "invalid_helper" if invalid else "scored",
                    "correct": False,
                    "f1": 0,
                },
            )
    save(
        output / "starts" / "orphan.json",
        {
            "call_id": "orphan",
            "condition": "format_reminder",
            "role": "helper",
            "available": False,
            "usage": {},
            "started": 20,
        },
    )
    return output, cases_path


def test_planned_denominators_protocol_content_cost_and_helper_agreement(tmp_path):
    m = implementation()
    output, cases = fixture(tmp_path)
    report = m.analyze(output, cases, draws=50)
    assert report["groups"]["trained_helper"]["em"] == 1
    assert report["groups"]["format_reminder"]["em"] == 0.5
    assert report["groups"]["format_reminder"]["status_counts"] == {
        "scored": 1,
        "missing_episode": 1,
    }
    pair = report["comparisons"]["trained_helper_minus_base_helper"]
    assert pair["wins"]["both_valid"] == 1
    assert pair["wins"]["protocol_involved"] == 1
    assert pair["em"]["estimate"] == 1
    assert report["method"]["component_clusters"] == [["a", "b"]]
    assert pair["helper_agreement"]["both_valid_steps"] == 2
    assert pair["helper_agreement"]["changed_answer_steps"] == 2
    assert pair["helper_agreement"]["changed_resolved_question_steps"] == 1
    assert report["new_physical_cost"]["calls"] == 14
    assert report["new_physical_cost"]["total_tokens"] == 65
    assert report["new_physical_cost"]["known_latency_seconds"] == 30
    assert report["new_physical_cost"]["unknown_usage_calls"] == 1
    assert report["unresolved_start_ids"] == ["orphan"]
    assert len(report["comparisons"]) == 3
    path = tmp_path / "analysis.json"
    m.write_report(report, path)
    assert "not annotated-step accuracy" in path.with_suffix(".md").read_text()
    with pytest.raises(FileExistsError):
        m.write_report(report, path)


@pytest.mark.parametrize("field,value", [("seed", 101), ("root_request_digest", "changed")])
def test_rejects_unmatched_frozen_roots_or_seeds(tmp_path, field, value):
    m = implementation()
    output, cases = fixture(tmp_path)
    path = output / "episodes/a-r0-trained_helper.json"
    row = json.loads(path.read_text())
    row[field] = value
    save(path, row)
    with pytest.raises(ValueError, match="matched"):
        m.analyze(output, cases, draws=10)


def test_rejects_changed_receipt_request_and_case_hash(tmp_path):
    m = implementation()
    output, cases = fixture(tmp_path)
    path = next((output / "calls").glob("*.json"))
    row = json.loads(path.read_text())
    row["request"]["prompt"] = "changed"
    save(path, row)
    with pytest.raises(ValueError, match="request"):
        m.analyze(output, cases, draws=10)
    cases.write_text(cases.read_text() + "\n")
    with pytest.raises(ValueError, match="cases"):
        m.analyze(output, cases, draws=10)


def test_unavailable_helper_is_unobserved_not_returned_protocol_failure(tmp_path):
    m = implementation()
    output, cases = fixture(tmp_path)
    path = output / "calls/a-r0-base_helper-helper-1.json"
    row = json.loads(path.read_text())
    row.update(available=False, text=None, error="generation failed")
    save(path, row)
    report = m.analyze(output, cases, draws=10)
    assert report["groups"]["base_helper"]["unobserved_outcomes"] == 1
    assert report["groups"]["base_helper"]["returned_protocol_invalid_outcomes"] == 0
    pair = report["comparisons"]["trained_helper_minus_base_helper"]
    assert pair["wins"]["missing_involved"] == 1
    assert pair["wins"]["protocol_involved"] == 0


def test_rejects_unmatched_downstream_sampling_contract(tmp_path):
    m = implementation()
    output, cases = fixture(tmp_path)
    for condition, seed in (("base_helper", 101), ("trained_helper", 999)):
        path = output / f"calls/a-r0-{condition}-helper-1.json"
        row = json.loads(path.read_text())
        row["request"]["seed"] = seed
        row["request_digest"] = probe.runtime.digest(row["request"])
        save(path, row)
    with pytest.raises(ValueError, match="sampling"):
        m.analyze(output, cases, draws=10)


def test_transitive_components_and_unequal_cluster_sizes_keep_parent_weighting():
    m = implementation()
    cases = [
        {"id": p, "metadata": {"component_ids": ids}}
        for p, ids in (("a", ["x"]), ("b", ["x", "y"]), ("c", ["y"]), ("d", []))
    ]
    clusters = m.component_clusters(cases)
    assert clusters == [["a", "b", "c"], ["d"]]
    result = m.clustered_interval({"a": 1, "b": 0, "c": 0, "d": 1}, clusters, 1000, 123)
    assert result["estimate"] == 0.5
    assert result["ci95"] == [1 / 3, 1]
