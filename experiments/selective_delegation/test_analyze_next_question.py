"""Focused planned-denominator math and native next-question receipt replay."""

import json
from pathlib import Path

import pytest


def test_eligible_cluster_restriction_keeps_ineligible_bridge_connectivity():
    import analyze_helper
    import analyze_next_question as analysis

    cases = [
        {"id": "a", "metadata": {"component_ids": ["x"]}},
        {"id": "bridge", "metadata": {"component_ids": ["x", "y"]}},
        {"id": "c", "metadata": {"component_ids": ["y"]}},
    ]
    values = {
        (p, r, arm): dict(
            em=int(p == "a" and arm == "feedback"),
            f1=0.0,
            observed=True,
            valid=True,
            present=True,
            status="scored",
        )
        for p in ("a", "c")
        for r in range(2)
        for arm in ("hidden", "feedback")
    }
    report = analysis.panel_summary(
        values,
        ["a", "c"],
        [cases[0], cases[2]],
        draws=50,
        seed=7,
        selected_clusters=analyze_helper.component_clusters(cases),
    )
    assert report["component_clusters"] == [["a", "c"]]
    assert report["contrasts"]["em"]["component"]["ci95"] == [0.5, 0.5]
    assert report["contrasts"]["em"]["parent"]["ci95"] == [0.0, 1.0]


def test_selected_and_eligible_denominators_preserve_missing_and_protocol():
    import analyze_next_question as analysis

    def result(em=0, *, observed=True, valid=True, eligible=True):
        return dict(
            em=em,
            f1=float(em),
            observed=observed,
            valid=valid,
            eligible=eligible,
            present=observed,
            status="scored" if observed else "missing",
        )

    values = {
        (p, r, arm): result(observed=False, valid=False, eligible=p != "c")
        for p in ("a", "b", "c")
        for r in range(2)
        for arm in ("hidden", "feedback")
    }
    for repeat in range(2):
        values["a", repeat, "hidden"] = result(valid=repeat == 0)
        values["a", repeat, "feedback"] = result(1)
        values["a", repeat, "hidden"].update(
            next_question="Where is #2?", resolved_question="Where is Paris?"
        )
        values["a", repeat, "feedback"].update(
            next_question="Where is Paris?", resolved_question="Where is Paris?"
        )
    values["b", 0, "hidden"] = result(1)
    values["b", 1, "hidden"] = values["b", 1, "feedback"] = result()
    cases = [
        {"id": p, "metadata": {"component_ids": ["shared"] if p in ("a", "b") else []}}
        for p in ("a", "b", "c")
    ]
    report = analysis.panel_summary(values, ["a", "b", "c"], cases, draws=50, seed=7)
    assert report["arms"]["hidden"]["planned"] == 6
    assert report["arms"]["feedback"]["missing"] == 3
    assert report["arms"]["feedback"]["em_lower"] == pytest.approx(2 / 6)
    assert report["paired"]["wins_both_valid"] == 1
    assert report["paired"]["wins_protocol"] == 1
    assert report["paired"]["unobserved"] == 3
    assert report["paired"]["changed_raw_next_question"] == 2
    assert report["paired"]["changed_resolved_next_question"] == 0
    assert report["contrasts"]["em"]["component"]["estimate"] == pytest.approx(1 / 6)
    eligible = analysis.panel_summary(values, ["a", "b"], cases[:2], draws=50, seed=7)
    assert eligible["arms"]["feedback"]["planned"] == 4
    assert eligible["contrasts"]["em"]["component"]["estimate"] == pytest.approx(0.25)


def test_replay_validates_native_seed_actual_binding_and_unavailable_prefix(tmp_path):
    import analyze_next_question as analysis
    import next_question_probe as collector
    import probe
    from test_eval_planner import CASE

    case = {**CASE, "answer": "Gold"}
    contract = {
        "mode": "trained_helper",
        "model": str(collector.evaluation.planner.BASE),
        "adapter_binding": {"adapter_model.safetensors": "helper"},
    }
    plan = {
        "root_adapter_binding": {"adapter_model.safetensors": "root"},
        "helper_contract": contract,
    }
    prefix = {
        "eligible": True,
        "source_hashes": {},
        "dependencies": [],
        "trace": [
            {"step": 1, "question": "Who?", "resolved_question": "Who?", "answer": "FIRST_SECRET"},
            {
                "step": 2,
                "question": "Where was #1 born?",
                "resolved_question": "Where was FIRST_SECRET born?",
                "answer": "SECOND_SECRET",
            },
        ],
    }
    calls = {}

    class Fake:
        output = tmp_path

        def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
            request = analysis.expected_request(prompt, role, seed, max_new_tokens, plan)
            request["input_token_ids"] = [1, 2]
            row = dict(
                call_id=identity,
                request=request,
                request_digest=probe.runtime.digest(request),
                input_token_ids=[1, 2],
                output_token_ids=[3],
                available=True,
                usage={"prompt_tokens": 2, "completion_tokens": 1},
                text='{"subquestions":["What is #2?"]}' if role == "root" else '{"answer":"Gold"}',
            )
            row.update(
                {
                    key: request[key]
                    for key in (
                        "condition",
                        "role",
                        "model",
                        "model_instance",
                        "helper_contract",
                        "adapter_enabled",
                        "adapter_sha256",
                    )
                }
            )
            calls[identity] = row
            return row

    row = collector.collect(Fake(), case, prefix, "hidden", 0)
    reconstructed = analysis.replay_episode(row, case, prefix, plan, calls)
    assert reconstructed["correct"] and reconstructed["next_question"] == "What is #2?"
    root = calls[row["episode_id"] + "-root"]
    assert "FIRST_SECRET" not in root["request"]["prompt"]
    assert "SECOND_SECRET" in calls[row["episode_id"] + "-helper"]["request"]["prompt"]
    root["request"]["seed"] += 1
    with pytest.raises(analysis.BindingError, match="native request"):
        analysis.replay_episode(row, case, prefix, plan, calls)
    unavailable = {**prefix, "eligible": False, "trace": prefix["trace"][:1]}
    missing = collector.collect(Fake(), case, unavailable, "feedback", 1)
    replay = analysis.replay_episode(missing, case, unavailable, plan, {})
    assert replay["status"] == "source_prefix_unavailable" and not replay["outcome_observed"]
    assert not replay["call_ids"]
    assert (
        json.loads((Path(tmp_path) / "episodes" / (missing["episode_id"] + ".json")).read_text())
        == missing
    )
