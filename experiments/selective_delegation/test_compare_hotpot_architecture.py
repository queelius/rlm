"""Small paired math and actual strict-score/native-request fixtures."""

import pytest


def test_official_special_answer_and_missing_protocol_are_distinct():
    import compare_hotpot_architecture as mod

    case = {"answer": "yes"}
    row = {"status": "scored"}
    outcome = mod.outcome(row, {"available": True, "text": '{"answer":"yes indeed"}'}, case)
    assert outcome["valid"] and outcome["em"] == outcome["f1"] == 0
    assert mod.outcome(None, None, case)["status"] == "missing_episode"
    assert mod.outcome({"status": "invalid_helper"}, None, case)["status"] == "invalid_helper"
    assert mod.outcome(row, {"available": False}, case)["status"] == "unavailable_final"


def test_paired_parent_means_and_protocol_win_decomposition():
    import compare_hotpot_architecture as mod

    values = {}
    for parent in ("a", "b"):
        for repeat in range(2):
            for arm in ("direct", "planner"):
                hit = (parent == "a" and arm == "planner") or (
                    parent == "b" and repeat == 0 and arm == "direct"
                )
                protocol = parent == "a" and arm == "direct"
                values[parent, repeat, arm] = {
                    "em": float(hit),
                    "f1": float(hit),
                    "valid": not protocol,
                    "status": "invalid_final" if protocol else "scored",
                }
    report = mod.paired(values, ["a", "b"], [["a", "b"]], draws=20)
    assert report["em"] == {"estimate": 0.25, "ci95": [0.25, 0.25]}
    assert report["wins"]["protocol_involved"] == 2
    assert report["losses"]["both_valid"] == 1


def test_native_final_seed_adapter_prompt_and_tokens_are_checked():
    import compare_hotpot_architecture as mod

    plan = {
        "model": "base",
        "helper_contract": {"mode": "base"},
        "adapter_files_sha256": {"adapter_model.safetensors": "root"},
    }
    request = {
        "prompt": "public prompt",
        "input_token_ids": [1, 2],
        "condition": "base",
        "role": "final",
        "model": "base",
        "model_instance": "root",
        "helper_contract": plan["helper_contract"],
        "adapter_enabled": False,
        "adapter_sha256": None,
        "seed": 1002,
        "sampling": {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": 128,
            "max_time": 90.0,
        },
    }
    call = {
        **request,
        "request": request,
        "request_digest": mod.probe.runtime.digest(request),
        "input_token_ids": [1, 2],
        "output_token_ids": [3],
        "available": True,
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }
    mod.validate_call(call, plan, "base", "final", "public prompt", 1002, 128)
    with pytest.raises(ValueError, match="request"):
        mod.validate_call(call, plan, "base", "final", "public prompt", 1003, 128)
    with pytest.raises(ValueError, match="request"):
        mod.validate_call(call, plan, "base", "final", "host gold leaked", 1002, 128)
    bad = {**call, "output_token_ids": [3, 4]}
    with pytest.raises(ValueError, match="token"):
        mod.validate_call(bad, plan, "base", "final", "public prompt", 1002, 128)


def test_end_to_end_sparse_panel_matches_separate_official_regrades(tmp_path):
    import json

    import compare_hotpot_architecture as mod
    from test_eval_planner import CASE

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return mod.score_hotpot.sha256(path)

    cases = [
        {
            **CASE,
            "id": f"p{i}",
            "split": "transfer",
            "dataset": "hotpotqa",
            "answer": "yes",
            "metadata": {"component_ids": [f"document-{i}"]},
        }
        for i in range(128)
    ]
    case = cases[0]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("\n".join(map(json.dumps, cases)))
    root, helper, model = tmp_path / "sft/48", tmp_path / "helper/36", tmp_path / "model"
    bindings = {}
    for adapter, step in ((root, 48), (helper, 36)):
        bindings[step] = {
            "STATE.json": save(adapter / "STATE.json", {"step": step}),
            "adapter_model.safetensors": save(adapter / "adapter_model.safetensors", {}),
        }
        save(adapter.parent / "PLAN.json", {})
    source = tmp_path / "source"
    dependency = source / "rl_planner.py"
    save(dependency, {})
    collector = source / "eval_planner.py"
    save(collector, {})
    manifest_sha = save(model / "local-research-manifest.json", {})
    outputs = {}
    for arm, condition in (("planner", "sft"), ("direct", "base")):
        output = outputs[arm] = tmp_path / arm
        contract = {"mode": "base", "model": str(model)}
        if arm == "planner":
            contract = {
                "mode": "trained_helper",
                "model": str(model),
                "adapter": str(helper),
                "adapter_binding": bindings[36],
                "training_plan_sha256": mod.score_hotpot.sha256(helper.parent / "PLAN.json"),
            }
        plan = {
            "case_ids": [c["id"] for c in cases],
            "repeats": 2,
            "conditions": [condition],
            "split": "transfer",
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": 0,
            "seed": mod.evaluation.SEED,
            "mode": arm,
            "execution": "isolated" if arm == "planner" else "direct",
            "helper_contract": contract,
            "cases_sha256": mod.score_hotpot.sha256(cases_path),
            "model": str(model),
            "model_manifest_sha256": manifest_sha,
            "adapter": str(root),
            "adapter_files_sha256": bindings[48],
            "training_plan_sha256": mod.score_hotpot.sha256(root.parent / "PLAN.json"),
            "dependencies": {str(dependency): mod.score_hotpot.sha256(dependency)},
            "source_sha256": mod.score_hotpot.sha256(collector),
            "environment": {},
        }
        save(output / "PLAN.json", plan)
        identity = f"p0-r0-{condition}" + ("-isolated" if arm == "planner" else "-direct")
        seed = plan["seed"] + int(mod.probe.runtime.digest("p0")[:6], 16)
        parsed = {"subquestions": ["Public subquestion?"]}
        trace = {
            "execution": "isolated",
            "steps": [
                {
                    "step": 1,
                    "question": "Public subquestion?",
                    "resolved_question": "Public subquestion?",
                    "answer": "yes",
                }
            ],
        }
        jobs = [
            (
                "-final",
                "final",
                mod.evaluation.direct_prompt(case),
                seed + 2,
                128,
                '{"answer":"yes indeed"}',
            )
        ]
        if arm == "planner":
            jobs = [
                (
                    "-root",
                    "root",
                    mod.evaluation.planner_prompt(case),
                    seed,
                    128,
                    json.dumps(parsed),
                ),
                (
                    "-helper-1",
                    "helper",
                    mod.evaluation.isolated_helper_prompt(case, "Public subquestion?"),
                    seed + 1,
                    384,
                    '{"answer":"yes"}',
                ),
                (
                    "-final",
                    "final",
                    mod.evaluation.final_prompt(case, parsed, trace),
                    seed + 2,
                    128,
                    '{"answer":"yes"}',
                ),
            ]
        call_ids = []
        for suffix, role, prompt, call_seed, cap, text in jobs:
            enabled = arm == "planner" and role != "final"
            request = {
                "condition": condition,
                "role": role,
                "model": str(model),
                "model_instance": "helper" if role == "helper" else "root",
                "adapter_enabled": enabled,
                "adapter_sha256": bindings[36 if role == "helper" else 48][
                    "adapter_model.safetensors"
                ]
                if enabled
                else None,
                "helper_contract": contract,
                "prompt": prompt,
                "seed": call_seed,
                "input_token_ids": [1],
                "sampling": {
                    "do_sample": True,
                    "temperature": 0.5,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_new_tokens": cap,
                    "max_time": 90.0,
                },
            }
            cid = identity + suffix
            call = {
                **request,
                "call_id": cid,
                "request": request,
                "request_digest": mod.probe.runtime.digest(request),
                "available": True,
                "text": text,
                "input_token_ids": [1],
                "output_token_ids": [2],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "started": 1,
                "ended": 2,
            }
            save(output / "calls" / (cid + ".json"), call)
            save(output / "starts" / (cid + ".json"), call)
            call_ids.append(cid)
        save(
            output / "episodes" / (identity + ".json"),
            {
                "episode_id": identity,
                "case_id": "p0",
                "repeat": 0,
                "condition": condition,
                "seed": seed,
                "call_ids": call_ids,
                "status": "scored",
            },
        )
        mod.score_hotpot.score(cases_path, output, tmp_path / ("analysis-" + arm))
    report = mod.analyze(outputs["planner"], outputs["direct"], cases_path, draws=20)
    assert report["groups"]["planner"]["correct"] == 1
    assert report["groups"]["direct"]["f1"] == 0
    assert report["groups"]["planner"]["status_counts"]["missing_episode"] == 255
    assert report["paired_planner_minus_direct"]["em"]["estimate"] == 1 / 256
    assert report["groups"]["planner"]["physical_cost"]["calls"] == 3
    assert report["method"]["seeds"]["paired_observed_final_requests"] == 1
    mod.write_report(report, tmp_path / "result.json")
    with pytest.raises(FileExistsError):
        mod.write_report(report, tmp_path / "result.json")
