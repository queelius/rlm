"""Small mathematical and saved-request fixtures, no model execution."""

import json

import analyze_direct_adapted as analysis
import eval_direct_adapted as runner
import pytest
from test_eval_planner import CASE


def test_parent_paired_math_separates_protocol_changes_and_cluster_weighting():
    parents, conditions = ["a", "b"], runner.CONDITIONS
    values = {}
    for parent in parents:
        for repeat in range(2):
            for condition in conditions:
                base = condition == "base_direct"
                em = float((parent == "a" and not base) or (parent == "b" and base and repeat == 0))
                valid = not (parent == "a" and base)
                values[parent, repeat, condition] = {
                    "em": em,
                    "f1": em,
                    "valid": valid,
                    "present": True,
                    "status": "scored" if valid else "invalid_answer",
                }
    result = analysis.compare(values, parents, 2, [["a", "b"]], *conditions, draws=20, seed=7)
    assert result["em"] == {"estimate": 0.25, "ci95": [0.25, 0.25]}
    assert result["wins"]["protocol_involved"] == 2
    assert result["losses"]["both_valid"] == 1
    assert result["wins"]["parents"] == ["a"]


@pytest.mark.parametrize("seed_base", [None, 2026092112])
def test_request_verification_rejects_adapter_or_seed_drift(seed_base):
    request = {
        "prompt": runner.evaluation.direct_prompt(CASE),
        "input_token_ids": [3, 4],
        "condition": "helper_sft_direct",
        "role": "direct_answer",
        "model": "model",
        "adapter_enabled": True,
        "adapter_name": "helper_sft",
        "adapter_sha256": "sha",
        "seed": (runner.evaluation.SEED if seed_base is None else seed_base)
        + int(runner.probe.runtime.digest(CASE["id"])[:6], 16)
        + 2,
        "sampling": runner.SAMPLING,
    }
    call = {
        **{
            k: request[k]
            for k in (
                "condition",
                "role",
                "model",
                "adapter_enabled",
                "adapter_name",
                "adapter_sha256",
            )
        },
        "request": request,
        "request_digest": runner.probe.runtime.digest(request),
        "available": True,
        "input_token_ids": [3, 4],
        "output_token_ids": [5],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }
    plan = {"model": "model", "helper_adapter_binding": {"adapter_model.safetensors": "sha"}}
    if seed_base is not None:
        plan["seed"] = seed_base
    analysis.validate_call(call, CASE, 0, "helper_sft_direct", plan)
    changed = {**request, "seed": request["seed"] + 1}
    with pytest.raises(ValueError, match="request"):
        analysis.validate_call(
            {**call, "request": changed, "request_digest": runner.probe.runtime.digest(changed)},
            CASE,
            0,
            "helper_sft_direct",
            plan,
        )
    with pytest.raises(ValueError, match="adapter"):
        analysis.validate_call(
            {**call, "adapter_sha256": "other"}, CASE, 0, "helper_sft_direct", plan
        )


def test_report_never_overwrites_either_sibling(tmp_path):
    path = tmp_path / "report.json"
    path.with_suffix(".md").write_text("owned")
    with pytest.raises(FileExistsError):
        analysis.write_report({"example": 1}, path)
    assert not path.exists()
    assert path.with_suffix(".md").read_text() == "owned"


def test_hotpot_native_regrade_keeps_missing_denominator_and_singleton_label(tmp_path, monkeypatch):
    cases = [
        {
            **CASE,
            "id": str(i),
            "split": "transfer",
            "dataset": "hotpotqa",
            "answer": "yes",
            "metadata": {},
        }
        for i in range(32)
    ]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("\n".join(map(json.dumps, cases)))
    sha = runner.probe.campaign.sha
    monkeypatch.setattr(runner, "HOTPOT_CASES_SHA", sha(cases_path))
    adapter = tmp_path / "adapter"
    model = tmp_path / "model"
    adapter.mkdir()
    model.mkdir()
    (adapter / "adapter_model.safetensors").write_text("fixture")
    runner.probe.runtime.save(tmp_path / "PLAN.json", {})
    for name in ("local-research-manifest.json", "generation_config.json"):
        runner.probe.runtime.save(model / name, {})
    output = tmp_path / "output"
    plan = {
        "schema": "paired-direct-helper-adapter-v1",
        "panel": "hotpot_explorer32",
        "dataset": "hotpotqa",
        "split": "transfer",
        "parents": 32,
        "maximum_calls": 128,
        "cases_sha256": sha(cases_path),
        "case_ids": [c["id"] for c in cases],
        "repeats": 2,
        "conditions": list(runner.CONDITIONS),
        "seed": 2026092112,
        "sampling": runner.SAMPLING,
        "metric": "official_hotpotqa_em_f1",
        "dependencies": {},
        "helper_adapter": str(adapter),
        "helper_adapter_binding": {
            "adapter_model.safetensors": sha(adapter / "adapter_model.safetensors")
        },
        "helper_training_plan_sha256": sha(tmp_path / "PLAN.json"),
        "model": str(model),
        "model_manifest_sha256": sha(model / "local-research-manifest.json"),
        "generation_config_sha256": sha(model / "generation_config.json"),
    }
    runner.probe.runtime.save(output / "PLAN.json", plan)

    class Client:
        def call(self, case, condition, repeat):
            enabled = condition == "helper_sft_direct"
            request = {
                "prompt": runner.evaluation.direct_prompt(case),
                "input_token_ids": [3],
                "condition": condition,
                "role": "direct_answer",
                "model": str(model),
                "adapter_enabled": enabled,
                "adapter_name": "helper_sft" if enabled else None,
                "adapter_sha256": plan["helper_adapter_binding"]["adapter_model.safetensors"]
                if enabled
                else None,
                "seed": 2026092112 + int(runner.probe.runtime.digest(case["id"])[:6], 16) + 2,
                "sampling": runner.SAMPLING,
            }
            cid = f"{case['id']}-r{repeat}-{condition}-answer"
            row = {
                **request,
                "request": request,
                "request_digest": runner.probe.runtime.digest(request),
                "call_id": cid,
                "available": True,
                "output_token_ids": [4],
                "text": '{"answer":"yes"}' if enabled else '{"answer":"yes indeed"}',
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            runner.probe.runtime.save(output / "calls" / (cid + ".json"), row)
            runner.probe.runtime.save(output / "starts" / (cid + ".json"), row)
            return row

    client = Client()
    client.output = output
    for condition in runner.CONDITIONS:
        runner.collect_episode(client, cases[0], condition, 0)
    result = analysis.analyze(output, cases_path, draws=20)
    assert result["groups"]["base_direct"]["f1"] == 0
    assert result["groups"]["helper_sft_direct"]["f1"] == 1 / 64
    assert result["groups"]["base_direct"]["planned"] == 64
    assert result["groups"]["base_direct"]["recorded"] == 1
    assert not result["all_direct_episodes_present"]
    assert result["method"]["parents_missing_component_ids"] == 32
    assert "not verified atomic independence" in analysis.markdown(result)
    assert result["method"]["metric"] == "official_hotpotqa_em_f1"
    assert result["baseline_agreement"] == {"output": None, "present": False}
    with pytest.raises(ValueError, match="Hotpot optional"):
        analysis.analyze(output, cases_path, baseline_output=tmp_path)
