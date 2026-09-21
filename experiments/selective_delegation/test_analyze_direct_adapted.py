"""Small mathematical and saved-request fixtures, no model execution."""

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


def test_request_verification_rejects_adapter_or_seed_drift():
    request = {
        "prompt": runner.evaluation.direct_prompt(CASE),
        "input_token_ids": [3, 4],
        "condition": "helper_sft_direct",
        "role": "direct_answer",
        "model": "model",
        "adapter_enabled": True,
        "adapter_name": "helper_sft",
        "adapter_sha256": "sha",
        "seed": runner.evaluation.SEED + int(runner.probe.runtime.digest(CASE["id"])[:6], 16) + 2,
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
