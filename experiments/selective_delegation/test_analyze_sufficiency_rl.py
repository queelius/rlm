import pytest


def test_actual046_profile_authenticates_metric_without_legacy_manifest_field():
    import copy
    import json
    from pathlib import Path

    import analyze_sufficiency_rl as analysis

    root = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
    plan = json.loads((root / "sufficiency-compositional-readout-001/PLAN.json").read_text())
    manifest = json.loads((root / "sufficiency-compositional-inputs-001/MANIFEST.json").read_text())
    assert "official_metric_sha256" not in manifest
    mapping = analysis.held_metric_sources(plan, manifest, analysis.Audit())
    assert mapping == plan["panel_profile"]["metric_sha256"]
    altered = copy.deepcopy(plan)
    altered["panel_profile"]["metric_sha256"].pop(next(iter(mapping)))
    with pytest.raises(ValueError, match="profile"):
        analysis.held_metric_sources(altered, manifest, analysis.Audit())
    assert (
        analysis.held_metric_sources({}, {"official_metric_sha256": mapping}, analysis.Audit())
        == mapping
    )


@pytest.mark.parametrize("component_count", [20, 30])
def test_frozen_panel_partition_accepts_exact20_or30_and_rejects_overlap(component_count):
    import analyze_sufficiency_rl as analysis

    parents = [f"p{i}" for i in range(32)]
    cases = [{"parent_id": parent} for parent in parents for _ in range(2)]
    paired_count = 32 - component_count
    clusters = [parents[i : i + 2] for i in range(0, 2 * paired_count, 2)]
    clusters += [[p] for p in parents[2 * paired_count :]]
    assert analysis.frozen_clusters(cases, {"component_clusters": clusters}, 384) == clusters
    with pytest.raises(ValueError, match="partition"):
        analysis.frozen_clusters(cases, {"component_clusters": clusters + [[parents[0]]]}, 384)
    with pytest.raises(ValueError, match="partition"):
        analysis.frozen_clusters(cases, {"component_clusters": clusters[:-1]}, 384)


def test_reward_protocol_and_zero_advantage_accounting():
    import analyze_sufficiency_rl as analysis

    groups = [[{"reward": 0, "all_valid": True} for _ in range(4)] for _ in range(16)]
    groups[0][3]["reward"] = 1
    result = analysis.reward_statistics(groups)
    assert result["paired_reward_sum"] == 1
    assert result["effective_groups"] == result["valid_mixed_groups"] == 1
    assert result["absolute_advantage_mass"] == pytest.approx(2)
    groups[0][0]["all_valid"] = False
    assert analysis.reward_statistics(groups)["valid_mixed_groups"] == 1
    assert analysis.reward_statistics(groups)["fully_protocol_valid_groups"] == 15
    with pytest.raises(ValueError):
        analysis.reward_statistics(groups[:-1])


def test_frozen_component_units_preserve_pairs_and_both_seeds():
    import analyze_sufficiency_rl as analysis

    def rows(values):
        return [
            {"parent_id": p, "seed": seed, "joint_em": v}
            for p, v in values.items()
            for seed in (1, 2)
        ]

    result = analysis.cluster_contrast(
        rows({"a": 0, "b": 0, "c": 0}),
        rows({"a": 1, "b": 1, "c": 0}),
        "joint_em",
        [["a", "b"], ["c"]],
        draws=1000,
    )
    assert result["estimate"] == pytest.approx(2 / 3)
    assert result["ci95"] == [0, 1]
    assert "parent_bootstrap_ci95" in result
    with pytest.raises(ValueError):
        analysis.cluster_contrast(rows({"a": 0}), rows({"a": 1}), "joint_em", [["b"]])


def test_native_training_request_unavailable_not_reward_zero():
    import analyze_sufficiency_rl as analysis

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, *args, **kwargs):
            return '{"answerable":true,"answer":"London"}'

    case = {"public": {"question": "Where?", "documents": []}}
    boundary = {"checkpoint": "fixed", "state": {"step": 0, "sample_cursor": 0}}
    commit = {"files": {"adapter_model.safetensors": "weights"}}
    request = {
        "prompt": analysis.paired.baseline.prompt(case),
        "input_token_ids": [1, 2],
        "seed": 7,
        "model": "base",
        "adapter_enabled": True,
        "role": "answer_sufficiency",
        "adapter": {
            "checkpoint": "fixed",
            "adapter_sha256": "weights",
            "optimizer_step": 0,
            "sample_cursor": 0,
        },
        "sampling": {
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 128,
            "max_time": 90.0,
        },
    }
    call = {
        "request": request,
        "request_digest": analysis.training.probe.runtime.digest(request),
        "input_token_ids": [1, 2],
        "available": False,
    }
    assert analysis.audit_training_call(
        call, case, 7, {"model": "base"}, boundary, commit, Tokenizer()
    ) == ("unavailable", None)
    call.update(
        available=True,
        output_token_ids=[3],
        generation_logps=[-0.5],
        usage={"prompt_tokens": 2, "completion_tokens": 1},
        text=Tokenizer().decode(),
    )
    assert (
        analysis.audit_training_call(
            call, case, 7, {"model": "base"}, boundary, commit, Tokenizer()
        )[0]
        == "valid"
    )
    request["seed"] = 8
    call["request_digest"] = analysis.training.probe.runtime.digest(request)
    with pytest.raises(ValueError, match="seed"):
        analysis.audit_training_call(
            call, case, 7, {"model": "base"}, boundary, commit, Tokenizer()
        )
