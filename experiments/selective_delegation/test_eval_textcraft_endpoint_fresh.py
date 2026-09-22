import copy

import eval_textcraft_endpoint_fresh as reader
import pytest


def test_actual007_template_keeps_every_slot_seed_and_budget():
    template, tasks = reader.template_inputs()
    plan = reader.bound_plan(
        template,
        "rl",
        {"path": "/checkpoint", "sha256": "a", "training_plan_sha256": "p"},
        "/training",
    )
    assert len(tasks) == 16 and len(plan["jobs"]) == 32
    assert [j["seed"] for j in plan["jobs"]] == [2026092204, 2026092205] * 16
    for a, b in zip(plan["jobs"], template["jobs"], strict=True):
        assert {k: v for k, v in a.items() if k != "condition"} == {
            k: v for k, v in b.items() if k != "condition"
        }
    assert plan["budget_seconds"] == 3600
    assert plan["max_global_calls"] == 96 and plan["max_global_output_tokens"] == 8192
    assert plan["max_new_tokens"] == 256 and plan["input_plus_output_limit"] == 8192
    assert plan["sampling"] == {"temperature": 0.5, "top_p": 1.0, "top_k": 0}
    assert "teacher" not in plan


def test_endpoint_is_terminal_actual_step_not_best_or_fixed23(tmp_path):
    terminal = dict(
        complete=True,
        failure=None,
        endpoint_usable=True,
        actual_optimizer_steps=2,
        committed_optimizer_steps=2,
    )
    state = dict(step=2)
    commit = dict(step=2)
    reader.validate_endpoint("rl", terminal, state, commit, 2)
    reader.validate_endpoint("matched_sft", terminal, state, commit, 2)
    for changed in (
        {**terminal, "endpoint_usable": False},
        {**terminal, "actual_optimizer_steps": 1},
    ):
        with pytest.raises(ValueError):
            reader.validate_endpoint("rl", changed, state, commit, 2)
    with pytest.raises(ValueError):
        reader.validate_endpoint("rl", terminal, {"step": 1}, {"step": 1}, 2)
    with pytest.raises(ValueError):
        reader.validate_endpoint("rl", terminal, state, commit, 0)


def test_comparison_rejects_seed_change_and_preserves_unknown_pairs():
    import analyze_textcraft_endpoint_fresh as analyzer

    template, _ = reader.template_inputs()
    plans = [template] + [
        reader.bound_plan(
            template,
            kind,
            {"path": "/checkpoint", "sha256": kind, "training_plan_sha256": kind},
            "/training",
        )
        for kind in ("rl", "matched_sft")
    ]
    analyzer.match_slots(plans)
    changed = copy.deepcopy(plans)
    changed[2]["jobs"][0]["seed"] += 1
    with pytest.raises(ValueError):
        analyzer.match_slots(changed)
    jobs = plans[0]["jobs"]
    known = {(j["task_id"], j["repeat"]): dict(observed=True, native_score=1) for j in jobs}
    result = analyzer.profiles.compare(jobs, known, {})
    assert result["unknown_pairs"] == 32 and result["complete_panel_difference"] is None
