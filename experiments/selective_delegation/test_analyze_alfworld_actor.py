import copy

import pytest


def test_role_identity_is_checked_before_shared_environment_projection():
    import analyze_alfworld_actor as m

    for role, enabled in (("flat", True), ("worker", True), ("manager", False)):
        req = {
            "role": role,
            "adapter_checkpoint": "cp33",
            "adapter_enabled": enabled,
            "adapter_sha256": "weights" if enabled else None,
        }
        row = {
            "request": req,
            "request_digest": m.runtime.digest(req),
            "adapter_enabled": enabled,
            "adapter_sha256": req["adapter_sha256"],
        }
        projected = m.adapter_neutral_projection(row, "cp33", "weights")
        assert not projected["request"]["adapter_enabled"]
        assert "adapter_checkpoint" not in projected["request"]
        assert row["request"]["adapter_checkpoint"] == "cp33"
        wrong = copy.deepcopy(row)
        wrong["request"]["adapter_enabled"] = not enabled
        wrong["request_digest"] = m.runtime.digest(wrong["request"])
        with pytest.raises(ValueError, match="role"):
            m.adapter_neutral_projection(wrong, "cp33", "weights")


def test_interaction_is_paired_game_level_and_requires_all_four_observed():
    import analyze_alfworld_actor as m

    games = {0: {"scene": "a"}, 1: {"scene": "b"}}
    rows = [
        {
            "game_index": g,
            "seed": s,
            "policy": p,
            "observed": True,
            "won": p in ("base_manager", "trained_manager", "trained_flat"),
        }
        for g in games
        for s in (1, 2)
        for p in m.POLICIES
    ]
    result = m.interaction(rows, games, (1, 2), draws=100)
    assert result["game_bootstrap"]["estimate"] == -1
    assert result["scene_bootstrap"]["estimate"] == -1
    rows[0]["observed"] = False
    result = m.interaction(rows, games, (1, 2), draws=100)
    assert result["unknown_pairs"] == 1 and result["game_bootstrap"] is None


def test_actual_completed_manager_trace_survives_only_adapter_metadata_projection():
    import json
    from pathlib import Path

    import analyze_alfworld_actor as m

    output = Path(
        "/project/alex_phd/runs/rlm-research-r4/sidecars/"
        "selective-delegation-20260921/alfworld-closed-loop-001"
    )

    def read(p):
        return json.loads(p.read_text())

    plan = read(output / "PLAN.json")
    job = next(j for j in plan["cases"] if j["policy"] == "manager_worker")
    row = read(output / "episodes" / (job["episode_id"] + ".json"))
    calls = {}
    for cid in row["call_ids"]:
        base = read(output / "calls" / (cid + ".json"))
        synthetic = copy.deepcopy(base)
        fields = m.collector.role_identity(base["role"], "fixture_weights")
        synthetic.update(fields)
        synthetic["request"].update(fields, adapter_checkpoint="cp33")
        synthetic["request_digest"] = m.runtime.digest(synthetic["request"])
        projected = m.adapter_neutral_projection(synthetic, "cp33", "fixture_weights")
        assert projected["request"] == base["request"]
        assert projected["request_digest"] == base["request_digest"]
        calls[cid] = projected
    audit = m.unseen.indexed_audit.audit_episode(job, row, calls, read, output, plan)
    assert audit["actions"] == row["actions"]
