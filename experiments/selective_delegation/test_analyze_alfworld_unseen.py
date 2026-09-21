def test_game_and_unequal_scene_clusters_preserve_paired_repeat_weighting():
    import analyze_alfworld_unseen as m

    games = {0: {"scene": "a"}, 1: {"scene": "a"}, 2: {"scene": "b"}}
    rows = [
        {"game_index": g, "seed": s, "policy": p, "observed": True, "won": p == "left" and g < 2}
        for g in games
        for s in (1, 2)
        for p in ("left", "right")
    ]
    result = m.paired(rows, games, (1, 2), "left", "right", draws=100)
    assert result["planned_pairs"] == 6 and result["wins"] == 4
    assert result["game_bootstrap"]["estimate"] == 2 / 3
    assert result["scene_bootstrap"]["estimate"] == 2 / 3
    assert result["scene_signflip"]["p_two_sided"] == 1
    assert result["game_signflip"]["p_two_sided"] == 0.5


def test_missing_pairs_are_unknown_and_disable_full_panel_intervals():
    import analyze_alfworld_unseen as m

    rows = [{"game_index": 0, "seed": 1, "policy": "left", "won": True, "observed": True}]
    r = m.paired(rows, {0: {"scene": "x"}}, (1, 2), "left", "right", draws=10)
    assert r["unknown_pairs"] == 2 and r["wins"] == 0
    assert r["game_bootstrap"] is None and r["scene_bootstrap"] is None


def test_qualified_auditors_replay_completed_native_examples():
    import json
    from pathlib import Path

    import analyze_alfworld_unseen as m
    from transformers import AutoTokenizer

    root = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
    tokenizer = AutoTokenizer.from_pretrained(
        m.collector.local.base.evaluation.planner.BASE,
        local_files_only=True,
        trust_remote_code=False,
    )
    for directory, policy in (
        ("alfworld-closed-loop-001", "manager_worker"),
        ("alfworld-local-reason-001", "local_reason"),
    ):
        output = root / directory
        def read(p):
            return json.loads(p.read_text())
        plan = read(output / "PLAN.json")
        job = next(j for j in plan["cases"] if j["policy"] == policy)
        row = read(output / "episodes" / (job["episode_id"] + ".json"))
        calls = {cid: read(output / "calls" / (cid + ".json")) for cid in row["call_ids"]}
        if policy == "local_reason":
            audit = m.local_audit.audit_episode(job, row, calls, read, output, plan, tokenizer)
        else:
            audit = m.indexed_audit.audit_episode(job, row, calls, read, output, plan)
        assert audit["actions"] == row["actions"]
