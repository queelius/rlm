def test_all_fresh_alf_arms_must_be_observed_before_control(tmp_path):
    import launch_qampari_sampling_control as launcher

    summary = {
        "planned_episodes": 72,
        "recorded_episodes": 72,
        "all_slots_recorded": True,
        "groups": {
            p: {"observed": 24, "missing_or_unobserved": 0}
            for p in ("flat", "manager_worker", "local_reason")
        },
    }
    assert launcher.predecessor_complete(summary)
    summary["groups"]["local_reason"]["missing_or_unobserved"] = 1
    assert not launcher.predecessor_complete(summary)
    cmd = launcher.command(tmp_path)
    assert cmd[cmd.index("--baseline") + 1].endswith("qampari-reading-001")
    assert cmd[cmd.index("--hours") + 1] == "0.5"
