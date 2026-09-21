def test_predecessor_requires_all48_observed_and_fixed_command(tmp_path):
    import launch_textcraft as launch

    summary = {
        "planned_episodes": 48,
        "recorded_episodes": 48,
        "groups": {
            p: {"observed": 24, "missing_or_unobserved": 0} for p in ("flat", "manager_worker")
        },
    }
    assert launch.predecessor_complete(summary)
    summary["groups"]["flat"]["missing_or_unobserved"] = 1
    assert not launch.predecessor_complete(summary)
    command = launch.command(tmp_path)
    assert command[command.index("--hours") + 1] == "1"
    assert command[command.index("--prepared") + 1].endswith("textcraft-inputs-001")
