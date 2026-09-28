"""Focused contracts for the additive early-transfer readout."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def implementation():
    assert importlib.util.find_spec("transfer_common") is not None, "transfer adapter missing"
    import transfer_common

    return transfer_common


def test_actual_endpoints_are_admitted_and_not_confused_with_warm_or_each_other():
    t = implementation()
    raw, binder = (t.admit_endpoint(mode) for mode in ("raw", "binder"))
    assert raw["adapter"]["sha256"] == (
        "856fb4755c3a2b3297a38d6bed7d6f8e638535eba6723dc01cc0ee4a916db62e"
    )
    assert binder["adapter"]["sha256"] == (
        "2621e7d47aa7af51736640af22fd6846081a82f60fffdcfa8178151a6e3854ab"
    )
    assert raw["training_execution_mode"] == "raw"
    assert binder["training_execution_mode"] == "binder"


def test_uncommitted_state_cannot_be_admitted(tmp_path):
    t = implementation()
    source = t.FAMILIAR / "raw/train-0001"
    summary = t.f.read(source / "SUMMARY.json")
    checkpoint = tmp_path / "boundaries/sample-0001/checkpoint-0001"
    summary["endpoint"] = str(checkpoint)
    t.f.c.save(tmp_path / "SUMMARY.json", summary)
    t.f.c.save(tmp_path / "PLAN.json", t.f.read(source / "PLAN.json"))
    original = Path(t.f.read(source / "SUMMARY.json")["endpoint"])
    state = t.f.read(original / "STATE.json")
    state["step"] = 2
    t.f.c.save(checkpoint / "STATE.json", state)
    t.f.c.save(checkpoint / "COMMIT.json", t.f.read(original / "COMMIT.json"))
    with pytest.raises(ValueError, match="state|STATE"):
        t.admit_endpoint("raw", tmp_path)


def test_cross_interface_plan_keeps_frozen_b_pairing_and_true_actor_lineage(tmp_path):
    t = implementation()
    args = t.arguments("raw", "binder", t.STUDY, prepare_only=True)
    plan, tasks = t.build_plan(args)
    assert plan["execution_mode"] == "binder"
    assert plan["actor_training_execution_mode"] == "raw"
    assert plan["dataset_group"] == "diagnostic" and plan["split"] == "train"
    assert plan["phase"] == "readout" and plan["planned_episodes"] == 16
    assert [task["id"].rsplit(".", 1)[1] for task in tasks] == [
        "1796",
        "256",
        "672",
        "1273",
        "1847",
        "38",
        "201",
        "964",
    ]
    assert [job["seed"] for job in plan["jobs"]] == [202609280900] * 8 + [202609280901] * 8
    assert plan["jobs"] == t.f.jobs(tasks, "binder", "readout", 1)
    assert plan["adapter"] == t.admit_endpoint("raw")["adapter"]
    assert plan["budget_seconds"] == 5400


def test_warm_controls_resolve_without_consulting_any_a_endpoint(tmp_path, monkeypatch):
    t = implementation()
    stage = t.load_file("transfer_test_fresh_stage", t.f.HERE / "stage.py")

    def forbidden(_):
        raise AssertionError("warm readout consulted a trained endpoint")

    monkeypatch.setattr(stage, "endpoint", forbidden)
    for mode in ("raw", "binder"):
        args = SimpleNamespace(study=tmp_path, kind="readout", actor="warm", mode=mode, update=1)
        out, argv, reason = stage.resolve(args)
        assert out == tmp_path / mode / "readout-warm" and reason is None
        assert argv[argv.index("--checkpoint") + 1] == str(t.f.WARM)
        assert argv[argv.index("--dataset") + 1] == str(t.f.DATA / "diagnostic")


def test_unknown_cells_do_not_become_zero_successes(tmp_path):
    t = implementation()
    report_module = t.load_file("transfer_test_compare", t.HERE / "compare.py")
    result = report_module.analyze(tmp_path, warm_study=tmp_path / "absent-warm")
    assert len(result["cells"]) == 6
    assert all(not cell["available"] for cell in result["cells"].values())
    assert all(
        contrast["unknown"] == 16 and contrast["difference"] is None
        for contrast in result["rl_minus_matching_warm"].values()
    )


def test_actual_runtime_seam_preserves_tokens_and_binder_native_replay(tmp_path):
    t = implementation()
    fixture = t.load_file("transfer_test_fixture", t.HERE / "fixture.py")
    report = fixture.run(tmp_path)
    assert report["GPU_loaded"] is False and report["scientific_model_calls"] == 0
    assert report["raw"]["native_success"] == report["binder"]["native_success"] == 1
    assert report["binder"]["bad_requested_ingredients_preserved"] is True
    assert report["raw"]["first_prompt_matches_qualified_b"] is True
    assert report["binder"]["first_prompt_matches_qualified_b"] is True


def test_scripted_fixture_is_never_admitted_as_a_scientific_readout(tmp_path):
    t = implementation()
    report_module = t.load_file("transfer_test_fixture_rejection", t.HERE / "compare.py")
    t.f.c.save(tmp_path / "SUMMARY.json", dict(complete=True))
    t.f.c.save(tmp_path / "PLAN.json", dict(fixture_only=True))
    with pytest.raises(ValueError, match="scripted fixture"):
        report_module.load_cell(tmp_path, "raw", "raw")


def test_paired_contrast_has_correct_direction_and_task_weighting():
    t = implementation()
    report_module = t.load_file("transfer_test_arithmetic", t.HERE / "compare.py")
    keys = [(str(i), repeat) for i in range(8) for repeat in (0, 1)]
    warm = dict.fromkeys(keys, 0)
    trained = {key: int(key[0] == "0") for key in keys}
    result = report_module.contrast(
        dict(warm=warm, trained=trained), dict(trained=1, warm=-1), keys
    )
    assert result["difference"] == 0.125
    assert (result["wins"], result["losses"], result["ties"]) == (2, 0, 14)
    trained[("0", 0)] = None
    partial = report_module.contrast(
        dict(warm=warm, trained=trained), dict(trained=1, warm=-1), keys
    )
    assert partial["difference"] is None and partial["unknown"] == 1
