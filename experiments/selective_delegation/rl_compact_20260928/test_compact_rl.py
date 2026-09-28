"""New compact-RL seam tests; scripted CPU tensors, never pretrained model inference."""

from types import SimpleNamespace

import pytest


def test_compact_schedule_pairs_fresh_tasks_and_seeds_without_relabeling_interface():
    import compact_common as c

    _, rows = c.f.load_dataset(c.f.DATA / "train", "collect")
    jobs = c.jobs(rows, "collect")
    assert len(jobs) == 32
    assert sorted({j["seed"] for j in jobs}) == list(range(202609280110, 202609280114))
    assert [j["task_id"] for j in jobs[:8]] == [r["id"] for r in rows]
    assert all(j["condition"] == "compact_observed_collect" for j in jobs)
    diagnostic = c.jobs(rows, "readout")
    assert len(diagnostic) == 16
    assert sorted({j["seed"] for j in diagnostic}) == [202609280900, 202609280901]


def test_missing_real_sft_checkpoint_cannot_become_a_bound_adapter(tmp_path, monkeypatch):
    import compact_common as c

    monkeypatch.setattr(c, "WARM", tmp_path / "checkpoint-0023")
    with pytest.raises(c.PendingCheckpoint):
        c.warm_binding()
    assert not list(tmp_path.glob("**/PLAN.json"))


def test_warm_readout_does_not_depend_on_rl_training_success(tmp_path, monkeypatch):
    import compact_common as c
    import compact_stage as stage

    monkeypatch.setattr(c, "STUDY", tmp_path)
    c.f.c.save(tmp_path / "train-0001/CONDITIONAL-SKIP.json", {"reason": "flat rewards"})
    output, argv, reason = stage.resolve("warm")
    assert output == tmp_path / "readout-warm" and reason is None
    assert str(c.WARM) in argv and "readout" in argv
    output, argv, reason = stage.resolve("updated")
    assert output == tmp_path / "readout-0001" and argv is None
    assert "endpoint" in reason


def test_training_rejects_noncompact_or_diagnostic_collections(tmp_path):
    import compact_common as c

    for schema, group in (
        ("textcraft-fresh-train-collection-20260928-v1", "train"),
        ("textcraft-compact-rl-collection-20260928-v1", "diagnostic"),
    ):
        directory = tmp_path / group
        c.f.c.save(directory / "PLAN.json", dict(schema=schema, dataset_group=group))
        args = SimpleNamespace(collection=directory, output=tmp_path / "trained", hours=2)
        with pytest.raises(ValueError, match="compact groupA"):
            c.prepare_training(args)


def test_saved_compact_responses_replay_natively_and_retain_original_token_targets(tmp_path):
    import compact_rl_fixture as fixture

    result = fixture.check(tmp_path / "native-fixture")
    assert result["native_score"] == 1 and result["replayed"] is True
    assert result["errors"]["invalid_schema"] == 1
    assert result["errors"]["rejected_action"] == 1
    assert result["errors"]["native_action_error"] == 1
    assert result["metric_records"] == result["calls"]
    assert result["compact_success_targets_preserved"] is True
    assert result["invalid_old_format_tokens_preserved"] is True
    assert result["generation_replay_gap"] < 1e-6
    assert result["all_original_tokens_receive_signed_credit"] is True
    assert result["positive_token_derivative"] == pytest.approx(-1 / 48)
    assert result["negative_token_derivative"] == pytest.approx(1 / 48)
    assert result["scientific_model_calls"] == 0


def test_prepared_contract_keeps_unknown_weights_out_of_scientific_plans():
    import prepare_compact_rl as prepare

    contract = prepare.describe()
    assert contract["checkpoint_pending_until_authenticated_complete"] is True
    assert "adapter" not in contract
    assert len([j for j in contract["jobs"] if j.get("output")]) == 4
    assert contract["execution_mode"] == "compact_observed"
    assert contract["jobs"][2]["name"] == "compact-rl-warm"


def test_missing_compact_or_baseline_endpoints_stay_unknown(tmp_path, monkeypatch):
    import compact_common as c
    import compact_compare as compare

    monkeypatch.setattr(c, "STUDY", tmp_path / "compact")
    monkeypatch.setattr(c, "BASELINE", tmp_path / "baseline")
    result = compare.analyze()
    for value in result["within_interface_native_success_gain"].values():
        assert value["known"] == 0 and value["unknown"] == 16
        assert value["difference"] is None
    assert result["compact_minus_full_binder_learning_gain"]["difference"] is None


def test_difference_of_gains_does_not_confuse_warm_score_with_learning():
    import compact_compare as compare

    keys = [("task", 0), ("task", 1)]
    compact = compare.difference(dict.fromkeys(keys, 1), dict.fromkeys(keys, 1), keys)
    binder = compare.difference(dict.fromkeys(keys, 0), dict.fromkeys(keys, 1), keys)
    assert compare.difference(binder, compact, keys) == dict.fromkeys(keys, -1)
    assert compare.difference(binder, {}, keys) == dict.fromkeys(keys, None)
