"""Focused CPU reward/credit/lineage fixtures; no model weights or GPU are loaded."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


def batch():
    episodes, calls = [], {}
    for task in range(8):
        for repeat in range(4):
            eid = f"task{task}-r{repeat}"
            score = [1, 1, 0, 0][repeat] if task == 0 else 1
            errors = [0, 6, 2, 8][repeat] if task == 0 else 0
            ids = [f"{eid}-c{i}" for i in range(8)]
            episodes.append(
                dict(
                    episode_id=eid,
                    task_id=f"textcraft_synth.train.{task}",
                    repeat=repeat,
                    observed=True,
                    native_score=score,
                    global_calls=8,
                    call_ids=ids,
                    errors={"native_action_error": errors},
                    status="finished",
                )
            )
            for cid in ids:
                calls[cid] = dict(
                    call_id=cid,
                    episode_id=eid,
                    available=True,
                    input_token_ids=[0, 1],
                    output_token_ids=[2, 3],
                    # Executed ingredients must never become sampled-action likelihood targets.
                    text='{"action":"craft","ingredients":{"wrong":1}}',
                    request=dict(
                        input_token_ids=[0, 1],
                        task_id=f"textcraft_synth.train.{task}",
                        sampling=dict(temperature=0.5, top_p=1.0, top_k=0),
                        model_manifest_sha256="model",
                        adapter_sha256="warmcp23",
                        adapter_commit_sha256="warmcommit",
                    ),
                )
    return episodes, calls


def test_full_batch_actual_signed_loss_matches_hand_calculated_four_reward_group():
    import cost_reward as r
    import torch

    episodes, calls = batch()
    untouched = copy.deepcopy((episodes, calls))
    credits, table = r.derive_credits(episodes, calls)
    rows = list(table["episodes"].values())[:4]
    assert [e["composite_reward"] for e in rows] == pytest.approx([1, 0.994, -0.002, -0.008])
    assert [e["composite_advantage"] for e in rows] == pytest.approx([0.672, 0.664, -0.664, -0.672])
    assert [e["native_advantage"] for e in rows] == pytest.approx([2 / 3, 2 / 3, -2 / 3, -2 / 3])
    # Real accumulated-call objective. Only each trajectory's first call has nonzero logp.
    logps = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    first = {episodes[i]["call_ids"][0]: i for i in range(4)}
    trainer = r.f.load_legacy("train.py")
    loss = sum(
        trainer.signed_objective(
            logps[first[c.call_id] : first[c.call_id] + 1]
            if c.call_id in first
            else torch.tensor([0.0]),
            c.advantage,
        )
        for c in credits
    )
    assert loss.item() == pytest.approx(-0.08375)
    loss.backward()
    assert logps.grad.tolist() == pytest.approx([-0.021, -0.02075, 0.02075, 0.021])
    assert (episodes, calls) == untouched
    assert table["mixed_native_groups"] == table["mixed_composite_groups"] == 1
    assert len(credits) == len(calls) == 256


def test_original_sampled_token_targets_and_temperature_survive_changed_credit():
    import cost_reward as r
    import torch

    episodes, calls = batch()
    credits, _ = r.derive_credits(episodes, calls)
    record = calls[credits[0].call_id]
    logits = torch.tensor([[[0.2, -0.3, 0.7, 0.1], [0.1, 0.4, -0.1, 0.8]]], requires_grad=True)

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, use_cache, logits_to_keep):
            assert input_ids.tolist() == [[0, 1, 2]]
            assert logits_to_keep == 2 and use_cache is False
            return SimpleNamespace(logits=logits)

    loss = r.f.rl.loss_math.replay_action_loss(Model(), record, credits[0].advantage)
    selected = torch.log_softmax(logits / 0.5, dim=-1)[0, [0, 1], [2, 3]]
    expected = -0.672 * selected.sum() / 32
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert record["output_token_ids"] == [2, 3]


@pytest.mark.parametrize(
    "field,value",
    [("native_action_error", -1), ("invalid_schema", True), ("native_action_error", 9)],
)
def test_rejects_invalid_or_impossible_error_counts(field, value):
    import cost_reward as r

    episodes, calls = batch()
    episodes[0]["errors"][field] = value
    with pytest.raises(ValueError, match="count|physical"):
        r.derive_credits(episodes, calls)


def test_success_order_is_strict_and_rejected_action_is_not_silently_charged():
    import cost_reward as r

    episode = dict(
        native_score=1,
        global_calls=96,
        call_ids=list(range(96)),
        errors={"invalid_schema": 48, "native_action_error": 48},
    )
    assert r.reward_row(episode)["composite_reward"] == pytest.approx(0.904)
    episode.update(native_score=0, errors={"rejected_action": 96})
    assert r.reward_row(episode)["composite_reward"] == 0
    episode.update(global_calls=97, call_ids=list(range(97)))
    with pytest.raises(ValueError, match="physical"):
        r.reward_row(episode)


def test_all_failure_cost_variation_is_visible_without_relabeling_native_successes():
    import cost_reward as r

    episodes, calls = batch()
    for episode in episodes[:4]:
        episode["native_score"] = 0
    credits, table = r.derive_credits(episodes, calls)
    assert table["mixed_native_groups"] == 0
    assert table["mixed_composite_groups"] == table["all_failure_cost_varying_groups"] == 1
    assert sum(e["native_score"] for e in table["episodes"].values()) == 28
    assert any(c.advantage > 0 for c in credits if c.episode_id == episodes[0]["episode_id"])


def test_unfinished_collection_is_conditional_skip_not_training_failure(tmp_path):
    import cost_reward as r

    assert r.collection_ready(tmp_path) is False
    r.f.c.save(tmp_path / "SUMMARY.json", {"complete": False, "failure": "cap"})
    assert r.collection_ready(tmp_path) is False


def test_real_saved_batch_loader_keeps_audited_native_scores_and_only_replaces_credit(tmp_path):
    import cost_reward as r
    import train_cost

    episodes, calls = batch()
    directory, output = tmp_path / "collection", tmp_path / "trained"
    jobs = [{"episode_id": e["episode_id"]} for e in episodes]
    r.f.c.save(directory / "PLAN.json", {"jobs": jobs, "source_sha256": {}})
    r.f.c.save(directory / "SUMMARY.json", {"complete": True, "failure": None})
    receipts = {}
    for episode in episodes:
        path = directory / "episodes" / (episode["episode_id"] + ".json")
        r.f.c.save(path, episode)
        receipts[str(path)] = r.f.sha(path)
    for cid, call in calls.items():
        path = directory / "calls" / (cid + ".json")
        r.f.c.save(path, call)
        receipts[str(path)] = r.f.sha(path)
        r.f.c.save(
            directory / "generation-logps" / (cid + ".json"),
            dict(call_sha256=r.f.sha(path), logps=[-1.0, -2.0], token_entropies=[0.8, 0.9]),
        )
    r.f.c.save(
        directory / "NATIVE-AUDIT.json",
        dict(
            observed=32,
            successes=30,
            mixed_groups=1,
            groups={},
            physical_cost={"calls": 256},
            first_response_diversity={},
            receipt_sha256=receipts,
            audits={
                e["episode_id"]: dict(
                    replayed=True, native_score=e["native_score"], errors=e["errors"], calls=8
                )
                for e in episodes
            },
        ),
    )
    trainer = train_cost.configure_trainer(output)
    loaded, credits, captured, admission = trainer.load_batch(
        directory, {"collection_plan_sha256": r.f.sha(directory / "PLAN.json")}
    )
    assert loaded == calls and len(captured) == 256
    assert credits[0].advantage == pytest.approx(0.672)
    assert admission["successes"] == 30 and admission["mixed_native_groups"] == 1
    assert admission["mixed_composite_groups"] == 1
    table = r.f.read(output / "REWARD-TABLE.json")
    assert table["native_labels_modified"] is False
    assert all(r.f.sha(Path(path)) == digest for path, digest in receipts.items())


def test_behavior_distinguishes_failed_early_finish_from_success_and_cap():
    import analyze_cost as analysis

    episode = dict(native_score=0, global_calls=1, call_ids=["a"], errors={}, status="finished")
    calls = {"a": {"text": '{"action":"finish","message":"done"}'}}
    result = analysis.behavior(episode, calls)
    assert result["unsuccessful_first_call_finish"] == 1
    assert result["unsuccessful_finish_within_four_calls"] == 1
    episode["native_score"] = 1
    assert analysis.behavior(episode, calls)["unsuccessful_explicit_finish"] == 0
    episode.update(native_score=0, status="global_call_cap")
    assert analysis.behavior(episode, calls)["unsuccessful_finish_within_four_calls"] == 0
    episode["status"] = "finished"
    calls["a"]["text"] = '{"action":"get_info","items":["x","y","x"]}'
    result = analysis.behavior(episode, calls)
    assert result["get_info_calls"] == 1 and result["distinct_requested_info_items"] == 2


def test_missing_endpoints_remain_unknown_without_loading_gpu(tmp_path, monkeypatch):
    import stage_cost

    monkeypatch.setattr(stage_cost.r, "STUDY", tmp_path)
    output, argv, reason = stage_cost.resolve("raw", "readout")
    assert output == tmp_path / "raw/readout-0001"
    assert argv is None and "No usable" in reason
