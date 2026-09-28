import importlib.util
import json

import pytest


def fresh():
    assert importlib.util.find_spec("fresh_common") is not None
    import fresh_common

    return fresh_common


def dataset(tmp_path, group="train"):
    f = fresh()
    rows = [dict(id=f"textcraft_synth.train.{i}", goal="Craft", misc={}) for i in range(8)]
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    f.c.save(
        tmp_path / "MANIFEST.json",
        dict(
            split="train",
            group=group,
            task_ids=[r["id"] for r in rows],
            tasks_sha256=f.sha(tmp_path / "tasks.jsonl"),
            world_seed=42,
            world_sha256="world",
            all_native_feasible=True,
            initial_context_qualified=True,
        ),
    )
    return rows


def test_actual_group_manifest_is_bound_and_diagnostic_cannot_train(tmp_path):
    f = fresh()
    rows = dataset(tmp_path, "diagnostic")
    manifest, loaded = f.load_dataset(tmp_path, "readout")
    assert loaded == rows and manifest["group"] == "diagnostic"
    with pytest.raises(ValueError, match="training group"):
        f.load_dataset(tmp_path, "collect")


def test_qualified_original_bytes_are_required_not_just_matching_ids(tmp_path):
    f = fresh()
    dataset(tmp_path)
    with (tmp_path / "tasks.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="task bytes"):
        f.load_dataset(tmp_path, "collect")


def test_fresh_training_and_diagnostic_seeds_are_paired_but_disjoint():
    f = fresh()
    rows = [dict(id=f"textcraft_synth.train.{i}") for i in range(8)]
    raw = f.jobs(rows, "raw", "collect", 1)
    binder = f.jobs(rows, "binder", "collect", 1)
    diagnostic = f.jobs(rows, "raw", "readout", 4)
    assert [(j["task_id"], j["seed"]) for j in raw] == [(j["task_id"], j["seed"]) for j in binder]
    assert not {j["seed"] for j in raw} & {j["seed"] for j in diagnostic}
    assert len(raw) == 32 and len(diagnostic) == 16


def test_fresh_native_public_trace_replays_with_original_bad_ingredient_targets(tmp_path):
    import copy
    import time

    import torch
    from transformers import AutoTokenizer

    f = fresh()
    manifest, tasks = f.load_dataset(f.DATA / "train", "collect")
    task = tasks[0]
    trace = f.read(f.DATA / "train/native" / (task["id"] + ".json"))["trace"]
    actions = [copy.deepcopy(step["action"]) for step in trace]
    changed = next(i for i, a in enumerate(actions) if a["action"] == "craft")
    requested = actions[changed]
    requested["ingredients"] = {k: v + 1 for k, v in requested["ingredients"].items()}
    answers = iter(json.dumps(a) for a in actions)
    tokenizer = AutoTokenizer.from_pretrained(f.c.BASE, local_files_only=True)

    class Model:
        device = torch.device("cpu")

        def parameters(self):
            return []

        def generate(self, **kwargs):
            tokens = tokenizer.encode(next(answers), add_special_tokens=False)
            return torch.cat(
                [kwargs["input_ids"], torch.tensor([tokens + [tokenizer.eos_token_id]])], dim=1
            )

    job = f.jobs(tasks, "binder", "collect", 1)[0]
    plan = dict(
        model=str(f.c.BASE),
        model_manifest_sha256="fixture",
        adapter=None,
        profile="original",
        jobs=[job],
        planned_episodes=1,
        dataset_manifest_sha256=f.GROUP_SHA["train"],
    )
    f.c.save(tmp_path / "PLAN.json", plan)
    client = f.c.NativeClient(
        Model(), tokenizer, tmp_path, time.time() + 60, "fixture", f.sha(tmp_path / "PLAN.json")
    )
    native = f.load_legacy("collect.py")
    assert native.p is f
    collector, _ = native.p.implementation("binder")
    world = f.c.bridge.load_world()
    result = collector.episode(task, job, client, world, tmp_path, time.time() + 60)
    assert result["observed"] and result["native_score"] == 1
    replay = f.audit_collection(tmp_path, plan, [task], tokenizer, world, "binder")
    assert replay["successes"] == 1
    call = f.read(tmp_path / "calls" / f"{job['episode_id']}-c{changed:03d}.json")
    _, target = f.rl.loss_math.causal_inputs(call)
    assert json.loads(tokenizer.decode(target, skip_special_tokens=True)) == requested
    first = f.read(tmp_path / "calls" / f"{job['episode_id']}-c000.json")
    import hashlib

    assert (
        hashlib.sha256(first["request"]["prompt"].encode()).hexdigest()
        == (manifest["audits"][0]["initial_prompt_sha256"])
    )


def test_failed_arm_stops_future_optimization_but_keeps_independent_warm_control(tmp_path):
    from types import SimpleNamespace

    from stage import resolve

    f = fresh()
    f.c.save(
        tmp_path / "raw/train-0001/SUMMARY.json",
        dict(endpoint_usable=False, failure="numerical replay mismatch"),
    )
    args = SimpleNamespace(study=tmp_path, kind="collect", mode="raw", actor="rl", update=2)
    _, argv, reason = resolve(args)
    assert argv is None and "Previous update" in reason
    args.kind, args.actor = "readout", "warm"
    _, argv, reason = resolve(args)
    assert argv is not None and str(f.WARM) in argv and reason is None


def test_extra_sft_uses_t1_target_token_mean_not_rl_temperature():
    import math
    from types import SimpleNamespace

    import torch
    from extra_sft import objective

    logits = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], requires_grad=True)

    class Model:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [[4, 5, 6]]
            assert kwargs["logits_to_keep"] == 2
            return SimpleNamespace(logits=logits)

    loss = objective(Model(), dict(input_ids=[4, 5, 6], target_ids=[1, 0]), 4)
    assert float(loss.detach()) == pytest.approx((2 * math.log1p(math.e) - 1) / 4)
    loss.backward()
    assert bool(torch.isfinite(logits.grad).all())


def test_all_prospective_checkpoint_readouts_and_independent_controls_are_queued(tmp_path):
    from launch_campaign import make_jobs

    jobs = make_jobs(tmp_path, 4)
    assert len(jobs) == 33
    for mode in ("raw", "binder"):
        for update in range(1, 5):
            assert any(
                j["output"] == str(tmp_path / mode / f"readout-{update:04d}")
                for j in jobs
                if "output" in j
            )
        assert any(j.get("output") == str(tmp_path / mode / "readout-warm") for j in jobs)
        assert any(j.get("output") == str(tmp_path / "sft" / mode / "readout-0001") for j in jobs)


def test_absent_diagnostic_cells_are_unknown_not_fake_zero_successes(tmp_path):
    f = fresh()
    spec = importlib.util.spec_from_file_location("fresh_report_test", f.HERE / "compare.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    result = report.analyze(tmp_path, 1)
    assert result["rl_minus_warm"]["raw"]["unknown"] == 16
    assert result["rl_minus_warm"]["raw"]["difference"] is None
    assert result["binder_minus_raw_learning_gain"]["difference"] is None
    assert all(not cell["available"] for cell in result["cells"].values())
