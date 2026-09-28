"""Saved audit-schema seams, not native runtime or model tests."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("readout.py")
    assert path.exists(), "controls reader has not been implemented"
    spec = importlib.util.spec_from_file_location("controls_reader_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, name, *, family="qwen", score=0, unknown=0, reordered=False):
    directory = tmp_path / name
    training = tmp_path / (name + "-training")
    checkpoint = training / "checkpoint-0023"
    train_plan = {"seed": 2026092208, "rows_sha256": "labels", "model": "base"}
    train_sha = save(training / "PLAN.json", train_plan)
    seeds = [2026092204] if family == "phi" else [2026092204, 2026092205]
    jobs = [
        dict(episode_id=f"{name}-{i}-{s}", task_id=f"task{i}", seed=s, repeat=k, policy="flat")
        for i in range(8)
        for k, s in enumerate(seeds)
    ]
    expected = copy.deepcopy(jobs)
    if reordered:
        jobs.reverse()
        for i, job in enumerate(jobs):
            job["episode_id"] = f"reordered-{i}"
    plan = dict(
        schema="phi-teaching-assistance-fixed-panel-20260928-v1"
        if family == "phi"
        else "textcraft-paired-breadth-v1",
        teacher="discovery" if family == "phi" else "public",
        assistance="raw",
        execution_mode="raw",
        world_seed=42,
        training_seed=2026092208,
        jobs=jobs,
        planned_episodes=len(jobs),
        tasks_sha256="tasks",
        manifest_sha256="manifest",
        world_sha256="world",
        model_manifest_sha256="model",
        sampling={"temperature": 0.5},
        max_global_calls=96,
        max_global_output_tokens=8192,
        max_new_tokens=256,
        input_plus_output_limit=8192,
        truncation=False,
        budget_seconds=2700,
        model="base",
        fixed_adapter=str(checkpoint),
        training_plan_sha256=train_sha,
        adapter=dict(
            path=str(checkpoint),
            sha256="adapter",
            commit_sha256="commit",
            state={"step": 23},
            training_plan_sha256=train_sha,
            training_rows_sha256="labels",
        ),
    )
    plan_sha = save(directory / "PLAN.json", plan)
    outcomes = [
        dict(
            task_id=j["task_id"],
            observed=i >= unknown,
            replayed=i >= unknown,
            native_score=score if i >= unknown else None,
            calls=3,
            output_tokens=60,
            errors={"native_action_error": 1},
        )
        for i, j in enumerate(jobs)
    ]
    if family == "phi":
        audit = dict(
            schema="phi-textcraft-native-audit-20260928-v1",
            passed=True,
            teacher="discovery",
            assistance="raw",
            world_seed=42,
            episodes=outcomes,
            observed=len(jobs) - unknown,
            unknown=unknown,
            successes=score * (len(jobs) - unknown),
            plan_sha256=plan_sha,
            physical_cost={"calls": 3 * len(jobs)},
        )
    else:
        audit = dict(
            audits={j["episode_id"]: o for j, o in zip(jobs, outcomes, strict=True)},
            groups={
                "fixture": dict(
                    planned=len(jobs),
                    observed=len(jobs) - unknown,
                    won=score * (len(jobs) - unknown),
                )
            },
            sha256={str(directory / "PLAN.json"): plan_sha},
            terminal={"ended": 1, "failure": None},
            physical_cost={"calls": 3 * len(jobs)},
            unresolved_starts=[],
            calls_without_episode=[],
        )
    save(directory / ("PHI-AUDIT.json" if family == "phi" else "NATIVE-AUDIT.json"), audit)
    save(directory / "TERMINAL-fixture.json", {"ended": 1, "failure": None})
    # The legacy summary is deliberately wrong; it is not the native denominator.
    save(directory / "SUMMARY.json", {"planned": 16, "successes": 0})
    spec = dict(
        name=name,
        path=str(directory),
        family=family,
        kind="baseline",
        teacher=plan["teacher"],
        mode="raw",
        world=42,
        step=23,
        training=str(training),
        checkpoint=str(checkpoint),
        baseline_checkpoint=None,
        expected_jobs=expected,
        expected_inputs=plan,
    )
    return spec, plan, audit


def test_generic_plan_sixteen_preserves_unknown_and_ignores_summary(tmp_path):
    spec, _, _ = fixture(tmp_path, "qwen", score=1, unknown=1)
    result = module().load_cell(spec)
    assert (result["planned"], result["observed"], result["successes"], result["unknown"]) == (
        16,
        15,
        15,
        1,
    )
    assert result["success_rate_bounds"] == [15 / 16, 1]


def test_phi_audit_uses_actual_eight_slots(tmp_path):
    spec, _, _ = fixture(tmp_path, "phi", family="phi", score=1)
    result = module().load_cell(spec)
    assert (result["planned"], result["observed"], result["successes"]) == (8, 8, 8)
    assert result["success_rate_bounds"] == [1, 1]


def test_phi_audit_labels_must_match_bound_plan(tmp_path):
    spec, _, audit = fixture(tmp_path, "phi", family="phi", score=1)
    audit["teacher"] = "known"
    save(Path(spec["path"]) / "PHI-AUDIT.json", audit)
    with pytest.raises(ValueError, match="Phi audit teacher"):
        module().load_cell(spec)


def test_phi_legacy_native_summary_cannot_substitute_for_dedicated_audit(tmp_path):
    spec, _, audit = fixture(tmp_path, "phi", family="phi", score=1)
    Path(spec["path"], "PHI-AUDIT.json").unlink()
    save(Path(spec["path"]) / "NATIVE-AUDIT.json", audit)
    result = module().load_cell(spec)
    assert result["status"] == "awaiting_native_audit"
    assert result["unknown"] == 8


def test_pairing_normalizes_reordered_episode_ids_by_task_seed(tmp_path):
    left, _, _ = fixture(tmp_path, "left", score=0)
    right, _, _ = fixture(tmp_path, "right", score=1, reordered=True)
    reader = module()
    pair = reader.compare_cells(reader.load_cell(left), reader.load_cell(right))
    assert (pair["planned_pairs"], pair["wins"], pair["losses"], pair["difference"]) == (
        16,
        16,
        0,
        1,
    )


def test_interface_pair_must_have_identical_actor(tmp_path):
    left, _, _ = fixture(tmp_path, "left", family="phi")
    right, _, _ = fixture(tmp_path, "right", family="phi")
    reader = module()
    with pytest.raises(ValueError, match="changed actor"):
        reader.compare_cells(reader.load_cell(left), reader.load_cell(right), same_actor=True)


def test_pair_must_have_identical_world(tmp_path):
    left, _, _ = fixture(tmp_path, "left")
    right, _, _ = fixture(tmp_path, "right")
    reader = module()
    lcell, rcell = reader.load_cell(left), reader.load_cell(right)
    rcell["_plan"]["world_sha256"] = "different-world"
    with pytest.raises(ValueError, match="world_sha256"):
        reader.compare_cells(lcell, rcell)


def test_unknown_pair_retains_full_denominator_and_no_point_estimate(tmp_path):
    left, _, _ = fixture(tmp_path, "left", score=0)
    right, _, _ = fixture(tmp_path, "right", score=1, unknown=1)
    reader = module()
    pair = reader.compare_cells(reader.load_cell(left), reader.load_cell(right))
    assert pair["unknown_pairs"] == 1
    assert pair["difference"] is None
    assert pair["difference_bounds"] == [14 / 16, 1]


def test_absent_cell_is_unknown_with_labeled_schedule_denominator(tmp_path):
    spec, _, _ = fixture(tmp_path, "unrun")
    spec["path"] = str(tmp_path / "not_started")
    result = module().load_cell(spec)
    assert (result["planned"], result["observed"], result["unknown"]) == (16, 0, 16)
    assert result["denominator_source"] == "frozen_schedule_no_actual_PLAN"


def test_wrong_teacher_is_rejected_even_when_reference_plan_says_discovery(tmp_path):
    spec, _, _ = fixture(tmp_path, "incorrect_known_baseline")
    spec["teacher"] = "quantity_corrected_original"
    with pytest.raises(ValueError, match="teacher"):
        module().load_cell(spec)


def test_explicit_order_baseline_does_not_follow_reference_plan(tmp_path):
    specs, contrasts = module().inventory(tmp_path)
    contrast = next(c for c in contrasts if c["name"] == "order_stable_visible_minus_known_w42")
    baseline = specs[contrast["left"]]
    assert baseline["teacher"] == "quantity_corrected_original"
    assert Path(baseline["path"]).name == "textcraft-breadth-p00-w42-soriginal-corrected-raw-001"


def test_dose_original_checkpoint_cannot_be_another_teacher(tmp_path):
    spec, plan, audit = fixture(tmp_path, "dose")
    spec.update(kind="dose", baseline_checkpoint="correct_original")
    training = Path(spec["training"])
    train = json.loads((training / "PLAN.json").read_text())
    train["original"] = {"checkpoint": "wrong_original", "plan_sha256": "other"}
    sha = save(training / "PLAN.json", train)
    plan["adapter"]["training_plan_sha256"] = plan["training_plan_sha256"] = sha
    audit["sha256"][str(Path(spec["path"]) / "PLAN.json")] = save(
        Path(spec["path"]) / "PLAN.json", plan
    )
    save(Path(spec["path"]) / "NATIVE-AUDIT.json", audit)
    with pytest.raises(ValueError, match="original checkpoint"):
        module().load_cell(spec)


def test_changed_native_plan_receipt_is_rejected(tmp_path):
    spec, plan, _ = fixture(tmp_path, "changed")
    plan["jobs"][0]["seed"] += 10
    save(Path(spec["path"]) / "PLAN.json", plan)
    with pytest.raises(ValueError, match="PLAN"):
        module().load_cell(spec)


def test_dose_pair_matches_actual_original_checkpoint_receipt(tmp_path):
    baseline, _, _ = fixture(tmp_path, "baseline")
    dose, plan, audit = fixture(tmp_path, "dose")
    dose.update(
        kind="dose",
        step=46,
        checkpoint=str(Path(dose["training"]) / "checkpoint-0046"),
        baseline_checkpoint=baseline["checkpoint"],
    )
    training_path = Path(dose["training"]) / "PLAN.json"
    training = json.loads(training_path.read_text())
    training["original"] = dict(
        checkpoint=baseline["checkpoint"],
        plan_sha256=hashlib.sha256(
            Path(baseline["training"], "PLAN.json").read_bytes()
        ).hexdigest(),
        checkpoint_commit_sha256="wrong-same-path-checkpoint",
        checkpoint_files={"adapter_model.safetensors": "adapter"},
    )
    plan["adapter"].update(
        path=dose["checkpoint"],
        state={"step": 46},
        training_plan_sha256=save(training_path, training),
        original_training_plan_sha256=training["original"]["plan_sha256"],
    )
    plan["cumulative_training_updates"] = 46
    audit["sha256"][str(Path(dose["path"]) / "PLAN.json")] = save(
        Path(dose["path"]) / "PLAN.json", plan
    )
    save(Path(dose["path"]) / "NATIVE-AUDIT.json", audit)
    reader = module()
    with pytest.raises(ValueError, match="original actor"):
        reader.compare_cells(reader.load_cell(baseline), reader.load_cell(dose))


def test_watcher_evidence_ignores_raw_files_and_unchanged_receipt_bytes(tmp_path, monkeypatch):
    spec, _, audit = fixture(tmp_path, "watched")
    reader = module()
    monkeypatch.setattr(reader, "inventory", lambda root: ({"cell": spec}, []))
    cache = {}
    before = reader.evidence_digest(tmp_path, cache)
    save(Path(spec["path"]) / "calls" / "not-reader-evidence.json", {"new": True})
    save(Path(spec["path"]) / "NATIVE-AUDIT.json", audit)
    assert reader.evidence_digest(tmp_path, cache) == before
    save(Path(spec["path"]) / "TERMINAL-fixture.json", {"ended": 2, "failure": None})
    assert reader.evidence_digest(tmp_path, cache) != before


def test_watcher_changes_only_on_evidence_and_stops_at_lease(tmp_path, monkeypatch):
    spec, _, _ = fixture(tmp_path, "watched")
    reader, seconds = module(), [0]
    monkeypatch.setattr(reader, "inventory", lambda root: ({"cell": spec}, []))
    monkeypatch.setattr(reader.time, "time", lambda: seconds[0])
    monkeypatch.setenv("SLURM_JOB_END_TIME", "65")

    def sleep(duration):
        seconds[0] += duration

    monkeypatch.setattr(reader.time, "sleep", sleep)
    monkeypatch.setattr(
        reader,
        "analyze",
        lambda root: dict(cutoff_utc="fixture", cells={}, contrasts=[], scope="test"),
    )
    output = tmp_path / "snapshots"
    reader.watch(tmp_path, output, deadline=120)
    assert seconds == [65]
    assert len(list(output.glob("snapshot-*.json"))) == 1
    assert len(list(output.glob("snapshot-*.md"))) == 1
