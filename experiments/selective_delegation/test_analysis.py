"""Scientific regression fixtures for repeated-arm analysis."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("analysis.py")
    assert path.exists(), "analysis implementation is missing"
    spec = importlib.util.spec_from_file_location("delegation_analysis", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path, rewards, failed=()):
    output = tmp_path / "run"
    for name in ("episodes", "checkpoints", "calls"):
        (output / name).mkdir(parents=True)
    ids = list(rewards) + list(failed)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps({"id": key, "split": "train"}) + "\n" for key in ids))
    (output / "PLAN.json").write_text(
        json.dumps(
            {
                "case_ids": ids,
                "repeats": 3,
                "arms": ["finish", "reconsider", "targeted", "decompose"],
            }
        )
    )
    for key in ids:
        initial = {
            "case_id": key,
            "state": None if key in failed else {"answer": "a"},
            "error": "checkpoint_invalid" if key in failed else None,
            "call_id": key + "-cp",
        }
        (output / "checkpoints" / (key + ".json")).write_text(json.dumps(initial))
        call = {
            "call_id": key + "-cp",
            "available": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        (output / "calls" / (key + "-cp.json")).write_text(json.dumps(call))
        if key in failed:
            continue
        for arm, scores in rewards[key].items():
            for repeat, score in enumerate(scores):
                identity = f"{key}-{arm}-{repeat}"
                call = {
                    "call_id": identity,
                    "available": True,
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3},
                }
                (output / "calls" / (identity + ".json")).write_text(json.dumps(call))
                episode = {
                    "case_id": key,
                    "arm": arm,
                    "repeat": repeat,
                    "available": True,
                    "valid": True,
                    "correct": bool(score),
                    "f1": score,
                    "call_ids": [key + "-cp", identity],
                }
                (output / "episodes" / (identity + ".json")).write_text(json.dumps(episode))
    return output, cases


def scores(**overrides):
    return {
        arm: overrides.get(arm, [0, 0, 0])
        for arm in ("finish", "reconsider", "targeted", "decompose")
    }


def test_repeatable_parent_heterogeneity_beats_fixed_policy(tmp_path):
    a = module()
    output, cases = fixture(
        tmp_path, {"p": scores(targeted=[1, 1, 1]), "q": scores(decompose=[1, 1, 1])}
    )
    report = a.analyze(output, cases, bootstrap_samples=200)
    cv = report["cross_validation"]
    assert cv["selected"]["em"] == 1
    assert cv["fixed"]["em"] == 0.5
    assert cv["selected_minus_fixed"]["em"]["estimate"] == 0.5
    assert "same-parent" in cv["interpretation"]
    assert report["paired_vs_finish"]["targeted"]["em"]["estimate"] == 0.5


def test_excluded_repeat_never_selects_its_own_winning_arm(tmp_path):
    a = module()
    output, cases = fixture(
        tmp_path, {"p": scores(reconsider=[1, 0, 0], targeted=[0, 1, 0], decompose=[0, 0, 1])}
    )
    report = a.analyze(output, cases, bootstrap_samples=50)
    assert report["cross_validation"]["selected"]["em"] == 0
    assert report["cross_validation"]["selected_minus_finish"]["em"]["estimate"] == 0
    assert not report["decision"]["repeatable_headroom_signal"]


def test_failed_checkpoint_stays_in_denominator_and_missing_blocks_readiness(tmp_path):
    a = module()
    output, cases = fixture(tmp_path, {"p": scores(finish=[1, 1, 1])}, failed=["bad"])
    report = a.analyze(output, cases, bootstrap_samples=50)
    assert report["arms"]["finish"]["em"] == 0.5
    assert report["arms"]["finish"]["checkpoint_failures"] == 3
    assert report["cross_validation"]["parents"] == 2
    (output / "episodes" / "p-finish-0.json").unlink()
    report = a.analyze(output, cases, bootstrap_samples=50)
    assert report["arms"]["finish"]["missing_episodes"] == 1
    assert not report["decision"]["collection_complete"]
    assert report["cross_validation"]["parents"] == 1


def test_physical_cost_counts_checkpoint_once_and_unknowns_are_explicit(tmp_path):
    a = module()
    output, cases = fixture(tmp_path, {"p": scores()})
    call_path = output / "calls" / "p-targeted-0.json"
    call = json.loads(call_path.read_text())
    call.update(available=False, usage={})
    call_path.write_text(json.dumps(call))
    report = a.analyze(output, cases, bootstrap_samples=50)
    assert report["physical_cost"]["calls"] == 13
    assert report["physical_cost"]["unknown_usage_calls"] == 1
    assert report["arms"]["finish"]["deployed_cost"]["calls"] == 6
    assert report["arms"]["finish"]["deployed_cost"]["prompt_tokens"] == 90
    assert report["arms"]["targeted"]["deployed_cost"]["unknown_usage_calls"] == 1


def test_reports_are_immutable(tmp_path):
    a = module()
    output, cases = fixture(tmp_path, {"p": scores()})
    report = a.analyze(output, cases, bootstrap_samples=20)
    path = tmp_path / "report.json"
    a.write_report(report, path)
    assert path.with_suffix(".md").exists()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        a.write_report(report, path)
    assert path.read_bytes() == before


def test_shared_atomic_components_are_disclosed_as_dependent_parents(tmp_path):
    a = module()
    output, cases = fixture(tmp_path, {"p": scores(), "q": scores(), "r": scores()})
    rows = [json.loads(line) for line in cases.read_text().splitlines()]
    for row, components in zip(rows, [["a"], ["a", "b"], ["b"]], strict=True):
        row["metadata"] = {"component_ids": components}
    cases.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = a.analyze(output, cases, bootstrap_samples=20)
    assert report["component_overlap"]["connected_parent_clusters"] == 1
    assert report["component_overlap"]["parents_sharing_components"] == 3
    assert report["component_overlap"]["largest_cluster"] == 3
    assert "conditional" in report["bootstrap"]["interpretation"]


def test_transport_contamination_blocks_positive_quality_flag(tmp_path):
    a = module()
    rewards = {
        str(i): scores(**{("targeted" if i % 2 else "decompose"): [1, 1, 1]}) for i in range(16)
    }
    output, cases = fixture(tmp_path, rewards)
    clean = a.analyze(output, cases, bootstrap_samples=100)
    assert clean["decision"]["repeatable_headroom_signal"]
    path = output / "episodes" / "0-finish-0.json"
    row = json.loads(path.read_text())
    row.update(available=False, valid=False, error="final_transport_failure")
    path.write_text(json.dumps(row))
    failed = a.analyze(output, cases, bootstrap_samples=100)
    assert not failed["decision"]["repeatable_headroom_signal"]
    assert failed["failures"]["transport_episodes"] == 1


def format_fixture(tmp_path):
    output, cases = fixture(tmp_path, {"p": scores(), "q": scores()}, failed=["bad"])
    followup = tmp_path / "format"
    for name in ("pairs", "calls"):
        (followup / name).mkdir(parents=True)

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source_hashes, plan_episodes = {}, {}
    for path in sorted((output / "episodes").glob("*.json")):
        row = json.loads(path.read_text())
        identity = path.stem
        row.update(episode_id=identity, seed=42)
        if row["arm"] != "finish":
            helper = {
                "call_id": identity + "-helper",
                "available": True,
                "usage": {"prompt_tokens": 6, "completion_tokens": 1},
            }
            helper_path = output / "calls" / (helper["call_id"] + ".json")
            helper_path.write_text(json.dumps(helper))
            row["call_ids"].insert(1, helper["call_id"])
        path.write_text(json.dumps(row))
        pins = {str(path): sha(path)}
        requests = {}
        for call_id in row["call_ids"]:
            call_path = output / "calls" / (call_id + ".json")
            call = json.loads(call_path.read_text())
            call["request"] = {
                "sampling_params": {"seed": 42, "temperature": 0.5, "max_tokens": 128}
            }
            call["request_digest"] = digest(call["request"])
            call_path.write_text(json.dumps(call))
            pins[str(call_path)] = sha(call_path)
            requests[call_id] = call["request_digest"]
        source_hashes.update(pins)
        new = {
            "call_id": identity + "-short-final",
            "available": True,
            "prompt": "Short phrase.",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "request": {"sampling_params": {"seed": 42, "temperature": 0.5, "max_tokens": 128}},
        }
        new["request_digest"] = digest(new["request"])
        (followup / "calls" / (new["call_id"] + ".json")).write_text(json.dumps(new))
        pair = {
            "episode_id": identity,
            "case_id": row["case_id"],
            "arm": row["arm"],
            "repeat": row["repeat"],
            "seed": 42,
            "available": True,
            "old": {"valid": True, "correct": False, "f1": 0.0},
            "new": {"valid": True, "correct": True, "f1": 1.0},
            "reused_call_ids": row["call_ids"][:-1],
            "new_call_id": new["call_id"],
            "source_hashes": pins,
            "source_request_digests": requests,
        }
        (followup / "pairs" / (identity + ".json")).write_text(json.dumps(pair))
        plan_episodes[identity] = {
            "prompt_digest": digest(new["prompt"]),
            "seed": 42,
            "temperature": 0.5,
            "max_tokens": 128,
            "source_request_digests": requests,
        }
    (followup / "PLAN.json").write_text(
        json.dumps(
            {
                "source_output": str(output),
                "source_plan_sha256": sha(output / "PLAN.json"),
                "cases_sha256": sha(cases),
                "source_hashes": source_hashes,
                "phrase_instruction": "Short phrase.",
                "episodes": plan_episodes,
                "source_exclusions": {"source_episode_missing": 12},
            }
        )
    )
    return output, cases, followup


def test_format_overlay_replaces_final_and_separates_physical_acquisition_cost(tmp_path):
    a = module()
    output, cases, followup = format_fixture(tmp_path)
    report = a.analyze(output, cases, format_output=followup, bootstrap_samples=20)
    assert report["arms"]["targeted"]["em"] == pytest.approx(2 / 3)
    assert report["arms"]["targeted"]["checkpoint_failures"] == 3
    assert report["physical_cost_split"]["original_acquisition"]["calls"] == 45
    assert report["physical_cost_split"]["new_final_only"]["calls"] == 24
    assert report["physical_cost"]["calls"] == 69
    # 6 valid attempts*(checkpoint12+helper7+newfinal8) + 3 failed checkpoints*12.
    assert report["arms"]["targeted"]["deployed_cost"]["known_total_tokens"] == 198
    assert report["format_comparison"]["arms"]["finish"]["new_minus_old"]["em"]["estimate"] == 1
    assert report["decision"]["format_checked"]


def test_missing_format_pair_is_missing_not_old_final_fallback(tmp_path):
    a = module()
    output, cases, followup = format_fixture(tmp_path)
    (followup / "pairs" / "p-targeted-0.json").unlink()
    report = a.analyze(output, cases, format_output=followup, bootstrap_samples=20)
    assert report["arms"]["targeted"]["recorded_episodes"] == 5
    assert report["arms"]["targeted"]["missing_episodes"] == 1
    assert report["cross_validation"]["parents"] == 2
    assert not report["decision"]["training_candidate"]


def test_format_rejects_changed_source_and_incorrect_cases_identity(tmp_path):
    a = module()
    output, cases, followup = format_fixture(tmp_path)
    call = output / "calls" / "p-targeted-0-helper.json"
    original = call.read_bytes()
    call.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="source hash"):
        a.analyze(output, cases, format_output=followup, bootstrap_samples=20)
    call.write_bytes(original)
    plan_path = followup / "PLAN.json"
    plan = json.loads(plan_path.read_text())
    plan["cases_sha256"] = "wrong"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="cases"):
        a.analyze(output, cases, format_output=followup, bootstrap_samples=20)
