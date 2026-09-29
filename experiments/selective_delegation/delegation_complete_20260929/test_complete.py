"""Narrow contracts: no32-response censoring, both roots audited, unknowns preserved."""

import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).with_name(name + ".py")
    assert path.exists(), "complete-goal component missing"
    spec = importlib.util.spec_from_file_location("complete_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_client_accepts_after32_but_retains_delegate_refusal_guard(tmp_path):
    c = load("complete")
    plan, _, _, tokenizer, collector, _ = c.build("fixed", tmp_path, fixture=True)
    source = c.ROOT / "textcraft-decomp-flexible-fixed-20260928-001/calls"
    saved = c.read(source / "decomp-val494-fixed-r0-c000.json")
    client = collector.NativeClient.__new__(collector.NativeClient)
    client.returned, client.tokenizer = 150, tokenizer
    assert client.ids(saved["request"]["prompt"]) == saved["input_token_ids"]
    prompt, control = saved["request"]["prompt"].rsplit(c.flex.routing.MARKER, 1)
    control = dict(json.loads(control), abort_requested=True)
    with pytest.raises(c.flex.base.AdmissionStop, match="two instruction"):
        client.ids(prompt + c.flex.routing.MARKER + json.dumps(control))
    assert plan["max_global_calls"] == 96


def test_all_three_arms_schedule_the_same_two_new_seed_slots(tmp_path):
    c = load("complete")
    slots = []
    for mode in ("flat", "fixed", "adaptive"):
        p, tasks, *_ = c.build(mode, tmp_path / mode, fixture=True)
        slots.append([(j["task_id"], j["seed"], j["repeat"]) for j in p["jobs"]])
        assert p["planned_episodes"] == 2 and p["budget_seconds"] == 1200
        assert p["max_global_output_tokens"] == p["input_plus_output_limit"] == 8192
        assert p["admission_response_cap"] is None and p["instruction_rejection_cap"] == 2
        assert tasks[0]["id"] == "textcraft_synth.val.494"
    assert (
        slots
        == [
            [("textcraft_synth.val.494", 2026092901, 0), ("textcraft_synth.val.494", 2026092902, 1)]
        ]
        * 3
    )


def test_saved_prefixes_and_both_completed_root_child_trees_replay(tmp_path):
    fixture = load("fixture")
    report = fixture.run(tmp_path / "fixture")
    assert report["scientific_model_calls"] == 0
    assert len(report["saved_screen_prefixes"]) == 3
    for case in report["saved_screen_prefixes"]:
        assert case["matched_saved_calls"] == 64
        assert case["observed"] == case["full_native_replays"] == 2
        assert case["physical_cost"]["calls"] == 66
    success = report["scripted_success"]
    assert success["successes"] == success["full_native_replays"] == 2
    assert success["role_cost"]["child"]["calls"] > 0
    directory = Path(success["output"])
    plan = fixture.c.read(directory / "PLAN.json")
    # A real mutation in the SECOND root must fail independent transition replay.
    node = directory / "nodes" / (plan["jobs"][1]["episode_id"] + "-n1.json")
    value = fixture.c.read(node)
    value["final_inventory"]["raw_a1"] = value["final_inventory"].get("raw_a1", 0) + 1
    node.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="state/history|inventory"):
        fixture.audit.analyze(directory, terminal=False)


def test_pairing_keeps_unknown_and_native_success_direction():
    compare = load("compare")
    left = {0: 0, 1: 1}
    right = {0: 1, 1: 1}
    result = compare.paired(left, right)
    assert (result["wins"], result["losses"], result["ties"], result["difference"]) == (
        1,
        0,
        1,
        0.5,
    )
    result = compare.paired(left, {0: None, 1: 1})
    assert result["unknown"] == 1 and result["difference"] is None
