"""Small receipt fixtures for planned-denominator planner comparisons."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def implementation():
    path = Path(__file__).with_name("analyze_planner.py")
    assert path.exists(), "planner analyzer missing"
    spec = importlib.util.spec_from_file_location("planner_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    output = tmp_path / "run"

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": parent,
                    "answer": "yes",
                    "metadata": {"answer_aliases": ["indeed"], "component_ids": [parent]},
                }
            )
            for parent in ("a", "b")
        )
    )
    save(
        output / "PLAN.json",
        {
            "case_ids": ["a", "b"],
            "repeats": 2,
            "conditions": ["base", "adapted"],
            "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
            "execution": "isolated",
        },
    )
    for parent in ("a", "b"):
        for repeat in range(2):
            for condition in ("base", "adapted"):
                if (parent, repeat, condition) == ("b", 1, "adapted"):
                    continue
                episode = f"{parent}-r{repeat}-{condition}"
                invalid = parent == "a" and condition == "base"
                correct = (parent == "a" and condition == "adapted") or (
                    parent == "b" and condition == "base"
                )
                ids = []
                for role in ("root",) if invalid else ("root", "final"):
                    cid = episode + "-" + role
                    ids.append(cid)
                    text = (
                        json.dumps({"subquestions": ["Who?", "Where is #1?"]})
                        if role == "root"
                        else json.dumps({"answer": "indeed" if correct else "no"})
                    )
                    save(
                        output / "calls" / (cid + ".json"),
                        {
                            "call_id": cid,
                            "condition": condition,
                            "role": role,
                            "available": True,
                            "text": text,
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                            "input_token_ids": [1, 2, 3],
                            "output_token_ids": [4, 5],
                        },
                    )
                save(
                    output / "episodes" / (episode + ".json"),
                    {
                        "episode_id": episode,
                        "case_id": parent,
                        "condition": condition,
                        "repeat": repeat,
                        "call_ids": ids,
                        "status": "invalid_helper" if invalid else "scored",
                        "correct": False,
                        "f1": 0,
                        "plan_valid": True,
                    },
                )
    return output, cases


def test_regrades_aliases_keeps_missing_denominator_and_decomposes_paired_changes(tmp_path):
    m = implementation()
    output, cases = fixture(tmp_path)
    report = m.analyze(output, cases, draws=100)
    assert report["method"]["independent_parent_count"] == 2
    assert report["method"]["episodes_per_condition"] == 4
    assert report["groups"]["base"]["em"] == 0.5
    assert report["groups"]["adapted"]["em"] == 0.5
    assert report["groups"]["adapted"]["missing_episodes"] == 1
    assert report["groups"]["base"]["status_counts"]["invalid_helper"] == 2
    pair = report["comparisons"]["adapted_minus_base"]
    assert pair["em"]["estimate"] == 0
    assert pair["wins"]["episodes"] == pair["losses"]["episodes"] == 2
    assert pair["wins"]["protocol_involved"] == 2
    assert pair["losses"]["both_scored"] == 1
    assert pair["losses"]["missing_involved"] == 1
    assert report["groups"]["base"]["plan_lengths"] == {"2": 4}
    assert report["groups"]["base"]["distinct_plans"] == 1
    assert report["groups"]["base"]["deployed_cost"]["calls"] == 6
    assert report["groups"]["base"]["per_planned_attempt"]["total_tokens"] == 7.5
    assert "2 parents, not 4 independent episodes" in m.markdown(report)
    path = tmp_path / "report.json"
    m.write_report(report, path)
    assert path.exists() and path.with_suffix(".md").exists()
    with pytest.raises(FileExistsError):
        m.write_report(report, path)


def test_mismatched_cases_or_native_token_receipts_are_rejected(tmp_path):
    m = implementation()
    output, cases = fixture(tmp_path)
    call_path = next((output / "calls").glob("*.json"))
    row = json.loads(call_path.read_text())
    row["usage"]["completion_tokens"] = 99
    call_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="native token"):
        m.analyze(output, cases, draws=10)
    cases.write_text(cases.read_text() + "\n")
    with pytest.raises(ValueError, match="cases"):
        m.analyze(output, cases, draws=10)


def split_fixture(tmp_path):
    original, cases = fixture(tmp_path)
    outputs = []
    for old, new in (("base", "sft"), ("adapted", "rl")):
        output = tmp_path / new
        output.mkdir()
        plan = json.loads((original / "PLAN.json").read_text())
        plan.update(conditions=[new], seed=123, source_sha256="same-sealed-source")
        adapter = tmp_path / (new + "-checkpoint")
        adapter.mkdir()
        checkpoint = adapter / "STATE.json"
        checkpoint.write_text(json.dumps({"step": 48 if new == "sft" else 1}))
        plan.update(
            adapter=str(adapter),
            adapter_files_sha256={
                "STATE.json": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            },
        )
        (output / "PLAN.json").write_text(json.dumps(plan))
        for directory in ("calls", "episodes"):
            (output / directory).mkdir()
            for path in (original / directory).glob("*.json"):
                row = json.loads(path.read_text())
                if row["condition"] == old:
                    # Rename only fixture identities and condition, never response text.
                    row["condition"] = new
                    for key in ("call_id", "episode_id"):
                        if key in row:
                            row[key] = row[key].replace("-" + old, "-" + new)
                    if "call_ids" in row:
                        row["call_ids"] = [
                            cid.replace("-" + old, "-" + new) for cid in row["call_ids"]
                        ]
                    name = path.name.replace("-" + old, "-" + new)
                    (output / directory / name).write_text(json.dumps(row))
        outputs.append(output)
    return *outputs, cases


def test_joins_separate_trained_conditions_with_source_hashes_and_no_base_rerun(tmp_path):
    m = implementation()
    sft, rl, cases = split_fixture(tmp_path)
    solo = m.analyze(sft, cases, draws=100)
    assert solo["comparisons"] == {}
    assert "Unpaired single-condition report; no effect estimate" in m.markdown(solo)
    result = m.analyze(sft, cases, comparison_output=rl, draws=100)
    assert set(result["groups"]) == {"sft", "rl"}
    assert set(result["comparisons"]) == {"rl_minus_sft"}
    assert result["comparisons"]["rl_minus_sft"]["wins"]["protocol_involved"] == 2
    assert result["groups"]["rl"]["missing_episodes"] == 1
    assert result["comparison_output"] == str(rl)
    assert [source["conditions"] for source in result["source_plans"]] == [["sft"], ["rl"]]
    for output in (sft, rl):
        assert str(output / "PLAN.json") in result["input_source_checkpoint_sha256"]
        plan = json.loads((output / "PLAN.json").read_text())
        assert str(Path(plan["adapter"]) / "STATE.json") in result["input_source_checkpoint_sha256"]
        assert (
            str(next((output / "episodes").glob("*.json")))
            in result["input_source_checkpoint_sha256"]
        )
    assert "rl_minus_sft" in m.markdown(result) and str(rl) in m.markdown(result)


def test_joins_base_sft_run_with_rl_run_and_reports_all_three_pairs(tmp_path):
    m = implementation()
    sft, rl, cases = split_fixture(tmp_path)
    plan_path = sft / "PLAN.json"
    plan = json.loads(plan_path.read_text())
    plan["conditions"] = ["base", "sft"]
    plan_path.write_text(json.dumps(plan))
    for directory in ("calls", "episodes"):
        for path in list((sft / directory).glob("*.json")):
            row = json.loads(path.read_text())
            row["condition"] = "base"
            for field in ("call_id", "episode_id"):
                if field in row:
                    row[field] = row[field].replace("-sft", "-base")
            if "call_ids" in row:
                row["call_ids"] = [cid.replace("-sft", "-base") for cid in row["call_ids"]]
            (sft / directory / path.name.replace("-sft", "-base")).write_text(json.dumps(row))
    result = m.analyze(sft, cases, comparison_output=rl, draws=100)
    assert set(result["groups"]) == {"base", "sft", "rl"}
    assert set(result["comparisons"]) == {"sft_minus_base", "rl_minus_base", "rl_minus_sft"}
    assert all(group["planned_episodes"] == 4 for group in result["groups"].values())
    assert result["comparisons"]["sft_minus_base"]["em"]["estimate"] == 0
    assert [s["conditions"] for s in result["source_plans"]] == [["base", "sft"], ["rl"]]


def test_helper_checkpoint_is_hashed_and_mutation_rejected(tmp_path):
    m = implementation()
    output, cases = fixture(tmp_path)
    helper = tmp_path / "helper" / "checkpoint-36"
    helper.mkdir(parents=True)
    state = helper / "STATE.json"
    state.write_text('{"step":36}')
    training_plan = helper.parent / "PLAN.json"
    training_plan.write_text('{"role":"helper"}')
    path = output / "PLAN.json"
    plan = json.loads(path.read_text())
    plan["helper_contract"] = {
        "mode": "trained_helper",
        "adapter": str(helper),
        "adapter_binding": {"STATE.json": hashlib.sha256(state.read_bytes()).hexdigest()},
        "training_plan_sha256": hashlib.sha256(training_plan.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(plan))
    report = m.analyze(output, cases, draws=10)
    assert str(state) in report["input_source_checkpoint_sha256"]
    state.write_text('{"step":35}')
    with pytest.raises(ValueError, match="hash changed"):
        m.analyze(output, cases, draws=10)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cases_sha256", "different"),
        ("case_ids", ["b", "a"]),
        ("repeats", 3),
        ("seed", 124),
        ("execution", "bundled"),
        ("source_sha256", "other-source"),
        ("conditions", ["sft"]),
        ("conditions", []),
        ("mode", "direct"),
        ("architecture", "different architecture"),
        ("helper_contract", {"mode": "format_reminder"}),
    ],
)
def test_comparison_rejects_mismatched_contract_or_duplicate_condition(tmp_path, field, value):
    m = implementation()
    sft, rl, cases = split_fixture(tmp_path)
    path = rl / "PLAN.json"
    plan = json.loads(path.read_text())
    plan[field] = value
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="comparison"):
        m.analyze(sft, cases, comparison_output=rl, draws=10)
