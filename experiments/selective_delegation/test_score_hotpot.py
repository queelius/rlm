"""Focused denominator and strict-parser contracts for HotpotQA regrading."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("score_hotpot.py")
    assert path.exists(), "Hotpot scorer implementation is missing"
    spec = importlib.util.spec_from_file_location("score_hotpot", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def call(identity, text, available=True):
    return {"call_id": identity, "available": available, "text": text}


def episode(case_id, condition, final_id):
    return {
        "episode_id": f"{case_id}-r0-{condition}",
        "case_id": case_id,
        "repeat": 0,
        "condition": condition,
        "call_ids": [f"{case_id}-r0-{condition}-root", final_id],
        "correct": False,
        "f1": 0.0,
        "metric": "official_musique_alias_max_em_f1",
    }


def test_regrade_keeps_all_planned_slots_and_rejects_non_strict_finals(tmp_path):
    m = module()
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "a",
                        "split": "transfer",
                        "dataset": "hotpotqa",
                        "answer": "The Answer",
                        "metadata": {"answer_aliases": []},
                    }
                ),
                json.dumps(
                    {
                        "id": "b",
                        "split": "transfer",
                        "dataset": "hotpotqa",
                        "answer": "yes",
                        "metadata": {"answer_aliases": []},
                    }
                ),
            ]
        )
        + "\n"
    )
    output = tmp_path / "evaluation"
    (output / "episodes").mkdir(parents=True)
    (output / "calls").mkdir()
    (output / "PLAN.json").write_text(json.dumps({"conditions": ["base", "sft"], "repeats": 1}))
    records = [
        episode("a", "base", "a-r0-base-final"),
        episode("a", "sft", "a-r0-sft-final"),
        episode("b", "base", "b-r0-base-final"),
    ]
    for record in records:
        (output / "episodes" / (record["episode_id"] + ".json")).write_text(json.dumps(record))
    for record in (
        call("a-r0-base-final", '{"answer":"the answer"}'),
        call("a-r0-sft-final", '{"answer":"the answer","extra":true}'),
        call("b-r0-base-final", '{"answer":"no"}'),
    ):
        (output / "calls" / (record["call_id"] + ".json")).write_text(json.dumps(record))

    report = m.score(cases, output, tmp_path / "report")
    assert report["conditions"]["base"]["planned"] == 2
    assert report["conditions"]["base"]["em"] == 0.5
    assert report["conditions"]["base"]["f1"] == 0.5
    assert report["conditions"]["sft"]["planned"] == 2
    assert report["conditions"]["sft"]["invalid_final"] == 1
    assert report["conditions"]["sft"]["missing_episode"] == 1
    assert report["conditions"]["sft"]["em"] == 0.0
    assert report["paired_comparison"]["base_only_correct"] == 1
    assert report["paired_comparison"]["both_incorrect"] == 1
    assert (tmp_path / "report" / "REPORT.json").exists()
    assert "the answer" not in (tmp_path / "report" / "REPORT.md").read_text().lower()


def test_strict_answer_contract_rejects_duplicate_or_extra_fields():
    m = module()
    assert m.parse_answer('{"answer":"ok"}') == "ok"
    for text in ('{"answer":"a","answer":"b"}', '{"answer":"a","extra":1}', "not json"):
        with pytest.raises(ValueError):
            m.parse_answer(text)


def single_fixture(tmp_path, condition="rl"):
    m = module()
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            json.dumps({"id": p, "split": "transfer", "dataset": "hotpotqa", "answer": "yes"})
            for p in ("a", "b")
        )
    )
    output = tmp_path / condition
    (output / "episodes").mkdir(parents=True)
    (output / "calls").mkdir()
    plan = {
        "conditions": [condition],
        "repeats": 2,
        "case_ids": ["a"],
        "cases_sha256": m.sha256(cases),
        "seed": 42,
        "execution": "isolated",
        "mode": "planner",
        "source_sha256": "same-sealed-code",
    }
    (output / "PLAN.json").write_text(json.dumps(plan))
    return m, cases, output, plan


def test_single_condition_uses_planned_subset_and_separates_protocol_from_missing(tmp_path):
    m, cases, output, plan = single_fixture(tmp_path)
    row = episode("a", "rl", "a-r0-rl-root")
    row.update(call_ids=["a-r0-rl-root"], status="invalid_plan")
    (output / "episodes" / (row["episode_id"] + ".json")).write_text(json.dumps(row))
    report = m.score(cases, output, tmp_path / "report")
    group = report["conditions"]["rl"]
    assert group["planned"] == 2
    assert group["protocol_failure"] == 1
    assert group["missing_episode"] == 1
    assert group["status_counts"] == {"invalid_plan": 1, "missing_episode": 1}
    assert report["paired_comparison"] is None
    assert "Unpaired single-condition" in (tmp_path / "report/REPORT.md").read_text()


def test_matching_single_outputs_pair_with_truthful_condition_names_and_hashes(tmp_path):
    m, cases, left, plan = single_fixture(tmp_path, "sft")
    right = tmp_path / "rl"
    (right / "episodes").mkdir(parents=True)
    (right / "calls").mkdir()
    other = {**plan, "conditions": ["rl"]}
    (right / "PLAN.json").write_text(json.dumps(other))
    row = episode("a", "rl", "a-r0-rl-final")
    row.update(call_ids=["a-r0-rl-final"], status="scored")
    (right / "episodes" / (row["episode_id"] + ".json")).write_text(json.dumps(row))
    receipt = call("a-r0-rl-final", '{"answer":"yes"}')
    receipt.update(condition="rl", role="final")
    (right / "calls" / "a-r0-rl-final.json").write_text(json.dumps(receipt))
    report = m.score(cases, left, tmp_path / "report", comparison_output=right)
    assert report["condition_order"] == ["sft", "rl"]
    assert report["conditions"]["rl"]["em"] == 0.5
    assert report["paired_comparison"]["second_only_correct"] == 1
    assert report["paired_comparison"]["second_only_correct_missing_involved"] == 1
    assert len(report["sources"]["evaluations"]) == 2
    other["seed"] = 43
    (right / "PLAN.json").write_text(json.dumps(other))
    with pytest.raises(ValueError, match="contract differs: seed"):
        m.score(cases, left, tmp_path / "bad-report", comparison_output=right)


def test_single_rejects_changed_cases_and_wrong_condition_final_receipt(tmp_path):
    m, cases, output, plan = single_fixture(tmp_path)
    row = episode("a", "rl", "a-r0-rl-final")
    (output / "episodes" / (row["episode_id"] + ".json")).write_text(json.dumps(row))
    receipt = call("a-r0-rl-final", '{"answer":"yes"}')
    receipt["condition"] = "sft"
    (output / "calls" / "a-r0-rl-final.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="condition"):
        m.score(cases, output, tmp_path / "bad-condition")
    cases.write_text(cases.read_text() + "\n")
    with pytest.raises(ValueError, match="cases"):
        m.score(cases, output, tmp_path / "bad-cases")


def test_native_receipt_digest_and_unavailable_final_accounting(tmp_path):
    m, cases, output, plan = single_fixture(tmp_path)
    row = episode("a", "rl", "a-r0-rl-final")
    row["status"] = "scored"
    (output / "episodes" / (row["episode_id"] + ".json")).write_text(json.dumps(row))
    receipt = call("a-r0-rl-final", None, available=False)
    receipt.update(condition="rl", role="final", request={"seed": 12, "prompt": "Public"})
    receipt["request_digest"] = hashlib.sha256(
        json.dumps(receipt["request"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    path = output / "calls" / "a-r0-rl-final.json"
    path.write_text(json.dumps(receipt))
    report = m.score(cases, output, tmp_path / "report")
    assert report["conditions"]["rl"]["unavailable_final_receipt"] == 1
    assert report["conditions"]["rl"]["protocol_failure"] == 0
    receipt["request"]["seed"] = 13
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="request digest"):
        m.score(cases, output, tmp_path / "bad-digest")
