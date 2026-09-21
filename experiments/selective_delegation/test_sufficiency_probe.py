import json

import pytest
import sufficiency_probe as probe


def test_official_group_uses_positive_aliases_and_both_labels():
    positive = {"answer": "canonical", "answer_aliases": ["alias"]}
    predictions = [
        {"answerable": True, "answer": "alias"},
        {"answerable": False, "answer": "unrelated"},
    ]
    assert probe.group_score(positive, predictions) == {"em": 1.0, "f1": 1.0, "suff": 1.0}
    predictions[1]["answerable"] = True
    assert probe.group_score(positive, predictions) == {"em": 0.0, "f1": 0.0, "suff": 0.0}
    assert probe.group_score(positive, [predictions[0], None])["em"] == 0


def test_parser_is_strict_and_prompt_projects_only_public_context():
    for text in [
        '{"answerable":1,"answer":"x"}',
        '{"answerable":true,"answer":"x","gold":"x"}',
        '{"answerable":true,"answerable":false,"answer":"x"}',
    ]:
        with pytest.raises(ValueError):
            probe.parse_output(text)
    row = {
        "public": {"question": "Q", "documents": []},
        "answer": "SECRET",
        "answerable": True,
        "source_id": "SECRET",
    }
    assert "SECRET" not in probe.prompt(row)
    assert json.loads(probe.prompt(row).split("\n", 1)[1]) == row["public"]


def test_summary_keeps_missing_separate_from_returned_protocol_and_native_cost(tmp_path):
    cases, jobs = [], []
    for parent in range(32):
        for label in (True, False):
            case = {
                "id": f"{parent}-{label}",
                "parent_id": str(parent),
                "answerable": label,
                "answer": "gold",
                "answer_aliases": [],
            }
            cases.append(case)
            for seed in probe.SEEDS:
                jobs.append({"episode_id": f"{case['id']}-{seed}"})
    plan = {"jobs": jobs}
    probe.native.save(tmp_path / "PLAN.json", plan)
    first = jobs[0]["episode_id"]
    probe.native.save(
        tmp_path / "episodes" / (first + ".json"), {"available": True, "prediction": None}
    )
    probe.native.save(
        tmp_path / "calls" / (first + ".json"),
        {
            "available": True,
            "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            "started": 1,
            "ended": 3,
        },
    )
    result = probe.summarize(tmp_path, plan, cases)
    assert result["returned_protocol_invalid"] == 1
    assert result["missing_or_inference_unavailable"] == 127
    assert result["returned_valid"] == 0
    assert len(result["groups"]) == 64
    assert result["group_answer_sufficiency"] == {"em": 0, "f1": 0, "suff": 0}
    assert result["physical_cost"]["calls"] == 1
    assert result["physical_cost"]["prompt_tokens"] == 20
    assert result["native_call_seconds"] == 2
