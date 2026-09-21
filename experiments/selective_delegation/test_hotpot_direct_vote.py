"""Focused contracts for the frozen three-sample Hotpot direct-vote control."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("hotpot_direct_vote.py")
    assert path.exists(), "direct-vote collector/analyzer implementation is missing"
    spec = importlib.util.spec_from_file_location("hotpot_direct_vote", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def voter(index, text, *, available=True):
    return {"index": index, "available": available, "text": text}


def test_vote_uses_official_normalization_and_original_index_breaks_three_way_tie():
    m = module()
    result = m.vote(
        [
            voter(0, '{"answer":"The Paris"}'),
            voter(1, '{"answer":"London"}'),
            voter(2, '{"answer":"Rome"}'),
        ]
    )
    assert result == {
        "status": "scored",
        "valid_voters": 3,
        "invalid_voters": 0,
        "unavailable_voters": 0,
        "chosen_index": 0,
        "chosen_raw_answer": "The Paris",
        "chosen_normalized_answer": "paris",
        "tied": True,
    }


def test_vote_marks_any_unavailable_voter_as_unobserved_three_sample_policy():
    m = module()
    result = m.vote(
        [
            voter(0, '{"answer":"wrong"}'),
            voter(1, "not-json"),
            voter(2, None, available=False),
        ]
    )
    assert result["status"] == "unavailable"
    assert result["chosen_raw_answer"] is None
    assert result["invalid_voters"] == 1
    assert result["unavailable_voters"] == 1


def test_vote_records_all_unavailable_or_invalid_as_unavailable():
    m = module()
    result = m.vote(
        [voter(0, None, available=False), voter(1, "[]"), voter(2, None, available=False)]
    )
    assert result["status"] == "unavailable"
    assert result["valid_voters"] == 0
    assert result["invalid_voters"] == 1
    assert result["unavailable_voters"] == 2
    assert result["chosen_index"] is None


def test_two_invalid_ballots_beat_one_valid_ballot_as_protocol_zero():
    m = module()
    result = m.vote([voter(0, "not-json"), voter(1, '{"answer":"right"}'), voter(2, "[]")])
    assert result["status"] == "protocol_zero"
    assert result["chosen_index"] == 0
    assert result["invalid_voters"] == 2
    assert result["chosen_raw_answer"] is None


def test_invalid_original_wins_three_way_tie_by_predeclared_index():
    m = module()
    result = m.vote(
        [voter(0, "not-json"), voter(1, '{"answer":"alpha"}'), voter(2, '{"answer":"beta"}')]
    )
    assert result["status"] == "protocol_zero"
    assert result["chosen_index"] == 0
    assert result["tied"] is True


def test_two_matching_valid_ballots_beat_one_invalid_ballot():
    m = module()
    result = m.vote(
        [voter(0, "not-json"), voter(1, '{"answer":"The Paris"}'), voter(2, '{"answer":"paris"}')]
    )
    assert result["status"] == "scored"
    assert result["chosen_index"] == 1
    assert result["chosen_raw_answer"] == "The Paris"


def test_build_jobs_binds_original_actual_seed_and_uses_only_two_offset_seeds(tmp_path):
    m = module()
    source = tmp_path / "original"
    (source / "calls").mkdir(parents=True)
    (source / "episodes").mkdir()
    plan = {
        "mode": "direct",
        "execution": "direct",
        "conditions": ["base"],
        "repeats": 2,
        "case_ids": ["opaque"],
        "seed": 1,
        "caps": {"final": 128},
        "temperature": 0.5,
        "top_p": 1.0,
        "top_k": 0,
    }
    (source / "PLAN.json").write_text(json.dumps(plan))
    for repeat, seed in enumerate((101, 201)):
        identity = f"opaque-r{repeat}-base-direct"
        (source / "episodes" / f"{identity}.json").write_text(
            json.dumps(
                {
                    "episode_id": identity,
                    "case_id": "opaque",
                    "repeat": repeat,
                    "call_ids": [identity + "-final"],
                }
            )
        )
        (source / "calls" / f"{identity}-final.json").write_text(
            json.dumps(
                {
                    "call_id": identity + "-final",
                    "available": True,
                    "request": {
                        "seed": seed,
                        "condition": "base",
                        "role": "final",
                        "adapter_enabled": False,
                        "sampling": {
                            "temperature": 0.5,
                            "top_p": 1.0,
                            "top_k": 0,
                            "max_new_tokens": 128,
                        },
                    },
                }
            )
        )
    jobs = m.build_jobs(source, ["opaque"])
    assert [(job["original_seed"], job["seed"], job["voter_index"]) for job in jobs] == [
        (101, 10101, 1),
        (101, 20101, 2),
        (201, 10201, 1),
        (201, 20201, 2),
    ]


def test_build_jobs_rejects_original_direct_receipt_that_changes_sampling(tmp_path):
    m = module()
    source = tmp_path / "original"
    (source / "calls").mkdir(parents=True)
    (source / "episodes").mkdir()
    (source / "PLAN.json").write_text(
        json.dumps(
            {
                "mode": "direct",
                "execution": "direct",
                "conditions": ["base"],
                "repeats": 2,
                "case_ids": ["opaque"],
                "caps": {"final": 128},
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
            }
        )
    )
    identity = "opaque-r0-base-direct"
    (source / "episodes" / f"{identity}.json").write_text(
        json.dumps(
            {
                "episode_id": identity,
                "case_id": "opaque",
                "repeat": 0,
                "call_ids": [identity + "-final"],
            }
        )
    )
    (source / "calls" / f"{identity}-final.json").write_text(
        json.dumps(
            {
                "call_id": identity + "-final",
                "request": {
                    "seed": 1,
                    "condition": "base",
                    "role": "final",
                    "adapter_enabled": False,
                    "sampling": {
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "top_k": 0,
                        "max_new_tokens": 128,
                    },
                },
            }
        )
    )
    with pytest.raises(ValueError, match="sampling"):
        m.build_jobs(source, ["opaque"])
