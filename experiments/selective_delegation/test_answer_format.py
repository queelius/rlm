"""Final-only intervention preserves evidence and separates reused work from new work."""

import importlib.util
import json
from pathlib import Path

import probe
import pytest


def implementation():
    path = Path(__file__).with_name("answer_format.py")
    assert path.exists(), "final-only ablation implementation missing"
    spec = importlib.util.spec_from_file_location("answer_format", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    case = {
        "id": "abc",
        "split": "train",
        "question": "Public question",
        "documents": [{"id": "3", "title": "Public title", "text": "Source evidence"}],
        "answer": "SECRET_GOLD",
        "metadata": {"answer_aliases": ["SECRET_ALIAS"], "source_id": "SECRET_SOURCE"},
    }
    state = {
        "answer": "Provisional",
        "evidence_question": "Check this",
        "subquestions": ["First", "Second"],
    }
    report = "Exact helper report\nwith formatting preserved."
    episode = {
        "episode_id": "abc-r0-targeted",
        "case_id": "abc",
        "split": "train",
        "arm": "targeted",
        "repeat": 0,
        "seed": 501,
        "available": True,
        "call_ids": ["abc-checkpoint", "abc-r0-targeted-helper", "abc-r0-targeted-final"],
        "checkpoint_digest": probe.runtime.digest(state),
    }
    records = []
    for cid, text, prompt in zip(
        episode["call_ids"],
        [json.dumps(state), report, '{"answer":"Old answer"}'],
        [
            probe.initial_prompt(case),
            probe.helper_prompt(case, state, "targeted"),
            probe.final_prompt(case, state, report),
        ],
        strict=True,
    ):
        request = {
            "model": probe.campaign.MODELS["4b"],
            "sampling_params": {"seed": 501, "temperature": 0.5, "max_tokens": 128},
        }
        record = {
            "call_id": cid,
            "available": True,
            "text": text,
            "prompt": prompt,
            "request": request,
            "request_digest": probe.runtime.digest(request),
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        probe.runtime.save(tmp_path / "calls" / f"{cid}.json", record)
        records.append(record)
    path = tmp_path / "episodes" / (episode["episode_id"] + ".json")
    probe.runtime.save(path, episode)
    return case, path, report, records


def test_prompt_excludes_gold_and_preserves_original_state_and_report(tmp_path):
    case, path, report, records = fixture(tmp_path)
    job = implementation().prepare_episode(path, case)
    assert job["prompt"] == probe.PHRASE_INSTRUCTION + records[-1]["prompt"]
    assert all(
        secret not in job["prompt"] for secret in ["SECRET_GOLD", "SECRET_ALIAS", "SECRET_SOURCE"]
    )
    assert job["report"] == report
    assert job["seed"] == 501 and job["temperature"] == 0.5 and job["max_tokens"] == 128
    assert job["reused_call_ids"] == [r["call_id"] for r in records[:-1]]


def test_modified_source_report_fails_identity_check(tmp_path):
    case, path, _, records = fixture(tmp_path)
    helper_path = tmp_path / "calls" / (records[1]["call_id"] + ".json")
    record = json.loads(helper_path.read_text())
    record["text"] = "Changed report"
    helper_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="source final prompt"):
        implementation().prepare_episode(path, case)


def test_one_new_final_cost_is_separate_from_hypothetical_deployed_cost(tmp_path):
    case, path, _, _ = fixture(tmp_path)
    mod = implementation()
    job = mod.prepare_episode(path, case)
    new = {
        "call_id": "new-final",
        "available": True,
        "text": '{"answer":"SECRET_GOLD"}',
        "usage": {"prompt_tokens": 20, "completion_tokens": 4},
    }
    result = mod.pair_result(job, case, new)
    assert result["new_physical_cost"] == {
        "calls": 1,
        "prompt_tokens": 20,
        "completion_tokens": 4,
        "unknown_usage_calls": 0,
    }
    assert result["hypothetical_deployed_cost"] == {
        "calls": 3,
        "prompt_tokens": 40,
        "completion_tokens": 8,
        "unknown_usage_calls": 0,
    }
    assert result["old"]["correct"] is False and result["new"]["correct"] is True
