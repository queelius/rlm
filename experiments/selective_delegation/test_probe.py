import json

import pytest
from probe import ARMS, checkpoint, cost, final_prompt, grade, public, summarize

CASE = {
    "question": "Who owns it?",
    "documents": [{"id": "p0", "title": "Title", "text": "Maya owns the bike."}],
    "answer": "Maya",
    "metadata": {"answer_aliases": ["M. Smith"], "source_id": "SECRET"},
}
STATE = {
    "answer": "Maya",
    "evidence_question": "Who owns the bike?",
    "subquestions": ["What is it?", "Who owns that?"],
}


def test_public_projection_and_common_full_source():
    assert set(public(CASE)) == {"question", "documents"}
    for arm in ARMS:
        prompt = final_prompt(CASE, STATE, None if arm == "finish" else "Evidence report")
        assert "Maya owns the bike." in prompt
        assert "SECRET" not in prompt and "M. Smith" not in prompt
        assert "Who owns the bike?" in prompt


def test_checkpoint_requires_exact_typed_shape():
    assert checkpoint(json.dumps(STATE)) == STATE
    for bad in ({**STATE, "extra": 1}, {**STATE, "subquestions": []}, {**STATE, "answer": 3}):
        with pytest.raises(ValueError):
            checkpoint(json.dumps(bad))


def test_span_instruction_changes_only_final_rendering_guidance():
    original = final_prompt(CASE, STATE, "Same saved report")
    span = final_prompt(CASE, STATE, "Same saved report", answer_style="span")
    assert span.endswith(original)
    assert "Keep qualifiers needed for the answer." in span
    assert "M. Smith" not in span and "SECRET" not in span


def test_alias_official_em_and_f1():
    assert grade('{"answer":"M. Smith"}', CASE)["correct"]
    assert grade('{"answer":"the Maya"}', CASE)["f1"] == 1
    assert not grade('{"answer":null}', CASE)["valid"]
    assert not grade("Maya", CASE)["valid"]


def test_cost_keeps_unknown_usage_separate():
    result = cost(
        [
            {"available": True, "usage": {"prompt_tokens": 12, "completion_tokens": 4}},
            {"available": False, "usage": {}},
        ]
    )
    assert result == {
        "calls": 2,
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "unknown_usage_calls": 1,
    }


def test_summary_preserves_planned_and_failed_checkpoint_denominators(tmp_path):
    (tmp_path / "PLAN.json").write_text(json.dumps({"case_ids": ["a", "b"], "repeats": 3}))
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/a.json").write_text(
        json.dumps({"case_id": "a", "state": None, "error": "checkpoint_invalid"})
    )
    result = summarize(tmp_path)
    assert result["planned_parents"] == 2
    assert result["checkpoint_failures"] == {"checkpoint_invalid": 1}
    assert result["groups"]["finish"]["planned_episodes"] == 6
    assert result["groups"]["finish"]["checkpoint_failed_episodes"] == 3
    assert result["groups"]["finish"]["unresolved_episodes"] == 3
