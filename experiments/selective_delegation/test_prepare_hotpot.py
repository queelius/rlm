"""Focused contracts for the immutable HotpotQA explorer-sample transfer panel."""

import importlib.util
import json
from pathlib import Path


def module():
    path = Path(__file__).with_name("prepare_hotpot.py")
    assert path.exists(), "Hotpot preparation implementation is missing"
    spec = importlib.util.spec_from_file_location("prepare_hotpot", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(identity, question, docs=10):
    distractors = [[f"Noise {index}", [f"noise text {index}"], 0.0] for index in range(docs - 2)]
    return {
        "_id": identity,
        "question": question,
        "answer": "The Gold",
        "title_a": "Support A",
        "para_a": ["support a text"],
        "title_b": "Support B",
        "para_b": ["support b text"],
        "supporting_facts": [[0], [0]],
        "distractors": distractors,
    }


def test_prepare_selects_public_ten_document_cases_without_muSiQue_question_overlap(tmp_path):
    source = tmp_path / "hotpot.json"
    source.write_text(
        json.dumps(
            [
                row("overlap", "The Question?"),
                row("kept-a", "Other question"),
                row("kept-b", "Third question"),
                row("short", "Short context", docs=2),
            ]
        )
    )
    musique = tmp_path / "musique.jsonl"
    musique.write_text(
        json.dumps(
            {
                "split": "train",
                "question": "the question",
                "documents": [{"title": "Old title", "text": "Old text"}],
            }
        )
        + "\n"
    )
    output = tmp_path / "output"
    receipt = module().prepare(source, musique, output, count=2, seed=7)

    cases = [json.loads(line) for line in (output / "cases.jsonl").open()]
    assert len(cases) == 2
    assert all(case["split"] == "transfer" and case["dataset"] == "hotpotqa" for case in cases)
    assert {case["metadata"]["source_id"] for case in cases} == {"kept-a", "kept-b"}
    assert all(len(case["documents"]) == 10 for case in cases)
    assert all(
        len(case["id"]) == 24 and all(c in "0123456789abcdef" for c in case["id"]) for case in cases
    )
    serialized = json.dumps(cases, ensure_ascii=False)
    assert "The Gold" in serialized  # Host-only gold remains in the saved case.
    assert "supporting_facts" in serialized
    assert receipt["selection"]["question_overlap_excluded"] == 1
    assert receipt["selection"]["eligible_after_question_exclusion"] == 2
    assert receipt["scoring"]["probe_grade_compatible"] is False
