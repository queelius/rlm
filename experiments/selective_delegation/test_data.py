"""Behavioral checks for leakage-free public inputs and frozen group splits."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("data.py")
    assert path.exists(), "data preparation implementation is missing"
    spec = importlib.util.spec_from_file_location("selective_data", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(name, components):
    return {
        "id": name,
        "question": "Who wrote the book?",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Book",
                "paragraph_text": "Visible\u2028text",
                "is_supporting": True,
            }
        ],
        "answer": "Secret gold",
        "answer_aliases": ["Secret alias"],
        "answerable": True,
        "question_decomposition": [
            {
                "id": c,
                "question": "Secret step",
                "answer": "Secret intermediate",
                "paragraph_support_idx": 0,
            }
            for c in components
        ],
    }


def test_public_projection_removes_all_host_annotations():
    case = {
        "id": "opaque",
        "question": "Public question",
        "answer": "Secret gold",
        "documents": [{"id": "0", "title": "Title", "text": "Visible", "is_supporting": True}],
        "metadata": {"source_id": "4hop-secret", "component_ids": ["secret"]},
    }
    assert module().public_projection(case) == {
        "question": "Public question",
        "documents": [{"id": "0", "title": "Title", "text": "Visible"}],
    }


def test_preparation_is_deterministic_excludes_prior_and_cross_split_components(tmp_path):
    archive = tmp_path / "fixture.zip"
    train = [row("2hop__a", [1, 2]), row("3hop1__b", [3, 4, 5])]
    dev = [
        row("2hop__prior", [10, 11]),
        row("2hop__overlap", [1, 12]),
        row("3hop1__valid", [20, 21, 22]),
        row("4hop1__overlap", [20, 30, 31, 32]),
        row("4hop1__transfer", [40, 41, 42, 43]),
    ]
    with zipfile.ZipFile(archive, "w") as stream:
        for split, rows in (("train", train), ("dev", dev)):
            stream.writestr(
                f"data/musique_ans_v1.0_{split}.jsonl",
                "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
            )
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        json.dumps({"dataset": "musique", "metadata": {"source_id": "2hop__prior"}}) + "\n"
    )
    mod = module()
    for target in (tmp_path / "one", tmp_path / "two"):
        mod.prepare(target, 2, 1, 1, archive=archive, prior_cases=prior)
    assert (tmp_path / "one/cases.jsonl").read_bytes() == (
        tmp_path / "two/cases.jsonl"
    ).read_bytes()
    cases = [json.loads(line) for line in (tmp_path / "one/cases.jsonl").open()]
    assert {x["metadata"]["source_id"] for x in cases} == {
        "2hop__a",
        "3hop1__b",
        "3hop1__valid",
        "4hop1__transfer",
    }
    groups = {
        split: {c for x in cases if x["split"] == split for c in x["metadata"]["component_ids"]}
        for split in ("train", "validation", "transfer")
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["transfer"]
    assert not groups["validation"] & groups["transfer"]
    assert all("hop" not in x["id"] for x in cases)
    assert all(x["documents"][0]["text"] == "Visible\u2028text" for x in cases)
    manifest = json.loads((tmp_path / "one/MANIFEST.json").read_text())
    assert manifest["counts"] == {"train": 2, "validation": 1, "transfer": 1}
    with pytest.raises(FileExistsError):
        mod.prepare(tmp_path / "one", 2, 1, 1, archive=archive, prior_cases=prior)
