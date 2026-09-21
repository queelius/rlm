"""Regression checks for excluding every known non-TRAIN study input."""

import json

import prepare_sufficiency_canonical_panel as canonical


def test_study_exposure_inventory_keeps_nontrain_rows_and_skips_train_rows(tmp_path):
    inputs = tmp_path / "inputs-001"
    inputs.mkdir()
    rows = [
        {"id": "train", "split": "train"},
        {"id": "dev", "split": "development"},
    ]
    (inputs / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    fresh = tmp_path / "fresh-dev-inputs-003"
    fresh.mkdir()
    (fresh / "cases.jsonl").write_text(json.dumps({"id": "fresh", "split": "development"}) + "\n")
    inventory = canonical.study_input_rows(tmp_path)
    assert [row["id"] for row in inventory["inputs-001/cases.jsonl"]] == ["dev"]
    assert [row["id"] for row in inventory["fresh-dev-inputs-003/cases.jsonl"]] == ["fresh"]
