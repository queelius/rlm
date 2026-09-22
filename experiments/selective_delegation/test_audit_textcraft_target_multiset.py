import json
from hashlib import sha256
from pathlib import Path

import pytest
from audit_textcraft_target_multiset import audit_rows, training_contract


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _target(value: dict[str, object]) -> str:
    return json.dumps(value, separators=(",", ":"))


def test_audit_ignores_row_order_and_reports_canonical_target_quantity_change(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    _write(
        old_path,
        [
            {
                "task_id": "a",
                "step": 0,
                "prompt": "same",
                "target": _target({"action": "get_info", "items": ["x"]}),
            },
            {
                "task_id": "a",
                "step": 1,
                "prompt": "p",
                "target": _target(
                    {
                        "action": "craft",
                        "ingredients": {"ore": 6},
                        "target_item": "t",
                        "output_count": 12,
                    }
                ),
            },
            {
                "task_id": "a",
                "step": 2,
                "prompt": "q",
                "target": _target({"action": "finish", "message": "done"}),
            },
        ],
    )
    _write(
        new_path,
        [
            {
                "task_id": "a",
                "step": 2,
                "prompt": "q",
                "target": _target({"message": "done", "action": "finish"}),
            },
            {
                "task_id": "a",
                "step": 0,
                "prompt": "same",
                "target": _target({"items": ["x"], "action": "get_info"}),
            },
            {
                "task_id": "a",
                "step": 1,
                "prompt": "p",
                "target": _target(
                    {
                        "output_count": 8,
                        "target_item": "t",
                        "ingredients": {"ore": 4},
                        "action": "craft",
                    }
                ),
            },
        ],
    )

    report = audit_rows(old_path, new_path)

    assert report["initial_prompts"] == {"identical_tasks": ["a"], "different_tasks": []}
    assert report["action_multisets"]["get_info"]["identical_tasks"] == ["a"]
    assert report["action_multisets"]["finish"]["identical_tasks"] == ["a"]
    assert report["action_multisets"]["craft"]["different_tasks"] == ["a"]
    assert report["target_differences"] == [
        {
            "task_id": "a",
            "old_only": [
                '{"action":"craft","ingredients":{"ore":6},'
                '"output_count":12,"target_item":"t"}'
            ],
            "new_only": [
                '{"action":"craft","ingredients":{"ore":4},'
                '"output_count":8,"target_item":"t"}'
            ],
        }
    ]


def test_training_contract_reports_equal_hyperparameters_and_step_zero_adapter(
    tmp_path: Path,
) -> None:
    plans = []
    for name in ("old", "new"):
        root = tmp_path / name
        checkpoint = root / "checkpoint-0000"
        checkpoint.mkdir(parents=True)
        (root / "PLAN.json").write_text(
            json.dumps({"seed": 7, "learning_rate": 1e-4, "planned_updates": 23})
        )
        adapter = checkpoint / "adapter_model.safetensors"
        adapter.write_bytes(b"same adapter bytes")
        files = {"adapter_model.safetensors": sha256(adapter.read_bytes()).hexdigest()}
        (checkpoint / "COMMIT.json").write_text(
            json.dumps({"files": files})
        )
        plans.append(root / "PLAN.json")

    contract = training_contract(*plans)

    assert contract["equal_plan_fields"] == ["learning_rate", "planned_updates", "seed"]
    assert contract["different_plan_fields"] == []
    assert contract["step_zero_adapter_equal"] is True
    assert contract["step_zero_adapter_bytes_match_commit"] == {"old": True, "new": True}

    (plans[1].parent / "checkpoint-0000" / "adapter_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="adapter bytes do not match COMMIT"):
        training_contract(*plans)


def test_audit_separates_raw_target_strings_from_canonical_actions_and_labels(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    _write(
        old_path,
        [
            {
                "task_id": "a",
                "step": 0,
                "prompt": "same",
                "target": '{"action":"get_info","items":["x"]}',
                "labels": [-100, 4, 5],
            }
        ],
    )
    _write(
        new_path,
        [
            {
                "task_id": "a",
                "step": 0,
                "prompt": "same",
                "target": '{"items":["x"],"action":"get_info"}',
                "labels": [-100, 4, 5],
            }
        ],
    )

    report = audit_rows(old_path, new_path)

    assert report["action_multisets"]["get_info"]["identical_tasks"] == ["a"]
    assert report["raw_target_multisets"]["get_info"]["different_tasks"] == ["a"]
    assert report["unmasked_label_multisets"]["get_info"]["identical_tasks"] == ["a"]
