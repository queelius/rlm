import copy

from prepare_textcraft_quantity_matched import correct_task


def test_correct_task_changes_only_train1029_first_quantity() -> None:
    task = {
        "id": "textcraft_synth.train.1029",
        "misc": {
            "gold_trajectory": [
                {
                    "action": "craft",
                    "target": ["t4_i1", 3],
                    "ingredients": {"raw_t8": 6},
                    "result_count": 12,
                },
                {
                    "action": "craft",
                    "target": ["later", 1],
                    "ingredients": {"raw": 2},
                    "result_count": 2,
                },
            ]
        },
    }
    original = copy.deepcopy(task)
    fixed = correct_task(task)
    assert task == original
    assert fixed["misc"]["gold_trajectory"] == [
        {
            "action": "craft",
            "target": ["t4_i1", 2],
            "ingredients": {"raw_t8": 4},
            "result_count": 8,
        },
        {
            "action": "craft",
            "target": ["later", 1],
            "ingredients": {"raw": 2},
            "result_count": 2,
        },
    ]
