from train_textcraft_quantity_matched import prepared_contract


def test_corrected_whole_trajectory_contract_is_pinned() -> None:
    manifest = prepared_contract()
    assert manifest["tasks"] == 32
    assert manifest["rows"] == 366
    assert manifest["action_target_differences"] == 1
    assert manifest["prompt_differences"] == 29
