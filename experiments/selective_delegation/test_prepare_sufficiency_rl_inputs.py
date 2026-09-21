def test_history_train_rows_and_shared_atoms_are_excluded():
    import prepare_sufficiency_rl_inputs as preparation

    exclusions = preparation.training.empty_exclusions()
    history = {
        "id": "hashed",
        "split": "train",
        "question": "History?",
        "metadata": {"source_id": "2hop__1_2", "component_ids": ["1", "2"]},
    }
    preparation.training.add_exclusion(history, exclusions)
    candidate = [
        {
            "id": "2hop__2_3",
            "question": "New question?",
            "answerable": label,
            "question_decomposition": [{"id": "2"}, {"id": "3"}],
        }
        for label in (True, False)
    ]
    assert preparation.training.exclusion_reasons(candidate, exclusions) == ["component"]
    ids = ["2hop__4_5", "2hop__6_7", "2hop__8_9"]
    selected = preparation.select(ids, 2)
    assert selected == preparation.select(list(reversed(ids)), 2)
    assert len(selected) == len(set(selected)) == 2
