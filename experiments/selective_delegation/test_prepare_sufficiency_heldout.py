import copy


def test_selection_ignores_labels_and_excludes_each_identity_channel():
    import prepare_sufficiency_heldout as m

    exclusions = {
        "parents": {"blocked-parent"},
        "questions": {"blocked question"},
        "components": {"blocked-atom"},
    }
    pair = [
        {
            "id": "p",
            "question": "Allowed?",
            "answerable": True,
            "question_decomposition": [{"id": "atom"}],
        }
    ]
    assert m.reasons(pair, exclusions) == []
    changed = copy.deepcopy(pair)
    changed[0]["id"] = "blocked-parent"
    assert "known_parent" in m.reasons(changed, exclusions)
    changed[0]["id"] = "p"
    changed[0]["question"] = "Blocked question"
    assert "known_question" in m.reasons(changed, exclusions)
    changed[0]["question"] = "Allowed?"
    changed[0]["question_decomposition"][0]["id"] = "blocked-atom"
    assert "known_component" in m.reasons(changed, exclusions)
    assert m.select({"b": [False], "a": [True]}, count=2) == m.select(
        {"a": [False], "b": [True]}, count=2
    )


def test_public_projection_reuses_qualified_schema_without_any_gold_fields():
    import prepare_sufficiency_heldout as m

    raw = {
        "question": "Which entity?",
        "answer": "SECRET",
        "answer_aliases": ["SECRET"],
        "answerable": True,
        "id": "SECRET",
        "question_decomposition": [{"answer": "SECRET"}],
        "paragraphs": [
            {"idx": 99, "title": "Title", "paragraph_text": "Evidence", "is_supporting": True}
        ],
    }
    public = m.canonical._public(raw)
    assert public == {
        "question": "Which entity?",
        "documents": [{"docid": "d0", "title": "Title", "text": "Evidence"}],
    }


def test_provenance_hashes_module_filename_strings_with_path_only_helper():
    from pathlib import Path

    import prepare_sufficiency_heldout as m

    hashes = m.source_hashes()
    assert hashes[str(Path(m.canonical.__file__).resolve())] == m.panel.sha256(
        Path(m.canonical.__file__)
    )


def test_prompt_lengths_request_token_sequence_not_mapping():
    import prepare_sufficiency_heldout as m

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            assert kwargs["enable_thinking"] is False
            assert messages[0]["content"] == m.baseline.prompt(case)
            return list(range(37))

    case = {"id": "fixture", "public": {"question": "Which?", "documents": []}}
    assert m.prompt_lengths(Tokenizer(), [case]) == {"fixture": 37}
