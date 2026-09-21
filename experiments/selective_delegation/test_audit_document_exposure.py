from audit_document_exposure import collect_panel, document_key, document_role


def _case(case_id, documents, support_ids):
    return {
        "id": case_id,
        "documents": documents,
        "metadata": {"supporting_paragraph_ids": support_ids},
    }


def test_exact_ignores_document_id_and_preserves_title_only_and_role_counts():
    train_support = _case("train-a", [{"id": "a", "title": "T", "text": "same"}], ["a"])
    train_distractor = _case("train-b", [{"id": "b", "title": "T", "text": "same"}], [])
    train_documents, train_titles = {}, {}
    for train_case in (train_support, train_distractor):
        for document in train_case["documents"]:
            key = document_key(document)
            train_documents.setdefault(key, set()).add(document_role(train_case, document))
            title_roles = train_titles.setdefault(document["title"], set())
            title_roles.add(document_role(train_case, document))
    panel = [
        _case("opaque-one", [{"id": "x", "title": "T", "text": "same"}], ["x"]),
        _case(
            "opaque-two",
            [
                {"id": "y", "title": "T", "text": "same"},
                {"id": "z", "title": "T", "text": "different"},
            ],
            [],
        ),
    ]

    result = collect_panel(panel, train_documents, train_titles)

    assert result["exact_title_text"] == {"shared_unique_documents": 1, "parents_with_match": 2}
    assert result["title"] == {"shared_unique_titles": 1, "parents_with_match": 2}
    assert result["exact_match_occurrences_by_panel_role"] == {"distractor": 1, "support": 1}
    assert result["exact_match_occurrences_by_train_role"] == {"distractor": 2, "support": 2}
    assert result["parents_detail"] == [
        {
            "parent_id": "opaque-one",
            "exact_title_text": {"match_occurrences": 1, "shared_unique_documents": 1},
            "title": {"shared_unique_titles": 1},
            "exact_match_occurrences_by_panel_role": {"support": 1},
            "exact_match_occurrences_by_train_role": {"distractor": 1, "support": 1},
        },
        {
            "parent_id": "opaque-two",
            "exact_title_text": {"match_occurrences": 1, "shared_unique_documents": 1},
            "title": {"shared_unique_titles": 1},
            "exact_match_occurrences_by_panel_role": {"distractor": 1},
            "exact_match_occurrences_by_train_role": {"distractor": 1, "support": 1},
        },
    ]
