import copy

import prepare_sufficiency_panel as panel


def test_projection_is_label_blind_and_renumbers_documents():
    row = {
        "question": "Public?",
        "id": "SECRET",
        "answer": "SECRET",
        "answerable": True,
        "question_decomposition": [{"answer": "SECRET"}],
        "paragraphs": [
            {"idx": 99, "title": "B", "paragraph_text": "Text B", "is_supporting": True},
            {"idx": 13, "title": "A", "paragraph_text": "Text A"},
        ],
    }
    other = copy.deepcopy(row)
    other["answerable"] = False
    other["paragraphs"].reverse()
    actual = panel.public_context(row)
    assert actual == panel.public_context(other)
    assert set(actual) == {"question", "documents"}
    assert [d["docid"] for d in actual["documents"]] == ["d0", "d1"]
    assert all(set(d) == {"docid", "title", "text"} for d in actual["documents"])
    assert "SECRET" not in str(actual)


def test_selection_keeps_natural_hops_and_both_variants():
    pairs = {
        f"{hop}__{i}": [{"answerable": False}, {"answerable": True}]
        for i, hop in enumerate(["2hop", "3hop", "4hop"])
    }
    chosen = panel.select_pairs(pairs, 3)
    assert set(chosen) == set(pairs)
    assert chosen == sorted(pairs, key=panel.selection_key)
