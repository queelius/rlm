"""Focused label-blind selection checks for a future paired sufficiency panel."""

import prepare_sufficiency_fresh_panel as fresh


def test_selection_is_fixed_seed_hash_order_without_hop_or_label_ranking():
    pairs = {f"p{i}": [{"answerable": bool(i % 2)}] for i in range(4)}
    assert fresh.select_pairs(pairs, 2) == sorted(pairs, key=fresh.selection_key)[:2]


def test_document_overlap_receipt_distinguishes_exact_text_from_title_only():
    selected = [
        {"public": {"documents": [{"title": "A", "text": "same"}, {"title": "B", "text": "new"}]}},
        {"public": {"documents": [{"title": "B", "text": "new"}]}},
    ]
    train_documents = [("A", "same"), ("B", "changed")]
    receipt = fresh.document_overlap(selected, train_documents)
    assert receipt == {
        "selected_unique_documents": 2,
        "exact_title_text_shared_unique_documents": 1,
        "title_shared_unique_documents": 2,
        "selected_variants_with_exact_overlap": 1,
        "selected_variants_with_title_overlap": 2,
    }
