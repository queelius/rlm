def test_fixed_balanced_hash_selection_not_atom_or_document_filter():
    import hashlib

    import prepare_sufficiency_compositional as m

    pairs = {
        f"p{i}": [{"question_decomposition": [None] * hop}]
        for hop, start in ((3, 0), (4, 20), (2, 40))
        for i in range(start, start + 20)
    }
    selected = m.select(pairs)
    assert len(selected) == 32
    for hop, start in ((3, 0), (4, 20)):
        expected = sorted(
            [f"p{i}" for i in range(start, start + 20)],
            key=lambda p: hashlib.sha256(f"2026092205:{p}".encode()).hexdigest(),
        )[:16]
        assert [
            p for p in selected if len(pairs[p][0]["question_decomposition"]) == hop
        ] == expected


def test_exclusion_does_not_require_disjoint_components():
    import prepare_sufficiency_compositional as m

    assert not m.excluded("new", "new question", {"old"}, {"old question"})
    assert m.excluded("old", "new question", {"old"}, {"old question"})
    assert m.excluded("new", "old question", {"old"}, {"old question"})
