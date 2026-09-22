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


def test_explicit_mixed_depth_panel_keeps_fixed_counts_and_seeded_selection():
    import hashlib
    from collections import Counter

    import prepare_sufficiency_compositional as m

    pairs = {
        f"h{hop}-{i}": [{"question_decomposition": [None] * hop}]
        for hop in (2, 3, 4)
        for i in range(24)
    }
    selected = m.select(pairs, seed=2026092209, quotas={2: 16, 3: 8, 4: 8})
    assert Counter(len(pairs[p][0]["question_decomposition"]) for p in selected) == {
        2: 16,
        3: 8,
        4: 8,
    }
    for hop, count in ((2, 16), (3, 8), (4, 8)):
        expected = sorted(
            (f"h{hop}-{i}" for i in range(24)),
            key=lambda p: hashlib.sha256(f"2026092209:{p}".encode()).hexdigest(),
        )[:count]
        assert [
            p for p in selected if len(pairs[p][0]["question_decomposition"]) == hop
        ] == expected
