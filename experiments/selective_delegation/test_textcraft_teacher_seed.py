def test_explicit_seed_context_changes_recipe_order_then_restores_original():
    import textcraft_teacher_seed as s

    old = s.recipe.SEED
    original = s.recipe.epoch_order(366, old, 0)
    with s.seed_override():
        assert s.recipe.SEED == 2026092291
        changed = s.recipe.epoch_order(366, s.recipe.SEED, 0)
        assert sorted(changed) == list(range(366)) and changed != original
    assert old == s.recipe.SEED


def test_existing_public_teacher_qualification_is_retained():
    import textcraft_teacher_seed as s

    prepared, identity = s.teacher_inputs("public")
    assert identity["manifest_sha256"] == s.public.MANIFEST_SHA
    assert identity["rows_sha256"] == s.public.read(prepared / "MANIFEST.json")["rows_sha256"]
    assert identity["teacher"] == "public"
