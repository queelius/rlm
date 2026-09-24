import textcraft_breadth_corrected as c


def test_actual_corrected_endpoint_changes_only_actor_identity():
    mode = (
        "binder"
        if "bind_observed_recipe_arguments" in c.Path(c.b.m.c.__file__).read_text()
        else "raw"
    )
    for seed in ("original", "2291"):
        old, tasks, _ = c.b.build(0, 42, seed, mode)
        new, newtasks, adapter = c.build(0, 42, seed, mode)
        assert tasks == newtasks
        assert c.b.m.normalize(old["jobs"]) == c.b.m.normalize(new["jobs"])
        assert old["budget_seconds"] == new["budget_seconds"] == 2700
        assert old["fixed_adapter"] != new["fixed_adapter"] == adapter["path"]
        assert "quantity-matched" in adapter["path"] and "checkpoint-0023" in adapter["path"]
        assert old["tasks_sha256"] == new["tasks_sha256"]
        assert old["world_sha256"] == new["world_sha256"]
