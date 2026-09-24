import json

import pytest
import textcraft_breadth as b


def test_frozen_panels_disjoint_and_native_first_goal():
    selection = json.loads((b.INPUT / "SELECTION.json").read_text())
    ids = sum(selection["panels"], [])
    assert len(ids) == len(set(ids)) == 64
    mode = (
        "binder"
        if "bind_observed_recipe_arguments" in b.Path(b.m.c.__file__).read_text()
        else "raw"
    )
    plan, tasks, _ = b.build(0, 42, "original", mode)
    assert len(plan["jobs"]) == 16 and plan["budget_seconds"] == 2700
    assert {j["task_id"] for j in plan["jobs"]} == {t["id"] for t in tasks}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(b.m.c.BASE, local_files_only=True)
    qualified = b.m.panel.prior.qualify(tasks[0], b.m.checked_world(plan), tok)
    assert qualified["native_score"] == 1
    assert qualified["initial_prompt_tokens"] + 256 <= 8192
    with pytest.raises(ValueError, match="wrong raw/binder"):
        b.build(0, 42, "original", "raw" if mode == "binder" else "binder")
