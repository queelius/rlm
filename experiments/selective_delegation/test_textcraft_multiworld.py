import json
from pathlib import Path

import pytest


def test_fixed_world44_plan_binds_actual_collector_and_all_saved_slots():
    import textcraft_multiworld as m

    plan, tasks, binding = m.build(44, "2291", "public")
    template = json.loads(m.TEMPLATE.read_text())
    assert len(tasks) == 8 and len(plan["jobs"]) == 16
    assert m.normalize(plan["jobs"]) == m.normalize(template["jobs"])
    assert plan["source_sha256"][str(Path(m.c.__file__).resolve())] == m.c.inputs.sha(
        Path(m.c.__file__)
    )
    assert plan["world_seed"] == 44 and plan["training_seed"] == 2026092291
    assert binding == m.seed.endpoint("public")
    assert m.checked_world(plan) is not None
    assert plan["budget_seconds"] == 2700
    with pytest.raises(ValueError, match="world"):
        m.build(47, "2291", "public")


def test_original_and_second_seed_endpoints_are_distinct_and_complete():
    import textcraft_multiworld as m

    for teacher in m.seed.TEACHERS:
        first, second = m.binding("original", teacher), m.binding("2291", teacher)
        assert first["state"]["step"] == second["state"]["step"] == 23
        assert first["commit_sha256"] != second["commit_sha256"]
