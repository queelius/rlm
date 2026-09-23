"""Reserve-world configuration must preserve native slots and fixed endpoints."""

from pathlib import Path

import pytest


def test_reserve_world47_qualified_plan_and_no_primary_reuse():
    import textcraft_multiworld as m

    plan, tasks, adapter = m.build(47, "2291", "public")
    assert len(tasks) == 8 and len(plan["jobs"]) == 16
    assert plan["world_seed"] == 47
    assert "textcraft-reserve-inputs-001" in plan["prepared"]
    assert plan["source_sha256"][str(Path(m.c.__file__).resolve())] == m.c.inputs.sha(
        Path(m.c.__file__)
    )
    assert adapter["state"]["step"] == 23
    assert m.checked_world(plan) is not None
    with pytest.raises(ValueError, match="world"):
        m.build(44, "2291", "public")
