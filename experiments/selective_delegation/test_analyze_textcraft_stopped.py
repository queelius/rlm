import copy

import pytest


def test_actual_stopped_plan_requires_explicit_pinned_amendment():
    import analyze_textcraft_endpoint_fresh as a

    path = a.reader.c.ROOT / "textcraft-fresh-stopped-rl1-001/PLAN.json"
    plan = a.reader.control.read(path)
    binding = a.binding_from_plan(plan, "rl")
    assert binding == plan["adapter"] and binding["actual_optimizer_steps"] == 1
    wrong = copy.deepcopy(plan)
    wrong.pop("stopped_run_amendment")
    with pytest.raises(ValueError, match="failed/capped/unusable"):
        a.binding_from_plan(wrong, "rl")
    wrong = copy.deepcopy(plan)
    wrong["stopped_run_amendment"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="amendment"):
        a.binding_from_plan(wrong, "rl")
