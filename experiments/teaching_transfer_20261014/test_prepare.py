"""Exercise the existing native constructor through the new task identity seam."""

import importlib.util
import sys
from pathlib import Path


def test_generated_task_is_native_solvable_and_has_no_starting_products():
    path = Path(__file__).with_name("prepare.py")
    assert path.exists(), "fresh preparation implementation missing"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("fresh_teaching_prepare", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    runtime = m.runtime()
    world = runtime.world(2026101401)
    task = m.build_task(world, "a1_i3_21", 2026101401)
    assert task["id"] == "teaching_structure.w2026101401.a1_i3_21"
    assert task["misc"]["target_items"] == {"a1_i3_21": 2}
    assert task["derived_world_seed"] == 2026101401
    assert all(world.is_base_item(i) for i in task["misc"]["initial_inventory"])
    assert runtime.prior.bridge.replay_gold(task, world)["native_score"] == 1
