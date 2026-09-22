import json
import time

import eval_textcraft as collector


def task_fixture():
    return json.loads(
        (collector.ROOT / "textcraft-inputs-001/tasks.jsonl").read_text().splitlines()[0]
    )


def test_instruction_render_is_only_fixed_suffix_on_real_public_state():
    assert hasattr(collector, "render_prompt"), "explicit instruction profile missing"
    task = task_fixture()
    frame = collector.bridge.Frame(
        collector.bridge.load_world(),
        dict(task["misc"]["initial_inventory"]),
        task["misc"]["target_items"],
        collector.bridge.Budget(),
        max_depth=0,
    )
    history = [{"action": {"action": "get_info", "items": ["x"]}, "feedback": "public"}]
    original = collector.bridge.public_prompt(frame, history, goal=task["goal"])
    assert collector.render_prompt(frame, history, goal=task["goal"]) == original
    amended = collector.render_prompt(
        frame, history, goal=task["goal"], profile="instruction_control"
    )
    assert amended == original + collector.INSTRUCTION_REMINDER
    assert "inventory_at_task_start" in collector.INSTRUCTION_REMINDER
    assert "target_items" in collector.INSTRUCTION_REMINDER
    assert "finish" in collector.INSTRUCTION_REMINDER
    assert "gold_trajectory" not in amended


def test_instruction_tokens_participate_in_existing_context_cap(tmp_path):
    assert hasattr(collector, "render_prompt"), "explicit instruction profile missing"
    task = task_fixture()

    class ContextBoundaryClient:
        def ids(self, prompt):
            assert prompt.endswith(collector.INSTRUCTION_REMINDER)
            return [1] * 7937  # +256 exceeds8192; no model call is permitted.

        def call(self, spec):
            raise AssertionError("over-context request must not be sent")

    row = collector.episode(
        task,
        {
            "episode_id": "t00-r0-flat",
            "task_id": task["id"],
            "repeat": 0,
            "seed": collector.SEEDS[0],
            "policy": "flat",
        },
        ContextBoundaryClient(),
        collector.bridge.load_world(),
        tmp_path,
        time.time() + 30,
        profile="instruction_control",
    )
    assert row["status"] == "context_cap"
    assert row["observed"] and row["global_calls"] == 0


def test_fixed_flat_inventory_has_sixteen_slots_and_no_recursive_unknowns(tmp_path):
    assert hasattr(collector, "render_prompt"), "explicit instruction profile missing"
    plan, tasks = collector.prepare(
        collector.ROOT / "textcraft-inputs-001",
        tmp_path,
        0.75,
        profile="instruction_control",
    )
    assert len(tasks) == 8 and len(plan["jobs"]) == plan["planned_episodes"] == 16
    assert {j["policy"] for j in plan["jobs"]} == {"flat"}
    assert plan["seeds"] == list(collector.SEEDS)
    assert plan["max_native_calls"] == 1536 and plan["budget_seconds"] == 2700
    assert plan["instruction_reminder"] == collector.INSTRUCTION_REMINDER
    assert plan["initial_token_audit"]["max_prompt_plus_cap"] <= 8192
    summary = collector.summarize(tmp_path, plan)
    assert summary["planned_episodes"] == 16 and set(summary["groups"]) == {"flat"}
    assert summary["groups"]["flat"]["missing_or_unknown"] == 16
