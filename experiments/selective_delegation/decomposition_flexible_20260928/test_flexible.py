"""Regression for the observed, unnecessarily rejected query order."""

import json
import sys
import unittest
from pathlib import Path

OLD = Path(__file__).resolve().parent.parent / "decomposition_20260928"
sys.path.insert(0, str(OLD))
import routing as original  # noqa: E402

try:
    import routing_flexible as routing
except ModuleNotFoundError:
    routing = original


class FlexibleTests(unittest.TestCase):
    def test_one_screening_slot_does_not_become_sixteen_unknowns(self):
        import run_flexible

        report = {"groups": {"flat": {"planned": 16, "observed": 0, "missing_or_unknown": 16}}}
        plan = {"jobs": [{"policy": "flat"}]}
        updated = run_flexible.correct_group_counts(report, plan)
        self.assertEqual(updated["groups"]["flat"]["planned"], 1)
        self.assertEqual(updated["groups"]["flat"]["missing_or_unknown"], 1)

    def frame(self, mode="fixed"):
        proxy = routing.make_bridge(mode)
        world = routing.native.load_world()
        frame = proxy.Frame(world, {"raw_a4": 22}, {"a8_i4_18": 2}, routing.native.Budget(), 1)
        return proxy, frame

    def test_valid_nonlexical_visible_query_executes(self):
        proxy, frame = self.frame()
        proxy.public_prompt(frame, [])
        frame.apply({"action": "get_info", "items": ["a8_i4_18"]})
        proxy.public_prompt(frame, [])
        reply = frame.apply({"action": "get_info", "items": ["a8_i2_18"]})
        self.assertEqual(reply[0]["item"], "a8_i2_18")
        self.assertIn("a8_i2_18", frame.observed)
        self.assertEqual(dict(frame.refusals), {})

    def test_query_is_suggested_not_required(self):
        proxy, frame = self.frame("flat")
        control = json.loads(proxy.public_prompt(frame, []).rsplit(routing.MARKER, 1)[1])
        self.assertIsNone(control["required_next_action"])
        self.assertEqual(control["decision"]["query"]["items"], ["a8_i4_18"])

    def test_delegate_refusals_still_stop_and_child_shares_state(self):
        proxy = routing.make_bridge("fixed")
        frame = proxy.Frame(None, {"branch": 1}, {"root": 1}, routing.native.Budget(), 1)
        frame.observed = {
            "root": {"recipes": [{"ingredients": {"branch": 3}, "result_count": 1}]},
            "branch": {"recipes": [{"ingredients": {"left": 1}, "result_count": 1}]},
        }
        for _ in range(2):
            proxy.public_prompt(frame, [])
            with self.assertRaisesRegex(ValueError, "requires"):
                frame.apply({"action": "view_inventory"})
        self.assertTrue(frame.routing()["abort_requested"])
        self.assertEqual(frame.refusals["delegate"], 2)
        proxy.public_prompt(frame, [])
        child = frame.apply({"action": "delegate", "targets": {"branch": 2}, "context": "x"})
        self.assertIs(child.inventory, frame.inventory)
        self.assertIs(child.budget, frame.budget)
        self.assertEqual(child.initial_inventory, {"branch": 1})
        self.assertIsNone(child.routing()["required_next_action"])

    def test_unrelated_error_history_does_not_count_as_delegate_refusal(self):
        proxy, frame = self.frame("flat")
        history = [{"feedback": "Rejected action: native failure"}] * 2
        control = json.loads(proxy.public_prompt(frame, history).rsplit(routing.MARKER, 1)[1])
        self.assertFalse(control["abort_requested"])


if __name__ == "__main__":
    unittest.main()
