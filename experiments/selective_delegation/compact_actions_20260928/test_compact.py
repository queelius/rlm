"""Focused strict-interface tests; no model or GPU."""

import json
import unittest

import compact_bridge as compact


class CompactTests(unittest.TestCase):
    def test_full_ingredient_schema_is_rejected_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            compact.parse_action(
                json.dumps(
                    {
                        "action": "craft",
                        "ingredients": {"ore": 1},
                        "target_item": "product",
                        "output_count": 1,
                    }
                )
            )

    def test_compact_schema_is_accepted(self):
        value = {"action": "craft", "target_item": "product", "output_count": 3}
        try:
            actual = compact.parse_action(json.dumps(value))
        except ValueError:
            actual = None
        self.assertEqual(actual, value)

    def test_unobserved_compact_craft_is_explicitly_rejected(self):
        frame = compact.Frame(compact.load_world(), {}, {"m0_i1": 1}, compact.Budget(), 0)
        with self.assertRaisesRegex(ValueError, "observed"):
            frame.apply({"action": "craft", "target_item": "m0_i1", "output_count": 1})
        self.assertEqual(frame.inventory, {})

    def test_binding_preserves_target_quantity_and_does_not_mutate_inputs(self):
        action = {"action": "craft", "target_item": "product", "output_count": 6}
        observed = {"product": {"recipes": [{"ingredients": {"ore": 2}, "result_count": 3}]}}
        bound = compact.bind_observed(action, observed)
        self.assertEqual(bound, {**action, "ingredients": {"ore": 4}})
        self.assertNotIn("ingredients", action)
        self.assertEqual(observed["product"]["recipes"][0]["ingredients"], {"ore": 2})
        with self.assertRaisesRegex(ValueError, "divisible"):
            compact.bind_observed({**action, "output_count": 5}, observed)
        with self.assertRaisesRegex(ValueError, "uniquely observed"):
            compact.bind_observed(action, {"product": {"recipes": []}})
        with self.assertRaisesRegex(ValueError, "uniquely observed"):
            compact.bind_observed(
                action, {"product": {"recipes": observed["product"]["recipes"] * 2}}
            )

    def test_bad_count_duplicate_and_note_schema_stay_strict(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                compact.parse_action(
                    json.dumps({"action": "craft", "target_item": "x", "output_count": value})
                )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compact.parse_action(
                '{"action":"craft","target_item":"x","output_count":1,"output_count":2}'
            )
        with self.assertRaises(ValueError):
            compact.parse_action('{"action":"finish","message":"done","extra":1}')


if __name__ == "__main__":
    unittest.main()
