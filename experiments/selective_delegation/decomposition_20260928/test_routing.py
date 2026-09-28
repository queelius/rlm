"""Public-only routing contracts, with no world or gold input."""

import unittest

import routing

TARGET = {"root": 1}
RECIPES = {
    "root": {"recipes": [{"ingredients": {"branch": 3}, "result_count": 1}]},
    "branch": {"recipes": [{"ingredients": {"left": 1, "right": 2}, "result_count": 2}]},
}


class RoutingTests(unittest.TestCase):
    def test_new_subgoal_is_shortage_not_total_requirement(self):
        row = routing.choose(TARGET, {}, {"branch": 1}, RECIPES, 80, 4096, 0, False)
        self.assertEqual(row["candidate"]["targets"], {"branch": 2})
        self.assertTrue(row["adaptive"])

    def test_immediately_craftable_branch_does_not_earn_helper(self):
        row = routing.choose(
            TARGET, {}, {"branch": 1, "left": 1, "right": 2}, RECIPES, 80, 4096, 0, False
        )
        self.assertIsNotNone(row["candidate"])
        self.assertFalse(row["adaptive"])
        self.assertEqual(row["reason"], "branch_already_craftable")

    def test_budget_and_existing_boundary_preclude_helper(self):
        row = routing.choose(TARGET, {}, {}, RECIPES, 10, 4096, 0, False)
        self.assertFalse(row["adaptive"])
        self.assertEqual(row["reason"], "reserve_root_budget")
        for depth, delegated in ((1, False), (0, True)):
            row = routing.choose(TARGET, {}, {}, RECIPES, 80, 4096, depth, delegated)
            self.assertIsNone(row["candidate"])

    def test_unknown_recipe_causes_public_query_not_world_lookup(self):
        row = routing.choose(TARGET, {}, {}, {}, 80, 4096, 0, False)
        self.assertEqual(row["query"], {"action": "get_info", "items": ["root"]})

    def test_observed_shared_shortage_precludes_helper(self):
        recipes = {
            **RECIPES,
            "root": {"recipes": [{"ingredients": {"branch": 3, "sibling": 1}, "result_count": 1}]},
            "sibling": {"recipes": [{"ingredients": {"left": 1}, "result_count": 1}]},
        }
        row = routing.choose(TARGET, {}, {}, recipes, 80, 4096, 0, False)
        self.assertFalse(row["adaptive"])
        self.assertEqual(row["reason"], "known_shared_stock_coupling")

    def test_charging_response_does_not_change_announced_boundary(self):
        proxy = routing.make_bridge("adaptive")
        budget = routing.native.Budget()
        budget.calls = 80  # Exactly16 calls remain when the instruction is emitted.
        frame = proxy.Frame(None, {}, TARGET, budget, 1)
        frame.observed = RECIPES
        proxy.public_prompt(frame, [])
        budget.charge(20)
        child = frame.apply({"action": "delegate", "targets": {"branch": 3}, "context": "x"})
        self.assertIs(child.inventory, frame.inventory)
        self.assertIs(child.budget, budget)
        self.assertEqual(child.initial_inventory, {})
        self.assertEqual(frame.initial_inventory, {})
        child.inventory["branch"] = 3
        self.assertEqual(frame.inventory["branch"], 3)


if __name__ == "__main__":
    unittest.main()
