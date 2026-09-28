"""Focused split/selection contracts; no model or GPU required."""

import unittest

import prepare


def task(number, root, depth=3, split="train"):
    return {
        "id": f"textcraft_synth.{split}.{number}",
        "goal": f"Craft {root}",
        "misc": {"target_items": {root: 1}, "max_depth": depth},
    }


class SelectionTests(unittest.TestCase):
    def test_hash_selection_is_order_independent_and_excludes_root_aliases(self):
        tasks = [task(i, f"root{i // 2}") for i in range(20)]
        structures = {t["id"]: {"dependency_chain": 3} for t in tasks}
        args = (structures, {tasks[2]["id"]}, {"root0", "root4"}, {3: 3}, 41)
        selected = prepare.select_groups(tasks, *args)
        self.assertEqual(selected, prepare.select_groups(list(reversed(tasks)), *args))
        rows = [t for group in selected.values() for t in group]
        self.assertEqual(list(map(len, selected.values())), [3, 3])
        roots = [next(iter(t["misc"]["target_items"])) for t in rows]
        self.assertEqual(len(set(roots)), 6)
        self.assertTrue(set(roots).isdisjoint({"root0", "root4"}))
        self.assertNotIn(tasks[2]["id"], {t["id"] for t in rows})

    def test_eval_source_identity_is_rejected_even_with_fresh_root(self):
        rows = [task(0, "new", split="val")]
        with self.assertRaisesRegex(ValueError, "official TRAIN"):
            prepare.select_groups(rows, {}, set(), set(), {3: 1}, 41)

    def test_insufficient_distinct_roots_does_not_relax_quota(self):
        rows = [task(0, "same"), task(1, "same")]
        structures = {t["id"]: {"dependency_chain": 3} for t in rows}
        with self.assertRaisesRegex(ValueError, "quota"):
            prepare.select_groups(rows, structures, set(), set(), {3: 1}, 41)


if __name__ == "__main__":
    unittest.main()
