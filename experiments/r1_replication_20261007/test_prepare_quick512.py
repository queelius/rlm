"""Focused native-predicate selection checks, without model or GPU dependencies."""

import unittest

from prepare_quick512 import first_eligible


class FirstEligibleTests(unittest.TestCase):
    def test_inclusive_boundary_retains_existing_order_and_skips_overlength(self):
        calls = []

        def tokenizer(text):
            calls.append(text)
            return {"input_ids": [0] * int(text.removeprefix("templated:"))}

        rows = [{"problem": str(length)} for length in (1025, 1024, 12, 1025, 20, 1)]
        selected, lengths, rejected = first_eligible(
            rows, tokenizer, lambda text: "templated:" + text, count=3
        )
        self.assertEqual(selected, [1, 2, 4])
        self.assertEqual(lengths, [1024, 12, 20])
        self.assertEqual(rejected, [0, 3])
        self.assertEqual(len(calls), 5)

    def test_insufficient_eligible_rows_fail(self):
        with self.assertRaisesRegex(ValueError, "Only1|Only 1"):
            first_eligible(
                [{"problem": "x"}], lambda text: {"input_ids": [1]}, lambda text: text, count=2
            )


if __name__ == "__main__":
    unittest.main()
