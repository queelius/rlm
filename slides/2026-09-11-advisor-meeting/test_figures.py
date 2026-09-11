"""Guard planned-denominator accounting and the portable two-model figure."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import figures
import fitz


class TwoModelFigureTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((figures.HERE / "data/claims.json").read_text())
        self.data["nested_batch_mistral"] = {
            "sizes": [8, 16, 32, 48, 64],
            "planned_denominators": [128, 256, 512, 768, 1024],
            "correct": [
                [77, 119, 190, 292, 355],
                [86, 165, 309, 453, 595],
                [89, 155, 291, 406, 532],
            ],
            "invalid_calls_by_format": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 1]],
            "planned_calls": 240,
            "available_calls": 240,
            "valid_calls": 237,
        }

    def test_invalid_batch_cannot_contribute_salvaged_correct_labels(self):
        data = copy.deepcopy(self.data)
        # A failed 64-record response leaves at most 960 strict successes.
        data["nested_batch_mistral"]["correct"][2][4] = 961
        with self.assertRaises(AssertionError):
            figures.validate(data)

    def test_rejects_a_denominator_from_a_different_batch_size(self):
        data = copy.deepcopy(self.data)
        data["nested_batch_mistral"]["planned_denominators"][4] = 768
        with self.assertRaises(AssertionError):
            figures.validate(data)

    def test_equal_work_table_uses_same_records_and_observed_block_times(self):
        work = self.data["equal_work"]
        self.assertEqual(work["planned_per_policy"], 768)
        self.assertEqual(sum(work["calls"]), 848)
        self.assertEqual(work["calls"], work["available_calls"])
        self.assertEqual(work["calls"], work["valid_calls"])
        source = (figures.HERE / "research-update.tex").read_text()
        for label, correct, seconds in zip(
            work["slide_labels"], work["correct"], work["seconds"], strict=True
        ):
            row = f"{label} & {100 * correct / 768:.0f}\\% & {seconds:.0f} seconds"
            self.assertIn(row, source)

    def test_crossing_keeps_all_planned_positions(self):
        crossing = self.data["output_order"]
        self.assertEqual(crossing["planned_per_policy"], 16 * 2 * 64)
        self.assertEqual(crossing["planned_calls"], 96)
        self.assertEqual(crossing["available_calls"], 96)
        self.assertEqual(crossing["valid_calls"], 96)
        self.assertEqual(crossing["correct"], [1704, 1687, 1668])

    def test_rebuild_emits_both_models_and_plain_language_legend(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "data").mkdir()
            (root / "data/claims.json").write_text(json.dumps(self.data))
            with patch.object(figures, "HERE", root):
                figures.main()
            output = root / "figures/batch-size-models.pdf"
            self.assertTrue(output.is_file(), "The two-model figure was not generated.")
            with fitz.open(output) as pdf:
                text = " ".join(page.get_text() for page in pdf)
            for phrase in (
                "Qwen3-4B",
                "Mistral-7B",
                "No matching tags",
                "Row numbers",
                "Arbitrary tags",
            ):
                self.assertIn(phrase, text)
            with fitz.open(root / "figures/batch-size-simple.pdf") as pdf:
                simple = " ".join(page.get_text() for page in pdf)
            for phrase in (
                "Qwen3-4B",
                "Mistral-7B",
                "Without matching names",
                "With matching names",
                "83%",
                "52%",
            ):
                self.assertIn(phrase, simple)
            self.assertNotIn("Row numbers", simple)


if __name__ == "__main__":
    unittest.main()
