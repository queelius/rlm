"""CPU checks for fixed quickscreen endpoints and completed evaluation receipts."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HELPER = Path(__file__).with_name("quickscreen_receipt.py")


class QuickscreenReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="quickscreen-guard-")
        self.root = Path(self.tmp.name)
        self.log = self.root / "training.log"
        self.model = self.root / "native/debug_fixture/saved_models/step_00033"
        self.model.mkdir(parents=True)
        (self.model / "model.safetensors").touch()
        self.evals = self.model.parents[1] / "eval_results"
        self.evals.mkdir()
        (self.evals / "33_math.json").write_text(json.dumps([{}] * 64))

    def tearDown(self):
        self.tmp.cleanup()

    def helper(self):
        self.assertTrue(HELPER.exists(), "Quickscreen endpoint helper is missing")
        spec = importlib.util.spec_from_file_location("quickscreen_receipt", HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def log_rows(self, rounds, updates):
        self.log.write_text("finish learn()\n" * rounds + f"misc/policy_sgd_step': {updates}.0\n")

    def test_control_uses_32_updates_and_natural_label33(self):
        self.log_rows(32, 32)
        self.assertEqual(self.helper().endpoint(self.log, self.root / "native", 32), self.model)

    def test_reuse_uses_64_updates_and_same_natural_label33(self):
        self.log_rows(32, 64)
        self.assertEqual(self.helper().endpoint(self.log, self.root / "native", 64), self.model)

    def test_reuse_rejects_collection_count_as_optimizer_count(self):
        self.log_rows(32, 32)
        with self.assertRaises(ValueError):
            self.helper().endpoint(self.log, self.root / "native", 64)

    def test_partial_and_extra_rounds_cannot_be_completed_endpoints(self):
        for rounds in (31, 33):
            with self.subTest(rounds=rounds):
                self.log_rows(rounds, 32)
                with self.assertRaises(ValueError):
                    self.helper().endpoint(self.log, self.root / "native", 32)

    def test_wrong_terminal_label_is_rejected(self):
        self.log_rows(32, 32)
        self.model.rename(self.model.with_name("step_00032"))
        with self.assertRaises(ValueError):
            self.helper().endpoint(self.log, self.root / "native", 32)

    def test_missing_monitor_and_wrong_last_update_counter_are_rejected(self):
        self.log_rows(32, 64)
        with self.log.open("a") as handle:
            handle.write("misc/policy_sgd_step': 65.0\n")
        with self.assertRaises(ValueError):
            self.helper().endpoint(self.log, self.root / "native", 64)
        self.log_rows(32, 64)
        (self.evals / "33_math.json").unlink()
        with self.assertRaises(ValueError):
            self.helper().endpoint(self.log, self.root / "native", 64)

    def test_completed_receipt_records_actual_reuse_budget_and_both_scores(self):
        self.log_rows(32, 64)
        for condition in ("chat", "raw"):
            output = self.root / f"eval-{condition}"
            output.mkdir()
            rows = [{"task_name": "math", "reward": [1]}] * 77
            rows += [{"task_name": "math", "reward": [0]}] * 51
            (output / "model_eval_out_fixture.json").write_text(json.dumps(rows))
        subprocess.run(
            [
                sys.executable,
                str(HELPER),
                "record",
                "--root",
                str(self.root),
                "--native",
                str(self.root / "native"),
                "--updates",
                "64",
                "--arm",
                "reuse",
                "--status",
                "completed",
                "--phase",
                "completed",
            ],
            check=True,
        )
        receipt = json.loads((self.root / "result.json").read_text())
        self.assertTrue(receipt["completed"])
        self.assertEqual(receipt["expected_optimizer_updates"], 64)
        self.assertEqual(receipt["terminal_label"], "step_00033")
        self.assertEqual(receipt["evaluations"]["raw"]["correct"], 77)
        self.assertEqual(receipt["evaluations"]["chat"]["n"], 128)

    def test_capped_receipt_cannot_claim_completed_scores(self):
        subprocess.run(
            [
                sys.executable,
                str(HELPER),
                "record",
                "--root",
                str(self.root),
                "--updates",
                "32",
                "--arm",
                "control",
                "--status",
                "capped",
                "--phase",
                "training",
                "--exit-code",
                "124",
            ],
            check=True,
        )
        receipt = json.loads((self.root / "result.json").read_text())
        self.assertFalse(receipt["completed"])
        self.assertEqual(receipt["exit_code"], 124)
        self.assertNotIn("evaluations", receipt)

    def test_evaluation_summary_requires_complete_128_rows(self):
        target = self.root / "eval"
        target.mkdir()
        rows = [{"task_name": "math", "reward": [1.0]}] * 77
        rows += [{"task_name": "math", "reward": [0.0]}] * 51
        saved = target / "model_eval_out_fixture.json"
        saved.write_text(json.dumps(rows))
        self.assertEqual(
            self.helper().evaluation(target), {"correct": 77, "n": 128, "path": str(saved)}
        )
        saved.write_text(json.dumps(rows[:-1]))
        with self.assertRaises(ValueError):
            self.helper().evaluation(target)


if __name__ == "__main__":
    unittest.main()
