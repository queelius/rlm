"""Reject undeclared changes to the reused optimizer's LoRA seam."""

import unittest

import train


class PhiTrainingContractTests(unittest.TestCase):
    def test_only_declared_projection_names_are_changed(self):
        arguments = dict(r=8, lora_alpha=16, target_modules=train.QWEN_TARGETS)
        mapped = train.phi_lora_kwargs(arguments)
        self.assertEqual(mapped["target_modules"], train.qualify.TARGET_MODULES)
        self.assertEqual(mapped["r"], 8)
        self.assertEqual(mapped["lora_alpha"], 16)
        self.assertEqual(arguments["target_modules"], train.QWEN_TARGETS)

    def test_unexpected_old_recipe_does_not_silently_adapt(self):
        with self.assertRaises(ValueError):
            train.phi_lora_kwargs(dict(r=16, lora_alpha=16, target_modules=["lm_head"]))


if __name__ == "__main__":
    unittest.main()
