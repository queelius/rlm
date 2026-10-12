"""Focused real-optimizer fixture; CPU by default, no pretrained-model allocation."""

import copy
import os
import tempfile
import unittest

import deepspeed
import torch
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

NATIVE_STEP = BF16_Optimizer.step
DEVICE = os.environ.get("SPO_FIXTURE_DEVICE", "cpu")


class ZeroNormBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("SPO_TEST_NATIVE_BF16") != "1":
            from bf16_zero_norm import install

            print(install(), flush=True)
        cls.directory = tempfile.TemporaryDirectory(prefix="spo-zero-norm-")
        deepspeed.init_distributed(
            dist_backend="nccl" if DEVICE == "cuda" else "gloo",
            init_method="file://" + cls.directory.name + "/distributed",
            auto_mpi_discovery=False,
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        BF16_Optimizer.step = NATIVE_STEP
        if hasattr(BF16_Optimizer, "_spo_zero_norm_receipt"):
            del BF16_Optimizer._spo_zero_norm_receipt
        torch.distributed.destroy_process_group()
        cls.directory.cleanup()

    def optimizer(self):
        parameter = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], dtype=torch.bfloat16, device=DEVICE)
        )
        optimizer = BF16_Optimizer(
            torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.1),
            {parameter: "weight"},
            clip_grad=1.0,
            grad_acc_dtype=torch.float32,
        )
        return parameter, optimizer

    def accumulate(self, parameter, optimizer, multiplier):
        for _ in range(4):
            optimizer.backward(parameter.float().sum() * multiplier / 4)

    def test_zero_gradient_step_preserves_adam_momentum_and_step(self):
        """Catches skipping a zero step, erasing momentum, or adding fake gradients."""
        parameter, optimizer = self.optimizer()
        self.accumulate(parameter, optimizer, 1.0)
        optimizer.step()
        master = optimizer.fp32_groups_flat_partition[0]
        before = master.detach().clone()
        reference = torch.nn.Parameter(before.clone())
        reference_optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0.1)
        # Clone tensors: optimizer.load_state_dict otherwise shares CPU state storage.
        reference_optimizer.load_state_dict(copy.deepcopy(optimizer.optimizer.state_dict()))
        reference.grad = torch.zeros_like(reference)
        reference_optimizer.step()

        self.accumulate(parameter, optimizer, 0.0)
        optimizer.step()

        self.assertEqual(float(optimizer._global_grad_norm), 0.0)
        self.assertEqual(float(optimizer.optimizer.state[master]["step"]), 2.0)
        self.assertFalse(torch.equal(master, before))  # Adam momentum/decay still acts.
        torch.testing.assert_close(master, reference, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.optimizer.state[master][key],
                reference_optimizer.state[reference][key],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(parameter, master.to(torch.bfloat16), rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(optimizer.fp32_groups_gradients_flat[0]), 0)

    def test_nonzero_step_is_identical_to_native(self):
        """Catches changes to normal clipping, Adam updates, or low-precision copies."""
        parameter, optimizer = self.optimizer()
        control_parameter, control = self.optimizer()
        for multiplier in (1.0, -3.0, 0.25):
            self.accumulate(parameter, optimizer, multiplier)
            self.accumulate(control_parameter, control, multiplier)
            optimizer.step()
            NATIVE_STEP(control)
            torch.testing.assert_close(
                optimizer.fp32_groups_flat_partition[0],
                control.fp32_groups_flat_partition[0],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(parameter, control_parameter, rtol=0, atol=0)

    def test_nonfinite_gradient_is_still_rejected(self):
        """Catches accidentally allowing the native non-finite norm sentinel (-1)."""
        for multiplier in (float("nan"), float("inf")):
            with self.subTest(multiplier=multiplier):
                parameter, optimizer = self.optimizer()
                before = optimizer.fp32_groups_flat_partition[0].detach().clone()
                self.accumulate(parameter, optimizer, multiplier)
                with self.assertRaises(AssertionError):
                    optimizer.step()
                self.assertEqual(float(optimizer._global_grad_norm), -1.0)
                torch.testing.assert_close(
                    optimizer.fp32_groups_flat_partition[0], before, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
