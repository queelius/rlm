"""CPU-only numerical fixtures; run with the pinned R1 environment's unittest."""

import ast
import copy
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

PACKAGE = Path(sys.prefix) / "lib/python3.10/site-packages/oat"


def adapter():
    path = Path(__file__).with_name("memory_adapter.py")
    assert path.exists(), "Memory adapter has not been implemented"
    spec = importlib.util.spec_from_file_location("memory_adapter_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_entropy(logits):
    return logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)


class EntropyAllocationShapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.softmax_token_counts = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._softmax.default:
            self.softmax_token_counts.append(args[0].shape[-2])
        return func(*args, **(kwargs or {}))


def cpu_learner(source, entropy):
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPOLearner"
    )
    names = {
        "learning_step",
        "get_completion_mask",
        "get_batch_logps",
        "compute_monte_carlo_advantages",
    }
    namespace = dict(
        torch=torch,
        np=np,
        defaultdict=defaultdict,
        logging=logging,
        time=time,
        List=list,
        entropy_from_logits=reference_entropy,
        chunked_entropy_from_logits=entropy,
        masked_mean=lambda values, mask, axis=None: (values * mask).sum(axis) / mask.sum(axis),
    )
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            method = textwrap.dedent(ast.get_source_segment(source, node))
            method = method.replace("torch.cuda.current_device()", '"cpu"')
            exec(compile(method, "<actual-learning-step-cpu-fixture>", "exec"), namespace)
    return type("CPULearner", (), {name: namespace[name] for name in names})()


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 4)
        self.projection = torch.nn.Linear(4, 17)

    def forward(self, input_ids, attention_mask):
        return {"logits": self.projection(self.embedding(input_ids))}


class Strategy:
    grad_acc_step = 2

    def __init__(self):
        self.microsteps = 0
        self.gradients = []

    def backward(self, loss, model, optimizer):
        (loss / self.grad_acc_step).backward()

    def get_gradient_norm(self, model):
        return torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))

    def optimizer_step(self, optimizer, model, scheduler):
        self.microsteps += 1
        if self.microsteps % self.grad_acc_step == 0:
            self.gradients.append([p.grad.clone() for p in model.parameters()])
            optimizer.step()
            optimizer.zero_grad()


class MemoryAdapterTests(unittest.TestCase):
    def test_chunked_entropy_preserves_values_and_gradients_for_noncontiguous_tail(self):
        module = adapter()
        torch.manual_seed(12)
        logits = torch.randn(2, 11, 17, dtype=torch.float64)[:, :9].requires_grad_()
        expected = reference_entropy(logits)
        with EntropyAllocationShapes() as allocations:
            actual = module.chunked_entropy_from_logits(logits, token_chunk_size=4)
        self.assertEqual(allocations.softmax_token_counts, [4, 4, 1])
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
        actual_grad = torch.autograd.grad(actual.sum(), logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)

    def test_actual_learning_step_preserves_updates_and_reported_statistics(self):
        module = adapter()
        original = (PACKAGE / "algorithms/ppo.py").read_text()
        patched = module.patch_ppo(original.encode()).decode()
        torch.manual_seed(43)
        initial = Policy()
        trajectory = dict(
            input_ids=torch.tensor(
                [[1, 2, 3, 4, 0], [1, 5, 6, 0, 0], [1, 2, 7, 8, 9], [1, 2, 10, 11, 0]]
            ),
            attention_mask=torch.tensor(
                [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
            ),
            rewards=[[1], [0], [0], [1]],
            prompt_ids_lens=[2, 1, 2, 2],
            loss_masks=[1] * 4,
        )
        learners, reports = [], []
        for source in (original, patched):
            learner = cpu_learner(source, module.chunked_entropy_from_logits)
            learner.args = SimpleNamespace(
                reward_scale=1,
                temperature=1,
                num_samples=2,
                critic_type="drgrpo",
                num_ppo_epochs=1,
                train_batch_size_per_device=1,
                beta=0,
                cliprange=0.2,
                reinforce_update=False,
            )
            learner.model, learner.ref_model = copy.deepcopy(initial), None
            learner.strategy, learner.scheduler = Strategy(), None
            learner.optimizer = torch.optim.SGD(learner.model.parameters(), lr=0.1)
            learner.masked_aggregator = lambda values, mask, axis: (values * mask).sum(axis) / 3000
            np.random.seed(42)
            reports.append(learner.learning_step(trajectory))
            learners.append(learner)
        self.assertEqual(len(learners[0].strategy.gradients), 2)
        self.assertEqual(len(learners[1].strategy.gradients), 2)
        for before, after in zip(
            learners[0].model.parameters(), learners[1].model.parameters(), strict=True
        ):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for first, second in zip(
            learners[0].strategy.gradients, learners[1].strategy.gradients, strict=True
        ):
            for before, after in zip(first, second, strict=True):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertGreater(reports[0]["policy_grad_norm"].item(), 0)
        for key in reports[0].keys() - {"get_grad_norm_time"}:
            torch.testing.assert_close(reports[0][key], reports[1][key], rtol=1e-6, atol=1e-7)

    def test_source_drift_rejected_and_copy_importable_without_shared_mutation(self):
        module = adapter()
        original = (PACKAGE / "algorithms/ppo.py").read_bytes()
        with self.assertRaisesRegex(ValueError, "hash"):
            module.patch_ppo(original + b"\n")
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "private-pythonpath"
            manifest = module.prepare_copy(PACKAGE, destination)
            self.assertEqual(manifest["changed_files"], ["algorithms/ppo.py"])
            code = (
                "import oat.algorithms.ppo as p; import torch; "
                "from pathlib import Path; "
                "assert str(Path(p.__file__).resolve()) == "
                f"{str(destination / 'oat/algorithms/ppo.py')!r}; "
                "assert p.chunked_entropy_from_logits(torch.zeros(1,3,7)).shape == (1,3); "
                "assert 'chunked_entropy_from_logits' in "
                "p.PPOLearner.learning_step.__code__.co_names; "
                "assert not torch.cuda.is_initialized(); "
                "print('PRIVATE_COPY_IMPORT_OK')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=60,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "2",
                    "PYTHONPATH": str(destination),
                    "LD_LIBRARY_PATH": str(Path(sys.base_prefix) / "lib")
                    + ":"
                    + os.environ.get("LD_LIBRARY_PATH", ""),
                },
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PRIVATE_COPY_IMPORT_OK", result.stdout)
            self.assertEqual((PACKAGE / "algorithms/ppo.py").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                module.prepare_copy(PACKAGE, destination)


if __name__ == "__main__":
    unittest.main()
