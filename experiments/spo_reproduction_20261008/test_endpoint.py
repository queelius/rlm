"""Finite CPU checkpoint fixture; native final pointer does not evaluate 11040."""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/project/alex_phd/research-cache/repos/SPO/src')
import torch
from treetune.trainers.ppo_trainer import PPOTrainer

MODE = 'adapter'


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='endpoint-fixture-', dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.ckpt = self.run / 'checkpoints/ckpt--iter_0690--epoch_2.00--step_11040'
        self.actor_dir = self.ckpt / 'actor/global_step11040'
        self.actor_dir.mkdir(parents=True)
        (self.ckpt / 'hf_pretrained').mkdir()
        torch.save({'global_step': 11040, 'iteration': 690, 'epoch': 2.0}, self.ckpt / 'custom_checkpoint_0.pkl')
        torch.save({}, self.ckpt / 'custom_checkpoint_10.pkl')
        torch.save({}, self.ckpt / 'custom_checkpoint_11.pkl')
        torch.save({'global_steps': 11040, 'lr_scheduler': {'last_batch_iteration': 11039}, 'module': {'tiny': torch.ones(1)}}, self.actor_dir / 'mp_rank_00_model_states.pt')
        torch.save({'tiny_optimizer_fixture': torch.ones(1)}, self.actor_dir / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt')
        torch.save({'tiny': torch.ones(1)}, self.ckpt / 'hf_pretrained/pytorch_model.bin')
        (self.ckpt / 'actor/latest').write_text('global_step11040')
        self.calls = []
        trainer = SimpleNamespace(
            state=SimpleNamespace(global_step=11040, iteration=690, epoch=2.0),
            cache_deepspeed_engines=True,
            checkpoints_dir=self.run / 'checkpoints',
            _actor_engine=SimpleNamespace(global_steps=11040, lr_scheduler=SimpleNamespace(last_batch_iteration=11039)),
            get_last_checkpoint=lambda **kwargs: SimpleNamespace(path=self.ckpt),
        )
        self.runtime = SimpleNamespace(
            trainer=trainer, exp_root=self.run, num_iterations=1000, early_stop_iteration=690,
            distributed_state=SimpleNamespace(is_main_process=True, num_processes=1),
            tokenizer=SimpleNamespace(save_pretrained=self.save_tokenizer),
            evaluate=lambda **kwargs: self.calls.append(kwargs),
        )
        # Exercise native pointer behavior first. It correctly names690 but performs no eval.
        PPOTrainer.save_final_checkpoint(trainer)

    def save_tokenizer(self, path):
        (path / 'tokenizer_config.json').write_text('{"fixture": true}')

    def finish(self):
        if MODE == 'adapter':
            from fixed_endpoint import finalize_and_evaluate
            finalize_and_evaluate(self.runtime)

    def test_exact_final_full_checkpoint_is_evaluated_without_optimizer_change(self):
        optimizer = self.actor_dir / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
        before = optimizer.read_bytes()
        self.finish()
        self.assertEqual(self.calls, [{'iteration': 690, 'last_policy_path': self.ckpt / 'hf_pretrained'}])
        self.assertEqual(self.runtime.trainer._actor_engine.global_steps, 11040)
        self.assertEqual(optimizer.read_bytes(), before)
        self.assertTrue((self.ckpt / 'hf_pretrained/tokenizer_config.json').exists())

    def test_rejects_wrong_actor_step_before_evaluation(self):
        self.runtime.trainer._actor_engine.global_steps = 11056
        with self.assertRaisesRegex(RuntimeError, 'actor'):
            self.finish()
        self.assertEqual(self.calls, [])

    def test_rejects_missing_optimizer_before_evaluation(self):
        (self.actor_dir / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt').unlink()
        with self.assertRaisesRegex(RuntimeError, 'optimizer'):
            self.finish()
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['native', 'adapter'])
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    MODE = args.mode
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EndpointTests))
    print(stream.getvalue())
    args.receipt.write_text(json.dumps({'mode': MODE, 'tests_run': result.testsRun, 'success': result.wasSuccessful(), 'failures': len(result.failures), 'errors': len(result.errors), 'output': stream.getvalue()}, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
