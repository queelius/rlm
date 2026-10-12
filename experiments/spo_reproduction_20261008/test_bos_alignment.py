"""Focused CPU integration fixture for actual SPO tokenizer / native MC paths.

The only model stand-in supplies explicitly synthetic token likelihoods; no model
weights or GPU are loaded. Native tokenizer, MC request construction, advantages,
truncation and PPO collation are exercised unchanged.
"""

import argparse
import ast
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
SOURCE = Path('/project/alex_phd/research-cache/repos/SPO')
ROOT = Path(__file__).resolve().parent
NATIVE = Path('/home/atowell/research-runs/spo-reproduction-20261008/spo-historical-resume160-attempt3/native')
SFT = Path('/project/alex_phd/research-cache/spo/models/models--realtreetune--rho-1b-sft-GSM8K/snapshots/b28fda3216f8178c1f37c3c764196e3ca2439a30')
sys.path.insert(0, str(SOURCE / 'src'))
MODE = 'adapter'

from datasets import Dataset, disable_progress_bar
from transformers import AutoTokenizer
from treetune.episode_generators.math_episode_generator_with_mc_advantages import MathEpisodeGeneratorWithMCAdvantages
from treetune.trainers.data_collator import PPODataCollator

disable_progress_bar()


class FixedLikelihoodFixture:
    """Model-free likelihood boundary; low probability at response positions 0..10."""

    def __init__(self):
        self.seen_queries = []

    def get_episodes_w_actor_logps(self, dataset):
        logprobs = []
        for row in dataset:
            query = row['query_token_ids']
            response = row['response_token_ids']
            self.seen_queries.append(list(query))
            logprobs.append(
                [-0.01] * (len(query) - 1)
                + [math.log(0.5)] * 11
                + [math.log(0.99)] * (len(response) - 11)
            )
        return dataset.add_column('actor_shifted_log_probs', logprobs)


class BosAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODE == 'adapter':
            from bos_only_adapter import install_bos_only
            install_bos_only()
        cls.tokenizer = AutoTokenizer.from_pretrained(str(SFT), local_files_only=True)
        cls.saved = Dataset.load_from_disk(str(NATIVE / 'checkpoints/episodes__iter0490/w_actLogp'))
        cls.sampled = Dataset.load_from_disk(str(NATIVE / 'temp_episodes/iteration__0490/infer_results/process_00/traj_results_ds'))
        cls.request_ids = {}
        path = NATIVE / 'temp_episodes/iteration__0490/infer_results/process_00/vllm_server.log'
        for line in path.read_text(errors='replace').splitlines():
            if 'Received request' not in line or 'SamplingParams(n=8,' not in line:
                continue
            prompt = ast.literal_eval(line.split('prompt: ', 1)[1].split(', sampling_params:', 1)[0])
            ids = ast.literal_eval(line.split('prompt_token_ids: ', 1)[1].split(', lora_request:', 1)[0])
            cls.request_ids[prompt] = ids

    def generator(self):
        generator = object.__new__(MathEpisodeGeneratorWithMCAdvantages)
        generator.tokenizer = self.tokenizer
        generator.append_bos_to_query = True
        generator.append_eos_to_response = True  # Configured but intentionally still ineffective.
        generator.max_sequence_length = 2048
        generator.cutpoint_interval = 5
        generator.seed = 42
        generator._cloud_log = lambda *args, **kwargs: None
        generator.distributed_state = SimpleNamespace(
            process_index=0, num_processes=1, is_local_main_process=True,
            is_main_process=True, wait_for_everyone=lambda: None,
        )
        return generator

    def trajectory(self, generator, index=0):
        saved = self.saved[index]
        query, response, offsets = generator._tokenize_trajectory(saved, return_offsets=True)
        return {
            'instance_idx': index,
            'process_idx': 0,
            'query_text': saved['query_text'],
            'response_text': saved['response_text'],
            'full_text': saved['query_text'] + saved['response_text'],
            'query_token_ids': query,
            'response_token_ids': response,
            'offsets': offsets,
            'score': 1.0,
            'data_instance': {
                key: value for key, value in self.sampled[index // 8].items()
                if not key.startswith('_treetune')
            },
        }

    def test_training_query_matches_real_vllm_request_before_likelihoods(self):
        # Catches missing or duplicate BOS while preserving actual response IDs.
        generator = self.generator()
        for index in [0, 8, 16, 24]:
            with self.subTest(saved_row=index):
                saved = self.saved[index]
                trajectory = self.trajectory(generator, index)
                self.assertEqual(trajectory['query_token_ids'], self.request_ids[saved['query_text']])
                self.assertEqual(trajectory['response_token_ids'], saved['response_token_ids'])
                self.assertNotEqual(trajectory['response_token_ids'][-1], self.tokenizer.eos_token_id)
                raw = self.tokenizer(saved['query_text'] + saved['response_text'], add_special_tokens=False, return_offsets_mapping=True)
                self.assertEqual(trajectory['offsets'], raw['offset_mapping'])
                self.assertEqual(len(trajectory['offsets']) + 1, len(trajectory['query_token_ids']) + len(trajectory['response_token_ids']))

    def test_bos_reaches_native_MC_scoring_then_cutpoints_advantages_and_PPO(self):
        # Catches a late-only PPO insertion or a query/response index shift.
        generator = self.generator()
        generator.trainer = FixedLikelihoodFixture()
        original = self.trajectory(generator)
        self.assertGreater(len(original['response_token_ids']), 11)
        with tempfile.TemporaryDirectory(prefix='mc-fixture-', dir=ROOT) as temporary:
            result_dir = Path(temporary) / 'infer_results/process_00'
            result_dir.mkdir(parents=True)
            unique, requests, mapping, trajectories = generator._create_value_estimation_requests(
                [original], result_dir, iteration=0,
            )
            self.assertEqual(generator.trainer.seen_queries[0], self.request_ids[original['query_text']])
            trajectory = trajectories[0]
            self.assertEqual(trajectory['cutpoints'].tolist(), [-1, 4, 9])
            self.assertEqual(len(unique), 3)
            self.assertEqual(len(mapping), 3)
            self.assertEqual(requests[0]['query'], original['query_text'])
            for request, count in zip(requests[1:], [5, 10]):
                self.assertEqual(request['query'], original['query_text'] + self.tokenizer.decode(original['response_token_ids'][:count]))
                self.assertNotIn(self.tokenizer.bos_token, request['query'])
            trajectory['cutpoints_values'] = [0.25, 0.5, 0.75]
            trajectory['cutpoints_values_std'] = [0.0, 0.0, 0.0]
            values, advantages = generator._compute_token_advantages(trajectory)
            n_response = len(original['response_token_ids'])
            self.assertEqual(len(values), n_response)
            self.assertEqual(advantages, [0.25] * 11 + [0.0] * (n_response - 11))
            episode = {**trajectory, 'advantages': advantages, 'scores': 1.0}
            batch = PPODataCollator()([episode])
            query_len = len(original['query_token_ids'])
            self.assertEqual(batch['input_ids'][0, :query_len].tolist(), self.request_ids[original['query_text']])
            self.assertEqual(batch['labels'][0, :query_len].tolist(), [-100] * query_len)
            self.assertEqual(batch['labels'][0, query_len:].tolist(), original['response_token_ids'])
            self.assertEqual(batch['advantages'][0, query_len:].tolist(), advantages)
            self.assertEqual(batch['actor_shifted_log_probs'].shape[1], batch['input_ids'].shape[1] - 1)

    def test_native_truncation_counts_BOS_without_putting_it_in_response_text(self):
        # Catches an off-by-one sequence cap or literal BOS leaking into MC text.
        generator = self.generator()
        saved = self.saved[0]
        actual_query = self.request_ids[saved['query_text']]
        generator.max_sequence_length = len(actual_query) + 12
        truncated = generator._truncate_response_to_max_length(saved['query_text'], saved['response_text'])
        expected = self.tokenizer.decode(actual_query[1:] + saved['response_token_ids'][:12], skip_special_tokens=False, clean_up_tokenization_spaces=False)[len(saved['query_text']):]
        self.assertEqual(truncated, expected)
        self.assertNotIn(self.tokenizer.bos_token, truncated)
        query, response = generator._tokenize_trajectory({'query_text': saved['query_text'], 'response_text': truncated})
        self.assertLessEqual(len(query) + len(response), generator.max_sequence_length)

    def test_false_BOS_flag_preserves_native_sequence(self):
        generator = self.generator()
        generator.append_bos_to_query = False
        saved = self.saved[0]
        query, response = generator._tokenize_trajectory(saved)
        self.assertEqual(query, saved['query_token_ids'])
        self.assertEqual(response, saved['response_token_ids'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['native', 'adapter'], required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    MODE = args.mode
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BosAlignmentTests)
    )
    output = stream.getvalue()
    print(output)
    args.receipt.write_text(json.dumps({
        'mode': MODE, 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'success': result.wasSuccessful(), 'output': output,
        'fixture_rows': [0, 8, 16, 24], 'iteration': 490,
        'limits': ['Synthetic CPU likelihoods test index alignment, not numerical model agreement.', 'No model weights loaded; no GPU used.'],
    }, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
