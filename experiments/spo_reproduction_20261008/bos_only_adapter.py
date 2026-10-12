"""Private BOS-only intervention; never edits the published SPO tree.

This restores the configured query-BOS operation before MC actor likelihoods,
cutpoints, advantages and PPO. EOS insertion remains intentionally unchanged.
It is not evidence of the original authors' effective historical implementation.
"""

from functools import wraps
import hashlib
import importlib
from pathlib import Path
import sys

SOURCE = Path('/project/alex_phd/research-cache/repos/SPO')
PINS = {
    'src/treetune/episode_generators/episode_generator_with_reward_function.py': '2d8542258523177686d7f301e207c346594127e32f34b17364de1b87c1ce7071',
    'src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py': 'e098eee0f10381bb46c66778c20d1a6434b2294ca9db9f059dc435b048054cde',
    'src/treetune/trainers/data_collator.py': 'bda9274e01adca7343afcf38c71873186b8d242d9ba505abe478d7232690c2b0',
    'src/treetune/trainers/ppo_trainer.py': '231c8a1456b2b7c5fcda6e8a3f4112f8465d9e401779290c7fe7d4a890dbbd01',
    'src/treetune/main.py': '2fac3bc68d39d8d8f16210ba047ed626ccd75828c4934d98a7c1ee8a59e396c2',
    'src/treetune/runtime/policy_iteration_runtime.py': 'c2aa807d4c678bae031b1d5766b9d29071334e7ac3ad95c6532c8e0bb97273c9',
    'src/treetune/trainers/deepspeed_policy_trainer.py': '7cf1e24baad23a4a80cf9befcaca54f42979702a92b8f26355892cdaa44a9104',
}


def verify_source(source=SOURCE):
    """Fail closed if the small audited scientific call path changed."""
    actual = {}
    for relative, expected in PINS.items():
        digest = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f'SPO source hash mismatch: {relative}: {digest}')
        actual[relative] = digest
    return actual


def install_bos_only():
    verify_source()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE / 'src'))
    module = importlib.import_module('treetune.episode_generators.episode_generator_with_reward_function')
    expected_path = SOURCE / 'src/treetune/episode_generators/episode_generator_with_reward_function.py'
    if Path(module.__file__).resolve() != expected_path.resolve():
        raise RuntimeError(f'Unexpected imported SPO module: {module.__file__}')
    cls = module.EpisodeGeneratorWithRewardFunction
    original = cls._tokenize_trajectory
    if getattr(original, '_spo_private_bos_only', False):
        return

    @wraps(original)
    def tokenize_with_configured_bos(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not self._should_append_bos_to_query():
            return result
        bos = self.tokenizer.bos_token_id
        if bos is None:
            raise ValueError('Configured query BOS requires a tokenizer BOS ID')
        # Offsets still refer to raw text: do not fabricate a character span.
        # Active MC cutpoints/advantages index response tokens, not these offsets.
        return ([bos] + result[0], *result[1:])

    tokenize_with_configured_bos._spo_private_bos_only = True
    cls._tokenize_trajectory = tokenize_with_configured_bos
