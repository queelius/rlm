"""Private native entry point. Preparation is not authorization to launch training."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
from bos_only_adapter import SOURCE, install_bos_only, verify_source

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'fresh_bos_only.jsonnet'
AUTHOR = Path('/project/alex_phd/runs/spo-reproduction-20261008/author-training-config.json')
AUTHOR_SHA256 = '894dce29ddf67d09a03e0de8f8f08de16c04f7cc876ded104c4870a21dfb2746'


def main():
    pins = verify_source()
    if hashlib.sha256(AUTHOR.read_bytes()).hexdigest() != AUTHOR_SHA256:
        raise RuntimeError('Recovered historical configuration hash changed')
    if sys.argv[1:] == ['--verify-only']:
        print(json.dumps({'source_pins': pins, 'author_config_sha256': AUTHOR_SHA256, 'gpu_or_training_started': False}))
        return
    if sys.argv[1:] != ['--configs', str(CONFIG), 'run_iteration_loop']:
        raise ValueError(f'Expected --configs {CONFIG} run_iteration_loop; no force_rerun or resume flags')
    stage = Path(os.environ['APP_DIRECTORY'])
    if not stage.is_absolute() or stage == Path('/'):
        raise ValueError('APP_DIRECTORY must be an absolute fresh run directory')
    if os.environ.get('APP_EXPERIMENT_NAME') != 'native':
        raise ValueError('APP_EXPERIMENT_NAME must be native')
    if os.environ.get('APP_SEED', '42') != '42':
        raise ValueError('Fresh reproduction seed must remain 42')
    if os.environ.get('APP_MINIMIZE_STORED_FILES') != 'False':
        raise ValueError('Set APP_MINIMIZE_STORED_FILES=False to retain full checkpoints')
    if (stage / 'native').exists():
        raise FileExistsError(f'Refusing automatic resume or overwrite: {stage / "native"}')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    install_bos_only()
    from bf16_zero_norm import install as install_zero_norm
    from fixed_endpoint import install_endpoint_hook
    zero_norm_receipt = install_zero_norm()
    install_endpoint_hook()
    print(json.dumps({'intervention': 'configured_query_BOS_only', 'robustness_adapter': zero_norm_receipt, 'EOS_behavior': 'published_source_unchanged', 'source_commit': '6c5f94723c20e4de68578545b8bb98186e956797', 'fresh_output': str(stage / 'native'), 'endpoint': 'verify full690/11040 then native evaluate; no extra update'}), flush=True)
    runpy.run_path(str(SOURCE / 'src/treetune/main.py'), run_name='__main__')


if __name__ == '__main__':
    main()
