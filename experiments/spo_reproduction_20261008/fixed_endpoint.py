"""Evaluate the already-saved exact 690-collection / 11040-update endpoint."""

from functools import wraps
import json
from pathlib import Path


def finalize_and_evaluate(runtime):
    """No extra optimizer update or checkpoint rewrite; fail on incomplete state."""
    import torch

    trainer = runtime.trainer
    if runtime.num_iterations != 1000 or runtime.early_stop_iteration != 690:
        raise RuntimeError('Fixed endpoint requires original1000 LR horizon and stop690')
    if runtime.distributed_state.num_processes != 1:
        raise RuntimeError('This audited endpoint adapter is single-GPU only')
    if (trainer.state.iteration, trainer.state.global_step) != (690, 11040):
        raise RuntimeError('Trainer did not finish exactly690 collections /11040 updates')
    actor = trainer._actor_engine
    if actor.global_steps != 11040 or actor.lr_scheduler.last_batch_iteration != 11039:
        raise RuntimeError('Live actor/scheduler state is not the fixed11040 endpoint')
    checkpoint = trainer.checkpoints_dir / 'ckpt--iter_0690--epoch_2.00--step_11040'
    actor_directory = checkpoint / 'actor/global_step11040'
    required = {
        'optimizer': actor_directory / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',
        'actor model': actor_directory / 'mp_rank_00_model_states.pt',
        'HF model': checkpoint / 'hf_pretrained/pytorch_model.bin',
        'trainer state': checkpoint / 'custom_checkpoint_0.pkl',
        'running scores': checkpoint / 'custom_checkpoint_10.pkl',
        'KL controller': checkpoint / 'custom_checkpoint_11.pkl',
    }
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Missing full-state {label}: {path}')
    if (checkpoint / 'actor/latest').read_text().strip() != 'global_step11040':
        raise RuntimeError('Saved actor latest tag does not name11040')
    if (trainer.checkpoints_dir / 'final').read_text().strip() != checkpoint.name:
        raise RuntimeError('Native final pointer does not name full690 checkpoint')
    state = torch.load(required['trainer state'], map_location='cpu')
    if state != {'global_step': 11040, 'epoch': 2.0, 'iteration': 690}:
        raise RuntimeError(f'Saved trainer state is not exact endpoint: {state}')
    # mmap inspects checkpoint metadata without rereading model tensor storage.
    model_state = torch.load(str(required['actor model']), map_location='cpu', mmap=True)
    if model_state['global_steps'] != 11040 or model_state['lr_scheduler']['last_batch_iteration'] != 11039:
        raise RuntimeError('Saved actor/scheduler metadata disagrees with endpoint')
    del model_state
    policy = checkpoint / 'hf_pretrained'
    # Permanent native checkpoints omit tokenizer files; native resume does this too.
    runtime.tokenizer.save_pretrained(policy)
    receipt = {
        'iteration': 690, 'optimizer_steps': 11040, 'scheduler_last_batch_iteration': 11039,
        'checkpoint': str(checkpoint), 'evaluation_policy': str(policy),
        'required_files_bytes': {str(path.relative_to(checkpoint)): path.stat().st_size for path in required.values()},
        'extra_optimizer_steps': 0, 'extra_checkpoint_saves': 0,
        'limits': ['Checks metadata and required file presence, not every tensor value.', 'Tokenizer metadata added before native evaluation; optimizer/model files unchanged.'],
    }
    with (runtime.exp_root / 'endpoint-validated.json').open('x') as output:
        json.dump(receipt, output, indent=2)
    runtime.evaluate(iteration=690, last_policy_path=policy)
    with (runtime.exp_root / 'endpoint-evaluated.json').open('x') as output:
        json.dump({**receipt, 'native_evaluate_returned': True}, output, indent=2)


def install_endpoint_hook():
    from treetune.runtime.policy_iteration_runtime import PolicyIterationRuntime

    original = PolicyIterationRuntime.run_iteration_loop
    if getattr(original, '_spo_fixed_endpoint', False):
        return

    @wraps(original)
    def run_and_evaluate_endpoint(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        finalize_and_evaluate(self)
        return result

    run_and_evaluate_endpoint._spo_fixed_endpoint = True
    PolicyIterationRuntime.run_iteration_loop = run_and_evaluate_endpoint
