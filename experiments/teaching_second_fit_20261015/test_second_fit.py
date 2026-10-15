"""Catch wrong-seed continuation, endpoint drift and broken actor pairing."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).with_name(name + '.py')
    assert path.exists(), 'second-fit implementation missing'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('second_fit_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_second_seed_original_and_two_new_epoch_orders():
    d = module('dose_common')
    directory, plan, receipt = d.original('known_recipe')
    assert plan['seed'] == 2026092291
    assert 'seed2026092291' in str(directory)
    batches = list(d.schedule(receipt['state']))
    assert len(batches) == 46
    assert batches[0] == (1, 0, [238, 158, 312, 337, 48, 21, 84, 226,
                                89, 335, 202, 101, 179, 334, 117, 141])
    assert batches[23] == (2, 0, [62, 341, 359, 123, 244, 49, 310, 4,
                                 50, 251, 253, 263, 264, 303, 215, 16])
    assert batches[-1] == (2, 352, [84, 77, 160, 1, 27, 149, 12,
                                    325, 36, 194, 334, 8, 274, 130])


def test_fixed69_requires_all46_continuation_updates():
    d = module('dose_common')
    state = dict(step=69, epoch=3, cursor=0, cumulative_microbatches=1098,
                 cumulative_target_tokens=26460)
    steps = [dict(step=i, rows=14 if i in (46, 69) else 16,
                  target_tokens=460 if i in (46, 69) else 380) for i in range(24, 70)]
    d.validate_dose(state, steps, 69)
    with pytest.raises(ValueError):
        d.validate_dose(state, steps[:-1], 69)
    with pytest.raises(ValueError):
        d.validate_dose({**state, 'step': 68}, steps, 69)


def test_readout_binds_existing_second_seed_actors_and_same_draws():
    m = module('readout')
    tasks = [dict(id=f't{i}') for i in range(8)]
    expected = [(f't{i}', j, seed) for i in range(8)
                for j, seed in enumerate([2026101403, 2026101404])]
    for arm in ('known23', 'repair23', 'discovery23'):
        binding = m.actor_binding(arm)
        plan = json.loads((Path(binding['path']).parent / 'PLAN.json').read_text())
        assert plan['seed'] == 2026092291
        assert binding['state']['step'] == 23
        assert [(x['task_id'], x['repeat'], x['seed']) for x in m.jobs(tasks, arm)] == expected


def test_readout_rejects_wrong_seed_and_wrong_endpoint():
    m = module('readout')
    m.check_actor_contract({'seed': 2026092291}, {'step': 23, 'epoch': 1, 'cursor': 0}, 23)
    with pytest.raises(ValueError):
        m.check_actor_contract({'seed': 2026092208}, {'step': 23, 'epoch': 1, 'cursor': 0}, 23)
    with pytest.raises(ValueError):
        m.check_actor_contract({'seed': 2026092291}, {'step': 68, 'epoch': 2, 'cursor': 352}, 69)
