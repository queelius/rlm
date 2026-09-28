"""Unknown crossed endpoints must not become zero-reward models."""

import analyze_crossed_rl_20260928 as crossed


def test_gain_requires_both_observed_cells():
    keys = [("a", 0), ("a", 1)]
    result = crossed.contrast({keys[0]: 1, keys[1]: 0}, {keys[0]: 0}, keys)
    assert result["known"] == 1
    assert result["unknown"] == 1
    assert result["difference"] is None
    assert result["wins"] == 1


def test_gain_direction_is_new_minus_reference():
    keys = [("a", 0), ("a", 1)]
    result = crossed.contrast(dict.fromkeys(keys, 1), dict.fromkeys(keys, 0), keys)
    assert result["difference"] == 1
    assert result["wins"] == 2


def test_actual_prepared_readout_contracts_are_paired():
    plans = [
        crossed.p.read(crossed.STUDY / mode / "readout-warm/PLAN.json")
        for mode in ("raw", "binder")
    ]
    crossed.check_pairing(plans)
