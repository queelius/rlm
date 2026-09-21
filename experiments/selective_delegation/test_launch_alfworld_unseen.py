import pytest


def test_fresh_sufficiency_handoff_accepts_observed_protocol_not_missing():
    import launch_alfworld_unseen as m

    summary = {
        "planned_variant_attempts": 128,
        "returned_valid": 120,
        "returned_protocol_invalid": 8,
        "missing_or_inference_unavailable": 0,
        "physical_cost": {"calls": 128, "failed_calls": 0},
    }
    m.require_predecessor(summary)
    summary["missing_or_inference_unavailable"] = 1
    with pytest.raises(RuntimeError):
        m.require_predecessor(summary)


def test_downstream_complete_handoff_requires_all_seventy_two_observed():
    import launch_alfworld_unseen as m

    summary = {
        "planned_episodes": 72,
        "recorded_episodes": 72,
        "all_slots_recorded": True,
        "groups": {
            p: {"planned": 24, "observed": 24, "missing_or_unobserved": 0}
            for p in ("flat", "manager_worker", "local_reason")
        },
    }
    assert m.complete(summary)
    summary["groups"]["local_reason"]["observed"] = 23
    assert not m.complete(summary)
