def test_zero_update_skip_requires_matching_committed_endpoint():
    import launch_sufficiency_heldout as m

    summary = {"actual_optimizer_steps": 0, "committed_sampled_blocks": 4, "endpoint": "fixed"}
    skip = {
        "reason": "zero_real_updates",
        "chain_resolved": True,
        "scientific_readout": False,
        "rl_endpoint": {"endpoint": "fixed", "state": {"step": 0, "sample_cursor": 4}},
    }
    assert m.zero_skip_matches(skip, summary)
    assert not m.zero_skip_matches(skip, {**summary, "actual_optimizer_steps": 1})
    assert not m.zero_skip_matches(skip, {**summary, "failure": "error"})
    assert not m.zero_skip_matches(skip, {**summary, "endpoint": "other"})


def test_full_planned_accounting_allows_protocol_not_missing():
    import launch_sufficiency_heldout as m

    g = {
        "planned_variant_attempts": 128,
        "missing_or_inference_unavailable": 0,
        "returned_valid": 120,
        "returned_protocol_invalid": 8,
        "physical_cost": {"calls": 128, "failed_calls": 0},
    }
    s = {
        "planned_calls": 384,
        "conditions": {c: dict(g) for c in ("warm_joint32", "rl_terminal", "matched_sft_terminal")},
    }
    assert m.complete(s)
    s["conditions"]["rl_terminal"]["missing_or_inference_unavailable"] = 1
    assert not m.complete(s)
