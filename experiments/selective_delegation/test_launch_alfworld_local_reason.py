import hashlib
import json
from argparse import Namespace

import pytest


def test_proposed_control_validates_but_cannot_launch_and_tampering_rejected(tmp_path):
    import launch_alfworld_local_reason as m

    hashes = {}
    for name in (
        f"{m.SOURCE}/alfworld_local_reason.py",
        f"{m.SOURCE}/alfworld_closed_loop.py",
        f"{m.SOURCE}/alfworld_probe.py",
        f"{m.SOURCE}/alfworld_bridge.py",
        f"{m.SOURCE}/launch_alfworld_local_reason.py",
        f"{m.OUTPUT}/PLAN.json",
        "alfworld-closed-loop-001/PLAN.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    decision = {
        "schema": "alfworld-local-reason-decision-v1",
        "source": m.SOURCE,
        "predecessor": "sufficiency-readout-positive_only-001",
        "status": "proposed",
        "sha256": hashes,
    }
    (tmp_path / "ALFWORLD-LOCAL-REASON-DECISION-001.json").write_text(json.dumps(decision))
    assert m.main(Namespace(root=tmp_path, validate_only=True)) == 0
    with pytest.raises(ValueError, match="proposal cannot launch"):
        m.main(Namespace(root=tmp_path, validate_only=False))
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        m.validate(tmp_path, decision)


def test_predecessor_requires_all_native_calls_but_not_all_protocol_valid():
    import launch_alfworld_local_reason as m

    summary = {
        "planned_variant_attempts": 128,
        "planned_pair_attempts": 64,
        "returned_valid": 120,
        "returned_protocol_invalid": 8,
        "missing_or_inference_unavailable": 0,
        "physical_cost": {"calls": 128, "failed_calls": 0, "unknown_usage_calls": 0},
    }
    m.require_complete(summary)
    summary["missing_or_inference_unavailable"] = 1
    with pytest.raises(RuntimeError, match="incomplete"):
        m.require_complete(summary)
