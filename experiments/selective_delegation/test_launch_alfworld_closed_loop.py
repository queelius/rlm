import hashlib
import json
from argparse import Namespace

import launch_alfworld_closed_loop as launcher
import pytest


def test_fixed_predecessor_and_no_gpu_from_proposal(tmp_path):
    relative = [
        launcher.SOURCE + "/" + name
        for name in (
            "alfworld_closed_loop.py",
            "alfworld_probe.py",
            "alfworld_bridge.py",
            "launch_alfworld_closed_loop.py",
        )
    ] + [launcher.OUTPUT + "/PLAN.json"]
    hashes = {}
    for name in relative:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    decision = dict(
        schema="alfworld-closed-loop-decision-v1",
        source=launcher.SOURCE,
        predecessor="hotpot-fresh-direct-vote-001",
        status="proposed",
        sha256=hashes,
    )
    (tmp_path / "ALFWORLD-CLOSED-LOOP-DECISION-001.json").write_text(json.dumps(decision))
    assert launcher.main(Namespace(root=tmp_path, validate_only=True)) == 0
    with pytest.raises(ValueError, match="proposal cannot launch"):
        launcher.main(Namespace(root=tmp_path, validate_only=False))
    assert launcher.command(tmp_path)[-2:] == ["--hours", "1"]
    (tmp_path / relative[0]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        launcher.validate(tmp_path, decision)
