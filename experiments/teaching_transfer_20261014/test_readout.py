"""Pin prospective evaluation slots and actor identity without loading a GPU."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("readout.py")
    assert path.exists(), "fresh readout adapter missing"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("fresh_teaching_readout", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_jobs_keep_all_eight_tasks_and_two_identical_draws_across_arms():
    m = module()
    tasks = [{"id": f"fresh{i}"} for i in range(8)]
    a, b = m.jobs(tasks, "repair"), m.jobs(tasks, "known")
    assert len(a) == 16
    assert [(j["task_id"], j["repeat"], j["seed"]) for j in a] == [
        (f"fresh{i}", repeat, seed)
        for i in range(8)
        for repeat, seed in enumerate([2026101403, 2026101404])
    ]
    assert [{k: v for k, v in j.items() if k != "condition"} for j in a] == [
        {k: v for k, v in j.items() if k != "condition"} for j in b
    ]
    with pytest.raises(ValueError, match="eight unique"):
        m.jobs(tasks[:-1] + tasks[:1], "known")


def test_actor_validation_rejects_modified_weights_without_rehashing_optimizer(tmp_path):
    m = module()
    checkpoint = tmp_path / "checkpoint-0023"
    checkpoint.mkdir()
    (tmp_path / "PLAN.json").write_text("{}")
    (checkpoint / "STATE.json").write_text('{"step":23}')
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"frozen-actor")
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    files = {p.name: digest(p) for p in checkpoint.iterdir()}
    (checkpoint / "COMMIT.json").write_text(json.dumps({"step": 23, "files": files}))
    binding = {
        "path": str(checkpoint),
        "sha256": files["adapter_model.safetensors"],
        "commit_sha256": digest(checkpoint / "COMMIT.json"),
        "state": {"step": 23},
        "training_plan_sha256": digest(tmp_path / "PLAN.json"),
    }
    m.validate_binding(binding)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"other-actor")
    with pytest.raises(ValueError, match="actor"):
        m.validate_binding(binding)
