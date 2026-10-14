"""Require ordinary A starts, paired seeds, and a complete fixed teaching endpoint."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).with_name("readout.py")
    assert path.exists(), "recovery readout missing"
    spec = importlib.util.spec_from_file_location("recovery_readout", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_warm_preparation_preserves_normal_starts_and_two_prospective_draws(tmp_path):
    m = module()
    plan, tasks = m.prepare(SimpleNamespace(arm="warm", output=tmp_path))
    assert len(tasks) == 8 and len(plan["jobs"]) == 16
    assert {j["seed"] for j in plan["jobs"]} == {202609290210, 202609290211}
    assert {j["task_id"] for j in plan["jobs"]} == {t["id"] for t in tasks}
    assert all(j["policy"] == "flat" for j in plan["jobs"])
    assert plan["prepared"].endswith("textcraft-fresh-train-20260928-001/train")
    assert plan["tasks_sha256"] == (
        "22e65d9017711dd117ef7884b5a7e51c6152fa791c47666382dda05b7407bf76"
    )
    assert plan["base_dtype"] == "float16"
    assert plan["budget_seconds"] == 5400


def test_endpoint_rejects_partial_fit_and_detects_modified_weights(tmp_path):
    m = module()
    (tmp_path / "SUMMARY.json").write_text('{"endpoint_usable":false}')
    with pytest.raises(ValueError, match="complete"):
        m.endpoint(tmp_path)
    cp = tmp_path / "boundaries/sample-0001/checkpoint-0001"
    cp.mkdir(parents=True)
    (tmp_path / "PLAN.json").write_text("{}")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    (cp / "STATE.json").write_text(
        json.dumps(dict(step=1, plan_sha256=sha(tmp_path / "PLAN.json")))
    )
    (cp / "adapter_model.safetensors").write_bytes(b"actor")
    (cp / "adapter_config.json").write_text("{}")
    (cp / "COMMIT.json").write_text(json.dumps(dict(files={p.name: sha(p) for p in cp.iterdir()})))
    summary = dict(
        complete=True, endpoint_usable=True, endpoint=str(cp), actual_optimizer_steps=[1]
    )
    (tmp_path / "SUMMARY.json").write_text(json.dumps(summary))
    assert m.endpoint(tmp_path)["sha256"] == sha(cp / "adapter_model.safetensors")
    (cp / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        m.endpoint(tmp_path)


def test_actual_native_prepare_only_entrypoint_imports_and_writes_plan(tmp_path, monkeypatch):
    m = module()
    fit = m.fit_module()
    fit.OUTPUT = tmp_path
    monkeypatch.setattr(m, "fit_module", lambda: fit)
    output = tmp_path / "eval-warm"
    m.run(SimpleNamespace(arm="warm", output=output, prepare_only=True))
    plan = json.loads((output / "PLAN.json").read_text())
    assert plan["planned_episodes"] == 16
    assert not list(output.glob("OWNER*"))
