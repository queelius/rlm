"""Fixed, hand-derived continuation accounting; no GPU or weight loading."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("summarize_continuations.py")
    assert path.exists(), "Continuation reporter is not implemented"
    spec = importlib.util.spec_from_file_location("continuations", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evaluation(step, env_steps, values):
    return [
        {
            "type": "eval_episode",
            "step": step,
            "env_steps": env_steps,
            "eval_seed": 10000 + i,
            "return": value,
        }
        for i, value in enumerate(values)
    ]


def fixture(tmp_path, name="curl", arm="curl", seed=123, score=800):
    root = tmp_path / "500k"
    child = root / name
    parent_run = tmp_path / "100k" / name / "run"
    config = {
        "arm": arm,
        "seed": seed,
        "num_train_steps": 12500,
        "init_steps": 1000,
        "action_repeat": 8,
        "domain_name": "cartpole",
        "task_name": "swingup",
        "num_eval_episodes": 10,
        "evaluation_seeds": list(range(10000, 10010)),
        "device": "cuda",
        "max_seconds": 7200,
        "checkpoint_seconds": 900,
    }
    write(parent_run / "config.json", config)
    checkpoint = parent_run / "latest.pt"
    checkpoint.write_bytes(b"trusted fixture; never load or hash large weights")
    end = {
        "type": "end",
        "reason": "completed",
        "step": 12500,
        "env_steps": 100000,
        "updates": 11500,
    }
    episode = {"type": "train_episode", "step": 12500, "truncated_by_budget": False}
    saved = {
        "type": "checkpoint",
        "reason": "completed",
        "step": 12500,
        "env_steps": 100000,
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }
    parent_records = [{"type": "start"}] + evaluation(0, 0, [10] * 10)
    parent_records += evaluation(12500, 100000, [200] * 10) + [episode, saved, end]
    (parent_run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in parent_records))
    stat = checkpoint.stat()
    parent = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": "f" * 64,
        "checkpoint_identity": {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino,
        },
        "config_path": str(parent_run / "config.json"),
        "config": config,
        "config_sha256": hashlib.sha256((parent_run / "config.json").read_bytes()).hexdigest(),
        "terminal_proof": {"end": end, "train_episode": episode, "checkpoint": saved},
    }
    write(
        child / "job.json",
        {
            "job": {
                "arm": arm,
                "seed": seed,
                "steps": 62500,
                "env_steps": 500000,
                "resume": str(checkpoint),
            },
            "parent": parent,
        },
    )
    child_config = {**config, "num_train_steps": 62500, "max_seconds": 3600}
    write(child / "run" / "config.json", child_config)
    records = [
        {"type": "resume", "step": 12500, "resume_path": str(checkpoint), "config": child_config}
    ]
    records += evaluation(12500, 100000, [999] * 10)
    records += evaluation(62500, 500000, list(range(score, score + 10)))
    records += [
        {"type": "end", "reason": "completed", "step": 62500, "env_steps": 500000, "updates": 61500}
    ]
    (child / "run" / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    write(child / "result.json", {"exit_code": 0, "complete": True, "reason": None})
    return root, child


def test_one_chain_is_one_seed_parent_join_once_and_fixed_endpoint_mean(tmp_path):
    root, _ = fixture(tmp_path)
    fixture(tmp_path, "control", "no_curl", score=700)
    summary = module().summarize(root)
    assert summary["arms"]["curl"]["n_seeds"] == 1
    assert summary["arms"]["curl"]["mean_return"] == 804.5
    assert summary["pairs"] == [{"seed": 123, "curl": 804.5, "no_curl": 704.5, "difference": 100.0}]
    curve = summary["runs"][0]["curve"]
    assert [p["env_steps"] for p in curve] == [0, 100000, 500000]
    assert [p["mean_return"] for p in curve] == [10, 200, 704.5]


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_parent",
        "changed_config",
        "missing_episode",
        "interrupted",
        "failed",
        "wrong_updates",
        "missing_config",
    ],
)
def test_invalid_or_incomplete_chains_never_produce_endpoint_scores(tmp_path, fault):
    root, child = fixture(tmp_path)
    records_path = child / "run" / "metrics.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    if fault == "wrong_parent":
        records[0]["resume_path"] = str(tmp_path / "unrelated.pt")
    elif fault == "changed_config":
        receipt = json.loads((child / "job.json").read_text())
        Path(receipt["parent"]["config_path"]).write_text("{}")
    elif fault == "missing_episode":
        records.pop(-2)
    elif fault == "interrupted":
        records[-1]["reason"] = "time_cap"
        write(child / "result.json", {"exit_code": 0, "complete": False, "reason": "runner_cap"})
    elif fault == "failed":
        records += [{"type": "failure", "error": "RuntimeError"}]
        write(child / "result.json", {"exit_code": 1, "complete": False})
    elif fault == "wrong_updates":
        records[-1]["updates"] = 61501
    else:
        (child / "run" / "config.json").unlink()
    records_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = module().summarize(root)
    assert summary["arms"]["curl"]["n_seeds"] == 0
    assert summary["arms"]["curl"]["mean_return"] is None
    assert summary["runs"][0]["endpoint_return"] is None
    if fault == "failed":
        assert summary["runs"][0]["status"] == "failure"
    if fault == "missing_config":
        assert summary["runs"][0]["status"] == "missing"


def test_repeated_child_attempts_are_retained_but_no_favorable_attempt_selected(tmp_path):
    root, _ = fixture(tmp_path)
    fixture(tmp_path, "repeat", score=950)
    summary = module().summarize(root)
    assert len(summary["runs"]) == 2
    assert all(r["curve"] for r in summary["runs"])
    assert all(r["endpoint_return"] is None for r in summary["runs"])
    assert summary["arms"]["curl"]["n_seeds"] == 0


def test_differing_scientific_configs_do_not_generate_primary_pair(tmp_path):
    root, _ = fixture(tmp_path)
    _, control = fixture(tmp_path, "control", "no_curl")
    receipt = json.loads((control / "job.json").read_text())
    parent_config = Path(receipt["parent"]["config_path"])
    config = json.loads(parent_config.read_text())
    config["discount"] = 0.9
    write(parent_config, config)
    receipt["parent"]["config"] = config
    receipt["parent"]["config_sha256"] = hashlib.sha256(parent_config.read_bytes()).hexdigest()
    write(control / "job.json", receipt)
    child_config = json.loads((control / "run" / "config.json").read_text())
    child_config["discount"] = 0.9
    write(control / "run" / "config.json", child_config)
    records = [
        json.loads(line) for line in (control / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    records[0]["config"] = child_config
    (control / "run" / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = module().summarize(root)
    assert summary["arms"]["no_curl"]["n_seeds"] == 1
    assert summary["pairs"] == []


def test_json_and_markdown_outputs_exist_without_plot_dependency(tmp_path):
    root, _ = fixture(tmp_path)
    report = module()
    output = tmp_path / "analysis"
    report.write_outputs(report.summarize(root), output)
    assert json.loads((output / "summary.json").read_text())["target_env_steps"] == 500000
    text = (output / "RESULTS.md").read_text()
    assert "continuation" in text.lower() and "804.50" in text
    assert "1/3" in text and "episode-boundary" in text


def test_partial_live_log_preserves_measured_curves_without_endpoint_score(tmp_path):
    root, child = fixture(tmp_path)
    path = child / "run" / "metrics.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + '\n{"type":')
    summary = module().summarize(root)
    run = summary["runs"][0]
    assert [point["env_steps"] for point in run["curve"]] == [0, 100000, 500000]
    assert run["endpoint_return"] is None


def test_missing_child_config_still_preserves_verified_parent_curve(tmp_path):
    root, child = fixture(tmp_path)
    (child / "run" / "config.json").unlink()
    summary = module().summarize(root)
    run = summary["runs"][0]
    assert [point["env_steps"] for point in run["curve"]] == [0, 100000]
    assert run["status"] == "missing"
