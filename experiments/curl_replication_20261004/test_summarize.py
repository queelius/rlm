"""Scientific accounting fixtures: missing or repeated observations must not become scores."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).with_name("summarize.py")
    assert path.exists(), "Result summarizer has not been implemented"
    spec = importlib.util.spec_from_file_location("curl_summarize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fixture(root, name, arm="curl", seed=123, reason="completed", returns=None):
    run = root / name
    run.mkdir()
    config = {
        "arm": arm,
        "seed": seed,
        "num_eval_episodes": 10,
        "evaluation_seeds": list(range(10000, 10010)),
    }
    (run / "config.json").write_text(json.dumps(config))
    records = [{"type": "start", "config": config}]
    for env_steps, values in [(0, [10] * 10), (100000, returns or list(range(100, 110)))]:
        records += [
            {
                "type": "eval_episode",
                "step": env_steps // 8,
                "env_steps": env_steps,
                "eval_seed": 10000 + i,
                "return": value,
            }
            for i, value in enumerate(values)
        ]
        records += [
            {
                "type": "eval_summary",
                "step": env_steps // 8,
                "env_steps": env_steps,
                "episodes": len(values),
                "mean_return": 999,
            }
        ]
    records += [{"type": "end", "reason": reason, "step": 12500, "env_steps": 100000}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run


def test_fixed_endpoint_uses_ten_episode_mean_and_excludes_pilot_and_incomplete(tmp_path):
    run_fixture(tmp_path, "reference")
    run_fixture(tmp_path, "control", "no_curl", returns=[90] * 10)
    run_fixture(tmp_path, "pilot-reference", seed=999)
    run_fixture(tmp_path, "interrupted", seed=456, reason="time_cap")
    run_fixture(tmp_path, "short_eval", seed=789, returns=[50] * 9)
    summary = load_module().summarize(tmp_path)
    assert summary["arms"]["curl"]["mean_return"] == 104.5
    assert summary["arms"]["curl"]["n_seeds"] == 1
    assert summary["pairs"] == [{"seed": 123, "curl": 104.5, "no_curl": 90.0, "difference": 14.5}]
    assert len(summary["runs"]) == 4
    assert sorted(r["status"] for r in summary["runs"]) == [
        "completed",
        "completed",
        "incomplete",
        "incomplete",
    ]


def test_repeats_and_resume_segments_do_not_count_as_extra_training_seeds(tmp_path):
    run_fixture(tmp_path, "original")
    run_fixture(tmp_path, "repeat", returns=[800] * 10)
    resumed = run_fixture(tmp_path, "resumed", "no_curl", seed=456)
    with (resumed / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({"type": "resume", "step": 12500}) + "\n")
    summary = load_module().summarize(tmp_path)
    assert summary["arms"]["curl"]["n_seeds"] == 0
    assert summary["arms"]["curl"]["mean_return"] is None
    assert summary["pairs"] == []
    assert len(summary["runs"]) == 4
    assert all(r["endpoint_return"] is None for r in summary["runs"] if r["arm"] == "curl")


def test_failure_and_truncated_json_are_reported_without_scientific_scores(tmp_path):
    failed = run_fixture(tmp_path, "failure")
    with (failed / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({"type": "failure", "error": "RuntimeError"}) + "\n")
    broken = run_fixture(tmp_path, "broken", seed=456)
    with (broken / "metrics.jsonl").open("a") as handle:
        handle.write('{"type":')
    summary = load_module().summarize(tmp_path)
    assert [r["status"] for r in summary["runs"]] == ["failure", "failure"]
    assert summary["arms"]["curl"]["n_seeds"] == 0


def test_checkpoint_batches_are_not_pooled_or_selected_by_best_score(tmp_path):
    run = run_fixture(tmp_path, "ambiguous")
    with (run / "metrics.jsonl").open("a") as handle:
        for i in range(10):
            handle.write(
                json.dumps(
                    {
                        "type": "eval_episode",
                        "step": 12500,
                        "env_steps": 100000,
                        "eval_seed": 10000 + i,
                        "return": 900,
                    }
                )
                + "\n"
            )
        handle.write(
            json.dumps({"type": "eval_summary", "step": 12500, "env_steps": 100000, "episodes": 10})
            + "\n"
        )
    summary = load_module().summarize(tmp_path)
    assert summary["arms"]["curl"]["n_seeds"] == 0


def test_failed_repeat_retains_failure_accounting(tmp_path):
    run_fixture(tmp_path, "original")
    failed = run_fixture(tmp_path, "failed_repeat")
    with (failed / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({"type": "failure", "error": "RuntimeError"}) + "\n")
    summary = load_module().summarize(tmp_path)
    assert sum(r["status"] == "failure" for r in summary["runs"]) == 1
    assert summary["arms"]["curl"]["n_seeds"] == 0


def test_cli_outputs_readable_summary_and_all_seed_curves(tmp_path):
    pytest.importorskip("matplotlib")
    run_fixture(tmp_path, "reference")
    output = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize.py")),
            "--runs",
            str(tmp_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    assert json.loads((output / "summary.json").read_text())["arms"]["curl"]["n_seeds"] == 1
    assert "1/3" in (output / "RESULTS.md").read_text()
    for suffix in ("pdf", "png"):
        assert (output / f"learning-curves.{suffix}").stat().st_size > 1000
