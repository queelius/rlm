"""Explicit three-arm reporting must never select attempts or pool a partial cohort."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("summarize_correspondence.py")
    assert path.exists(), "Three-arm reporter is not implemented"
    spec = importlib.util.spec_from_file_location("correspondence", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(root, arm, seed, value=100, reason="completed"):
    path = root / f"{arm}-{seed}"
    path.mkdir()
    config = {
        "arm": arm,
        "seed": seed,
        "domain_name": "cartpole",
        "task_name": "swingup",
        "action_repeat": 8,
        "num_train_steps": 12500,
        "num_eval_episodes": 10,
        "evaluation_seeds": list(range(10000, 10010)),
    }
    if arm == "shuffled_curl":
        config["contrastive_control"] = {
            "kind": "deranged_encoded_key_rows",
            "labels": "diagonal_unchanged",
            "encoder_optimizer_steps": "upstream_two",
        }
    rows = [{"type": "start", "step": 0, "config": config}]
    for step in range(0, 12501, 500):
        score = value if step == 12500 else 8
        rows += [
            {
                "type": "eval_episode",
                "step": step,
                "env_steps": step * 8,
                "eval_seed": 10000 + i,
                "return": score + i,
            }
            for i in range(10)
        ]
        rows.append({"type": "eval_summary", "mean_return": 999})
    rows.append(
        {
            "type": "end",
            "reason": reason,
            "step": 12500,
            "env_steps": 100000,
            "updates": 11500,
            "eval_env_steps": 260000,
        }
    )
    save(path, config, rows)
    return (arm, seed, path)


def save(path, config, rows):
    (path / "config.json").write_text(json.dumps(config))
    (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_complete_nine_run_cohort_has_hand_derived_three_arm_means_and_differences(tmp_path):
    inputs = [
        fixture(tmp_path, arm, seed, score)
        for seed in (123, 456, 789)
        for arm, score in [("curl", 100), ("no_curl", 80), ("shuffled_curl", 10)]
    ]
    result = module().summarize(inputs)
    assert result["status"] == "completed"
    assert result["arms"]["shuffled_curl"]["mean_return"] == 14.5
    assert result["arms"]["curl"]["seed_sd"] == 0
    assert result["aggregate_differences"] == {
        "curl_minus_no_curl": 20.0,
        "shuffled_minus_curl": -90.0,
        "shuffled_minus_no_curl": -70.0,
    }
    assert len(result["runs"]) == 9 and len(result["groups"]) == 3


def test_partial_cohort_preserves_scores_but_has_no_pooled_arm_means(tmp_path):
    result = module().summarize([fixture(tmp_path, "shuffled_curl", 123, 0)])
    assert result["status"] == "partial"
    assert result["aggregate_differences"] is None
    assert all(arm["mean_return"] is None for arm in result["arms"].values())
    assert sum(run["status"] == "missing" for run in result["runs"]) == 8
    assert next(r for r in result["runs"] if r["status"] == "completed")["endpoint_return"] == 4.5


def test_duplicate_explicit_arm_seed_is_refused_even_for_identical_path(tmp_path):
    entry = fixture(tmp_path, "curl", 123)
    with pytest.raises(ValueError, match="duplicate"):
        module().summarize([entry, entry])


@pytest.mark.parametrize(
    "reason,exit_code,explicit_failure,status",
    [
        ("time_cap", 0, False, "incomplete"),
        ("signal_15", 0, False, "incomplete"),
        ("time_cap", 1, False, "failure"),
        ("time_cap", 0, True, "failure"),
    ],
)
def test_owner_receipt_distinguishes_clean_incomplete_from_failure(
    tmp_path, reason, exit_code, explicit_failure, status
):
    entry = fixture(tmp_path, "curl", 123, reason=reason)
    path = entry[2]
    if explicit_failure:
        with (path / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"type": "failure", "error": "RuntimeError"}) + "\n")
    (path / "result.json").write_text(
        json.dumps({"complete": False, "exit_code": exit_code, "reason": reason})
    )
    result = module().summarize([entry])
    run = next(r for r in result["runs"] if (r["arm"], r["seed"]) == ("curl", 123))
    assert run["status"] == status
    assert run["endpoint_return"] is None and result["aggregate_differences"] is None


@pytest.mark.parametrize(
    "mutation,status",
    [
        ("nonfinite", "invalid"),
        ("missing_episode", "incomplete"),
        ("failure", "failure"),
        ("resume", "incomplete"),
        ("wrong_counter", "incomplete"),
        ("wrong_start", "invalid"),
    ],
)
def test_invalid_or_unfinished_attempt_never_has_scientific_endpoint(tmp_path, mutation, status):
    entry = fixture(tmp_path, "curl", 123)
    path = entry[2]
    cfg = json.loads((path / "config.json").read_text())
    rows = [json.loads(s) for s in (path / "metrics.jsonl").read_text().splitlines()]
    if mutation == "nonfinite":
        rows[1]["return"] = float("nan")
    elif mutation == "missing_episode":
        rows.pop(-3)
    elif mutation == "failure":
        rows.append({"type": "failure", "error": "RuntimeError"})
    elif mutation == "resume":
        rows.append({"type": "resume", "step": 12500})
    elif mutation == "wrong_counter":
        rows[-1]["updates"] = 11499
    else:
        rows[1]["eval_seed"] = 10001
    save(path, cfg, rows)
    run = next(
        r for r in module().summarize([entry])["runs"] if r["arm"] == "curl" and r["seed"] == 123
    )
    assert run["status"] == status and run["endpoint_return"] is None


def test_scientific_configuration_mismatch_blocks_complete_group_pooling(tmp_path):
    inputs = [
        fixture(tmp_path, arm, seed)
        for seed in (123, 456, 789)
        for arm in ("curl", "no_curl", "shuffled_curl")
    ]
    path = inputs[-1][2]
    cfg = json.loads((path / "config.json").read_text())
    rows = [json.loads(s) for s in (path / "metrics.jsonl").read_text().splitlines()]
    cfg["discount"] = 0.5
    rows[0]["config"] = cfg
    save(path, cfg, rows)
    result = module().summarize(inputs)
    assert result["status"] == "partial" and result["aggregate_differences"] is None
    assert result["groups"][-1]["status"] == "config_mismatch"


@pytest.mark.parametrize("field", ["labels", "encoder_optimizer_steps"])
def test_shuffled_condition_cannot_hide_changed_labels_or_optimizer_work(tmp_path, field):
    entry = fixture(tmp_path, "shuffled_curl", 123)
    path = entry[2]
    cfg = json.loads((path / "config.json").read_text())
    rows = [json.loads(s) for s in (path / "metrics.jsonl").read_text().splitlines()]
    cfg["contrastive_control"][field] = "different"
    rows[0]["config"] = cfg
    save(path, cfg, rows)
    run = next(
        r
        for r in module().summarize([entry])["runs"]
        if r["arm"] == "shuffled_curl" and r["seed"] == 123
    )
    assert run["status"] == "invalid" and run["endpoint_return"] is None


def test_cli_writes_explicit_partial_report_and_three_seed_figure(tmp_path):
    pytest.importorskip("matplotlib")
    fitz = pytest.importorskip("fitz")
    entry = fixture(tmp_path, "shuffled_curl", 123)
    script = Path(__file__).with_name("summarize_correspondence.py")
    assert script.exists(), "Three-arm reporter is not implemented"
    output = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            entry[0],
            str(entry[1]),
            str(entry[2]),
            "--output",
            str(output),
            "--plot",
        ],
        check=True,
    )
    assert json.loads((output / "summary.json").read_text())["status"] == "partial"
    report = (output / "RESULTS.md").read_text()
    assert "Shuffled image matching" in report and "missing" in report and "No pooled" in report
    with fitz.open(output / "learning-curves.pdf") as pdf:
        text = pdf[0].get_text()
        spans = [
            span
            for block in pdf[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
        ]
        xlabel = next(span for span in spans if span["text"] == "Simulator steps used for training")
        legend = [
            span
            for span in spans
            if span["text"].startswith(
                (
                    "CURL with image matching",
                    "Same crops without image matching",
                    "Shuffled image matching",
                )
            )
        ]
        assert all(
            not fitz.Rect(xlabel["bbox"]).intersects(fitz.Rect(span["bbox"])) for span in legend
        )
    assert "Training seed 123" in text and "Training seed 789" in text
    assert "Shuffled image matching" in text
    assert "Partial cohort" in text
    assert (output / "learning-curves.png").stat().st_size > 1000
