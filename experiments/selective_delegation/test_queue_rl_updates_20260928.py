"""Do not silently evaluate a failed optimizer's checkpoint as an RL result."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("queue_rl_updates_20260928.py")


def test_unusable_endpoint_records_skip_without_loading_model(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    (training / "SUMMARY.json").write_text(
        json.dumps({"endpoint_usable": False, "failure": "replay mismatch", "endpoint": None})
    )
    output = tmp_path / "readout"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--readout-from",
            str(training),
            "--mode",
            "raw",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    skipped = json.loads((output / "CONDITIONAL-SKIP.json").read_text())
    assert skipped["GPU_loaded"] is False
    assert skipped["training_summary"]["failure"] == "replay mismatch"
    assert not (output / "PLAN.json").exists()


def test_missing_training_summary_is_explicitly_unknown_not_a_measured_failure(tmp_path):
    output = tmp_path / "readout"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--readout-from",
            str(tmp_path / "missing"),
            "--mode",
            "binder",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    skipped = json.loads((output / "CONDITIONAL-SKIP.json").read_text())
    assert skipped["training_summary"] is None
    assert skipped["observed_episodes"] == 0
    assert skipped["native_successes"] is None
