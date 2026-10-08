"""Local CPU command fixtures; GPU checks are injected, never real GPU jobs."""

import fcntl
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


def owner():
    path = Path(__file__).with_name("run_queue.py")
    assert path.exists(), "Serial native-job owner is missing"
    spec = importlib.util.spec_from_file_location("spo_run_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path, commands):
    return dict(
        root=str(tmp_path / "runs"),
        lock=str(tmp_path / "GPU.lock"),
        deadline=time.time() + 30,
        cleanup_grace=0.05,
        jobs=[
            dict(
                id=f"job-{i}",
                argv=[sys.executable, "-c", command],
                cwd=str(tmp_path),
                env={"QUEUE_TEST_SECRET": "do-not-publish"},
                seconds=2,
            )
            for i, command in enumerate(commands)
        ],
    )


def test_success_checks_gpu_before_and_after_each_job_and_omits_env_values(tmp_path):
    module = owner()
    cfg = config(
        tmp_path, ["print('first')", "import os;assert os.environ['WANDB_MODE']=='offline'"]
    )
    cfg["jobs"][1]["env"]["WANDB_MODE"] = "online"
    checks = []

    def idle():
        checks.append(len(checks))
        return []

    assert module.run(cfg, gpu_pids=idle) == 0
    assert len(checks) == 4
    for job in cfg["jobs"]:
        receipt = json.loads((Path(cfg["root"]) / job["id"] / "result.json").read_text())
        assert receipt["exit_code"] == 0 and receipt["complete"] is True
        assert receipt["status"] == "command_succeeded"
        assert receipt["argv"] == job["argv"]
        assert receipt["started"] <= receipt["ended"]
        assert receipt["pid"] > 0 and receipt["start_ticks"] > 0
        assert receipt["owner_pid"] > 0 and receipt["owner_start_ticks"] > 0
        assert receipt["scoped_cleanup"]["remaining_owned_members"] == []
        assert "do-not-publish" not in json.dumps(receipt)


def test_busy_gpu_refuses_command_before_creating_job_directory(tmp_path):
    module = owner()
    cfg = config(tmp_path, ["raise AssertionError('must not run')"])
    with pytest.raises(RuntimeError, match="GPU compute PIDs"):
        module.run(cfg, gpu_pids=lambda: [2032223])
    assert not (Path(cfg["root"]) / "job-0").exists()


def test_compute_pid_after_child_exit_stops_without_starting_followon(tmp_path):
    module = owner()
    cfg = config(tmp_path, ["pass", "raise AssertionError('must not run')"])
    checks = iter([[], [2032223]])
    assert module.run(cfg, gpu_pids=lambda: next(checks)) == 1
    receipt = json.loads((Path(cfg["root"]) / "job-0" / "result.json").read_text())
    assert receipt["exit_code"] == 0
    assert receipt["complete"] is False and receipt["gpu_pids_after"] == [2032223]
    assert not (Path(cfg["root"]) / "job-1").exists()


@pytest.mark.parametrize(
    "command,seconds,marker,want_code,want_complete",
    [
        ("import sys;sys.exit(7)", 2, None, 7, False),
        ("import time;time.sleep(30)", 0.2, None, 124, False),
        ("pass", 2, "missing.marker", 0, False),
        ("from pathlib import Path;Path('done.marker').touch()", 2, "done.marker", 0, True),
    ],
)
def test_failure_timeout_and_success_marker_control_completion(
    tmp_path, command, seconds, marker, want_code, want_complete
):
    module = owner()
    cfg = config(tmp_path, [command, "raise AssertionError('must not run after failure')"])
    cfg["jobs"][0]["seconds"] = seconds
    if marker:
        cfg["jobs"][0]["success_marker"] = str(tmp_path / marker)
    if want_complete:
        cfg["jobs"] = cfg["jobs"][:1]
    assert module.run(cfg, gpu_pids=lambda: []) == (0 if want_complete else 1)
    receipt = json.loads((Path(cfg["root"]) / "job-0" / "result.json").read_text())
    assert receipt["exit_code"] == want_code and receipt["complete"] is want_complete
    assert not (Path(cfg["root"]) / "job-1").exists()


def test_existing_job_directory_refuses_overwrite(tmp_path):
    module = owner()
    cfg = config(tmp_path, ["pass"])
    job = Path(cfg["root"]) / "job-0"
    job.mkdir(parents=True)
    (job / "result.json").write_text("previous artifact")
    with pytest.raises(FileExistsError):
        module.run(cfg, gpu_pids=lambda: [])
    assert (job / "result.json").read_text() == "previous artifact"


def test_held_shared_gpu_lock_refuses_command(tmp_path):
    module = owner()
    cfg = config(tmp_path, ["raise AssertionError('must not run')"])
    with Path(cfg["lock"]).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.run(cfg, gpu_pids=lambda: [])
    assert not (Path(cfg["root"]) / "job-0").exists()


def test_gpu_check_time_cannot_launch_past_allocation_deadline(tmp_path, monkeypatch):
    module = owner()
    cfg = config(tmp_path, ["raise AssertionError('must not run')"])
    clock = [100]
    cfg["deadline"] = 120
    monkeypatch.setattr(module.time, "time", lambda: clock[0])

    def slow_idle_check():
        clock[0] = 121
        return []

    assert module.run(cfg, gpu_pids=slow_idle_check) == 1
    assert not (Path(cfg["root"]) / "job-0").exists()


def test_expired_deadline_refuses_command(tmp_path):
    module = owner()
    cfg = config(tmp_path, ["raise AssertionError('must not run')"])
    cfg["deadline"] = time.time() - 1
    assert module.run(cfg, gpu_pids=lambda: []) == 1
    assert not (Path(cfg["root"]) / "job-0").exists()


def test_cli_dry_run_validates_without_lock_or_job_artifacts(tmp_path):
    module = owner()
    cfg = config(tmp_path, ["raise AssertionError('must not run')"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    result = subprocess.run(
        [sys.executable, module.__file__, "--config", str(path), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert "do-not-publish" not in result.stdout
    assert not Path(cfg["lock"]).exists() and not Path(cfg["root"]).exists()
    cfg["jobs"][0]["id"] = "../overwrite"
    path.write_text(json.dumps(cfg))
    bad = subprocess.run(
        [sys.executable, module.__file__, "--config", str(path), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert bad.returncode != 0
