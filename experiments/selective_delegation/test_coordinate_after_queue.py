import argparse
import hashlib
import json
import sys
import time

import coordinate_after_queue as gate
import psutil
import pytest


def fixture_args(tmp_path, pid, created):
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(dict(status="accepted", jobs=[dict(output=str(tmp_path / "science"))]))
    )
    invocation = tmp_path / "INVOCATION.json"

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    invocation.write_text(
        json.dumps(
            dict(
                pid=pid,
                started=created + 0.5,
                receipt=str(previous),
                receipt_sha256=digest(previous),
            )
        )
    )
    return argparse.Namespace(
        previous_invocation=invocation,
        previous_receipt=previous,
        previous_pid=pid,
        previous_create_time=created,
        previous_invocation_sha256=digest(invocation),
        previous_receipt_sha256=digest(previous),
        receipt=None,
        output=tmp_path / "next",
        prepare_only=True,
    )


def test_live_supervisor_blocks_without_owners_and_reused_pid_is_not_old_process(tmp_path):
    process = psutil.Process()
    args = fixture_args(tmp_path, process.pid, process.create_time())
    outputs, _ = gate.validate(args)
    assert not gate.ready(args.previous_pid, args.previous_create_time, outputs)
    assert gate.ready(args.previous_pid, args.previous_create_time - 100, outputs)
    gate.main(args)
    assert not args.output.exists()


def test_mixed_gpu_cpu_predecessor_checks_owned_outputs_but_waits_whole_supervisor(tmp_path):
    process = psutil.Process()
    args = fixture_args(tmp_path, process.pid, process.create_time())
    previous = json.loads(args.previous_receipt.read_text())
    previous["jobs"].append(dict(name="cpu-analysis", argv=[sys.executable, "analyze.py"]))
    args.previous_receipt.write_text(json.dumps(previous))
    args.previous_receipt_sha256 = gate.sha(args.previous_receipt)
    invocation = json.loads(args.previous_invocation.read_text())
    invocation["receipt_sha256"] = args.previous_receipt_sha256
    args.previous_invocation.write_text(json.dumps(invocation))
    args.previous_invocation_sha256 = gate.sha(args.previous_invocation)
    outputs, _ = gate.validate(args)
    assert outputs == [tmp_path / "science"]
    assert not gate.ready(process.pid, process.create_time(), outputs)
    assert gate.ready(process.pid, process.create_time() - 100, outputs)


def test_dead_supervisor_allows_failed_preflight_but_waits_for_extant_owner(tmp_path):
    dead = 1_000_000_000
    assert not psutil.pid_exists(dead)
    args = fixture_args(tmp_path, dead, time.time() - 10)
    outputs, _ = gate.validate(args)
    assert gate.ready(dead, args.previous_create_time, outputs)
    outputs[0].mkdir()
    (outputs[0] / "OWNER-x.json").write_text(json.dumps(dict(pid=dead, create_time=1)))
    assert not gate.ready(dead, args.previous_create_time, outputs)
    (outputs[0] / "TERMINAL-x.json").write_text(json.dumps(dict(failure="failed preflight")))
    assert gate.ready(dead, args.previous_create_time, outputs)


def test_executes_accepted_cpu_jobs_without_nonexistent_last_owner(tmp_path, monkeypatch):
    args = fixture_args(tmp_path, 1_000_000_000, time.time() - 10)
    args.receipt = tmp_path / "accepted.json"
    args.receipt.write_text(
        json.dumps(
            dict(
                status="accepted",
                maximum_seconds=300,
                jobs=[
                    dict(
                        name="cpu",
                        argv=[sys.executable, "-c", "print('real job')"],
                        cap_seconds=10,
                        pins={},
                        output=str(tmp_path / "never-owner"),
                    )
                ],
            )
        )
    )
    args.prepare_only = False
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 1000))
    gate.main(args)
    result = json.loads((args.output / "GATE-RESULT.json").read_text())
    assert result["records"][0]["returncode"] == 0
    assert "real job" in (args.output / "cpu.log").read_text()
    args.previous_create_time -= 100
    with pytest.raises(ValueError):
        gate.validate(args)
