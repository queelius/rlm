"""Append reviewed September29 experiments to the unchanged active queue."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/"
    "a100-lora-roundtrip/gpu/training/.venv/bin/python"
)
BASE_PLANS = {
    42: "044448fd919e6d1f0c00f40eec5eabe8069bc96718319a10b82960dcfc93dbd9",
    50: "3ecd555109a9631ac19fb105fe3c59293ec2a9e11cf73397dea311375028654f",
}
DELEGATION_SHA = "25a8051b79804e5e393e18a1c8fb50941e10c2f9db680450f8d110dde978cd14"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jobs(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("reviewed receipt changed: " + str(path))
    jobs = json.loads(path.read_text())["jobs"]
    if not jobs:
        raise ValueError("empty prepared receipt")
    return [dict(job, pins={**job["pins"], str(path): expected_sha}) for job in jobs]


def combine(groups):
    jobs, names, outputs, pins = [], set(), set(), {}
    for group in groups:
        for job in group:
            if job["name"] in names:
                raise ValueError("duplicate job name: " + job["name"])
            names.add(job["name"])
            if output := job.get("output"):
                if output in outputs:
                    raise ValueError("duplicate scientific output: " + output)
                outputs.add(output)
            for path, digest in job["pins"].items():
                if path in pins and pins[path] != digest:
                    raise ValueError("conflicting input pin: " + path)
                pins[path] = digest
            jobs.append(job)
    return jobs


def base_jobs():
    entry = HERE.parent / "phi_base_restart_20260929/restart_v2.py"
    jobs = []
    for world, expected in BASE_PLANS.items():
        output = ROOT / f"textcraft-phi-base-w{world}-20260929-002"
        path = output / "PLAN.json"
        if sha(path) != expected:
            raise ValueError("reviewed base PLAN changed")
        plan = json.loads(path.read_text())
        prepared = Path(plan["prepared"])
        provenance = plan["restart_provenance"]
        paths = [
            path,
            prepared / "tasks.jsonl",
            prepared / "MANIFEST.json",
            Path(plan["model"]) / "local-research-manifest.json",
            ROOT / "textcraft-phi-transfer-20260928-001/QUALIFICATION.json",
            ROOT / "textcraft-phi-transfer-20260928-001/native-fixture-001/VERIFICATION.json",
            *(Path(provenance[k]) for k in ("original_plan", "failed_job", "unused_preparation")),
        ]
        pins = {**plan["source_sha256"], **plan["trusted_source"]["source_sha256"]}
        pins.update({str(p): sha(p) for p in paths})
        argv = [PYTHON, str(entry), "--world", str(world)]
        jobs.extend(
            [
                dict(
                    name=f"phi-base-restart-w{world}",
                    argv=argv,
                    output=str(output),
                    cap_seconds=1200,
                    pins=pins,
                ),
                dict(
                    name=f"phi-base-restart-w{world}-audit",
                    argv=["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *argv, "--audit"],
                    cap_seconds=300,
                    pins=pins,
                ),
            ]
        )
    return jobs


def main(args):
    sys.path.insert(0, str(HERE.parent))
    import queue_transfer_20260928 as queue

    phi = ROOT / "textcraft-phi-repair-20260929-001/PREPARED-JOBS.json"
    delegation = ROOT / "textcraft-delegation-complete-20260929-001/PREPARED-JOBS.json"
    jobs = combine(
        [
            load_jobs(phi, args.phi_receipt_sha256),
            load_jobs(delegation, DELEGATION_SHA),
            base_jobs(),
        ]
    )
    shared = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            Path(queue.__file__),
            HERE.parent / "run_followon_queue_20260928.py",
            queue.GENERIC,
        )
    }
    jobs = [dict(job, pins={**job["pins"], **shared}) for job in jobs]
    queue.accept(
        "evidence-followups-20260929-001",
        "information-first-tail-20260928-002",
        jobs,
        "Does matched-answer teaching repair transfer to Phi? Does one helper improve "
        "whole-task success? Restore two preflight-only base references "
        "without scientific changes.",
        "Phi: eight exposed roots, one fit seed, two worlds; no domain-transfer claim. "
        "Delegation: six attempts on ONE exposed root; fixed routing, not learned recursion. "
        "Base: two episodes/world, descriptive only. "
        "All previous sources and active jobs unchanged. "
        "Useful execution estimate 0.75–1.5 A100 hours; summed caps are not ETA.",
        launch=args.launch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-receipt-sha256", required=True)
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args())
