"""CPU-only preparation of immutable parent-dispatched jobs; never launches a GPU job."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import cost_reward as r

PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)


def prepare() -> dict:
    pins, jobs = {}, []
    for mode in ("raw", "binder"):
        r.prepare_training(
            SimpleNamespace(
                collection=r.FRESH_STUDY / mode / "collect-0001",
                output=r.STUDY / mode / "train-0001",
                hours=2,
            )
        )
        pins.update(r.source_pins(mode))
        for path in (
            r.FRESH_STUDY / mode / "collect-0001/PLAN.json",
            r.FRESH_STUDY / mode / "train-0001/PLAN.json",
            r.STUDY / mode / "train-0001/PLAN.json",
        ):
            pins[str(path)] = r.f.sha(path)
    for group in ("train", "diagnostic"):
        r.f.dataset_binding(r.f.DATA / group)
        for name in ("MANIFEST.json", "tasks.jsonl"):
            path = r.f.DATA / group / name
            pins[str(path)] = r.f.sha(path)
    pins[str(r.f.DATA / "MANIFEST.json")] = r.f.CURRICULUM_SHA
    environment = r.FRESH_STUDY / "ENVIRONMENT-PREPARED.json"
    pins[str(environment)] = r.f.sha(environment)
    for kind in ("train", "readout"):
        for mode in ("raw", "binder"):
            jobs.append(
                dict(
                    name=f"error-cost-{mode}-{kind}-0001",
                    argv=[PYTHON, str(r.HERE / "stage_cost.py"), "--kind", kind, "--mode", mode],
                    output=str(r.STUDY / mode / f"{kind}-0001"),
                    cap_seconds=7500 if kind == "train" else 5700,
                    pins=pins,
                )
            )
    jobs.append(
        dict(
            name="error-cost-first-step-comparison",
            argv=[
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=",
                PYTHON,
                str(r.HERE / "analyze_cost.py"),
                "--output",
                str(r.STUDY / "COMPARISON.json"),
            ],
            cap_seconds=600,
            pins=pins,
        )
    )
    receipt = dict(
        status="prepared_for_parent_dispatch_only",
        study=str(r.STUDY),
        question="Does tiny error cost provide useful one-step learning beyond identical-batch "
        "terminal reward, or mainly teach cheap unsuccessful stopping?",
        future_input_admission="Both first v002 fresh collections must independently finish. "
        "Incomplete arms skip without partial training or fabricated failures.",
        GPU_launched=False,
        environment_receipt=str(environment),
        environment_receipt_sha256=r.f.sha(environment),
        jobs=jobs,
        expected_useful_hours="1.3–3h: each arm roughly15–40min training +25–50min diagnostic",
        scientific_stage_caps_hours=7,
        upper_job_caps_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
        dispatch="Parent must integrate these after prior GPU owners release; this file neither "
        "launches nor independently races the live fresh-four-cycle queue. It is a prepared job "
        "list, not an ACCEPTED receipt. Readout PLANs bind the actual cost checkpoints at run.",
        coordinator_lock=str(
            r.f.c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock"
        ),
        policy="Exactly one fresh-Adam changed-reward update from publiccp23 per interface. "
        "No repeated cost-objective reuse and no diagnostic-guided checkpoint selection. "
        "Warm and terminal-only first-step diagnostics are reused from their existing queue.",
    )
    r.f.persist(r.STUDY / "PREPARED-JOBS.json", receipt)
    return receipt


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    receipt = prepare()
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "study",
                    "GPU_launched",
                    "expected_useful_hours",
                    "upper_job_caps_hours",
                )
            },
            indent=2,
        )
    )
