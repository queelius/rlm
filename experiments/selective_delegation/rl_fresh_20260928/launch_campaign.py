"""Prepare or explicitly launch the bounded fresh-TRAIN study after its predecessor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import fresh_common as f

PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)
SUPERVISOR = "/project/alex_phd/envs/prime-rl-5990b1b/bin/python"
GENERIC = f.ROOT / "source-queue-handoff-generic-004/run_independent_queue.py"
STUDY = f.ROOT / "textcraft-fresh-rl-20260928-002"
PREDECESSOR = f.ROOT / (
    "textcraft-teaching-order-20260928-001/eval-random_visible-s2026092208-p00-w50-raw"
)


def make_jobs(study: Path, updates: int) -> list[dict]:
    if updates not in range(1, 5):
        raise ValueError("one to four fixed prospective checkpoints")
    pins = {}
    for mode in ("raw", "binder"):
        pins.update(f.source_pins(f.LEGACY / "collect.py", mode))
        pins.update(f.source_pins(f.LEGACY / "train.py", mode))
    for filename in ("stage.py", "extra_sft.py", "launch_campaign.py", "compare.py"):
        path = f.HERE / filename
        pins[str(path)] = f.sha(path)
    pins[str(f.LEGACY / "compare.py")] = f.sha(f.LEGACY / "compare.py")
    pins[str(f.prior.LIBRARY / "train_textcraft_sft.py")] = f.sha(
        f.prior.LIBRARY / "train_textcraft_sft.py"
    )
    for group in ("train", "diagnostic"):
        f.dataset_binding(f.DATA / group)
        for name in ("MANIFEST.json", "tasks.jsonl"):
            path = f.DATA / group / name
            pins[str(path)] = f.sha(path)
    pins[str(f.DATA / "MANIFEST.json")] = f.CURRICULUM_SHA
    jobs = []

    def add(kind, mode="raw", actor="rl", update=1):
        suffix = f"{update:04d}"
        if kind == "collect":
            out, cap = study / mode / f"collect-{suffix}", 9300
        elif kind == "train":
            out, cap = study / mode / f"train-{suffix}", 7500
        elif kind == "sft":
            out, cap = study / "sft/train-0001", 3000
        elif actor == "warm":
            out, cap = study / mode / "readout-warm", 5700
        elif actor == "sft":
            out, cap = study / "sft" / mode / "readout-0001", 5700
        else:
            out, cap = study / mode / f"readout-{suffix}", 5700
        jobs.append(
            dict(
                name=f"fresh-{mode}-{actor}-{kind}-{suffix}",
                argv=[
                    PYTHON,
                    str(f.HERE / "stage.py"),
                    "--study",
                    str(study),
                    "--kind",
                    kind,
                    "--mode",
                    mode,
                    "--actor",
                    actor,
                    "--update",
                    str(update),
                ],
                output=str(out),
                cap_seconds=cap,
                pins=pins,
            )
        )

    for update in range(1, updates + 1):
        for kind in ("collect", "train"):
            for mode in ("raw", "binder"):
                add(kind, mode=mode, update=update)
        if update == 1:
            for mode in ("raw", "binder"):
                add("readout", mode=mode, actor="warm")
        for mode in ("raw", "binder"):
            add("readout", mode=mode, update=update)
        if update == 1:
            # Independent control still runs if both RL arms skip or fail.
            add("sft")
            for mode in ("raw", "binder"):
                add("readout", mode=mode, actor="sft")
        jobs.append(
            dict(
                name=f"fresh-comparison-{update:04d}",
                cap_seconds=600,
                pins=pins,
                argv=[
                    "/usr/bin/env",
                    "CUDA_VISIBLE_DEVICES=",
                    PYTHON,
                    str(f.HERE / "compare.py"),
                    "--study",
                    str(study),
                    "--update",
                    str(update),
                    "--output",
                    str(study / f"COMPARISON-{update:04d}.json"),
                ],
            )
        )
    return jobs


def main(args):
    jobs = make_jobs(args.study, args.updates)
    receipt = dict(
        status="accepted" if args.launch else "prepared_for_parent_dispatch",
        question="Does exact observed-recipe execution change terminal-reward learnability "
        "on new optimization goals, and does learning transfer to disjoint TRAIN roots?",
        authorized_by="September28 autonomous research; "
        "parent chooses dispatch after teacher controls",
        predecessor=str(args.predecessor.resolve()),
        maximum_seconds=64 * 3600,
        jobs=jobs,
        prospective_updates_per_arm=args.updates,
        expected_useful_hours="First cycle/control roughly5–8h; four cycles roughly14–24h "
        "if both arms remain informative. Zero-credit or failed arms reduce useful coverage.",
        upper_job_caps_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
        queue_bound_hours=64,
        dataset_manifest_sha256=f.CURRICULUM_SHA,
        order="Complete both independently admitted collection/update arms, independently "
        "run warm diagnostic controls, retain each fixed checkpoint/readout, then extraSFT. "
        "Later stages stop only their own arm after unusable optimizer boundaries.",
        selection="No diagnosticB outcome gate or favorable-checkpoint selection. "
        "All1..N checkpoint readouts remain planned regardless previous diagnostic success.",
        resource_policy="One unchanged coordinator lock per GPU stage; authenticated predecessor "
        "release and supervised per-job caps. Independent arm/control follows a failed job only "
        "after owner release; no transport retries or partial-batch updates.",
    )
    if not args.launch:
        f.persist(args.study / f"PREPARED-CAMPAIGN-{args.updates:04d}.json", receipt)
        print(
            json.dumps(
                dict(
                    jobs=len(jobs),
                    upper_job_caps_hours=receipt["upper_job_caps_hours"],
                    expected_useful_hours=receipt["expected_useful_hours"],
                    predecessor=receipt["predecessor"],
                    GPU_launched=False,
                ),
                indent=2,
            )
        )
        return
    for job in jobs:
        if not job.get("output"):
            continue
        path = Path(job["output"])
        if (
            list(path.glob("OWNER-*.json"))
            or (path / "CONDITIONAL-SKIP.json").exists()
            or (path / "calls").exists()
        ):
            raise ValueError("already attempted scientific stage: " + str(path))
    args.queue_output.mkdir(parents=True, exist_ok=False)
    receipt["allocation_end"] = os.environ["SLURM_JOB_END_TIME"]
    f.c.save(args.queue_output / "ACCEPTED.json", receipt)
    argv = [
        SUPERVISOR,
        str(GENERIC),
        "--receipt",
        str(args.queue_output / "ACCEPTED.json"),
        "--output",
        str(args.queue_output / "queue"),
    ]
    with (args.queue_output / "supervisor.log").open("x") as stream:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    f.c.save(args.queue_output / "LAUNCH.json", dict(pid=child.pid, started=time.time(), argv=argv))
    print(json.dumps(dict(pid=child.pid, receipt=str(args.queue_output / "ACCEPTED.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--predecessor", type=Path, default=PREDECESSOR)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--queue-output", type=Path, default=f.ROOT / "fresh-rl-queue-20260928-002")
    parser.add_argument("--launch", action="store_true")
    main(parser.parse_args())
