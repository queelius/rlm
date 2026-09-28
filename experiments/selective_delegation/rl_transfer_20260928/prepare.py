"""CPU-only immutable descriptors; this module never launches a process or GPU job."""

import argparse
import json
from pathlib import Path

import transfer_common as t


def prepare(study: Path) -> dict:
    evidence = t.warm_evidence()
    plans, pins = {}, dict(evidence["source_sha256"])
    pins[str(t.WARM_RECEIPT)] = t.WARM_RECEIPT_SHA
    for actor in ("raw", "binder"):
        for mode in ("raw", "binder"):
            args = t.arguments(actor, mode, study, prepare_only=True)
            plan, _ = t.prepare_collection(args, t.HERE / "collect.py")
            plans[f"{actor}_{mode}"] = str(args.output / "PLAN.json")
            pins.update(plan["source_sha256"])
            pins.update(plan["actor_lineage"]["artifact_sha256"])
            pins[str(args.output / "PLAN.json")] = t.f.sha(args.output / "PLAN.json")
    for path in (
        t.f.DATA / "MANIFEST.json",
        t.f.DATA / "diagnostic/MANIFEST.json",
        t.f.DATA / "diagnostic/tasks.jsonl",
        t.f.c.BASE / "local-research-manifest.json",
        t.f.LEGACY / "compare.py",
    ):
        pins[str(path)] = t.f.sha(path)
    jobs = [
        dict(
            name=f"familiar-rl-transfer-{actor}-actor-{mode}-execution",
            argv=[
                t.PYTHON,
                str(t.HERE / "collect.py"),
                "--study",
                str(study),
                "--actor",
                actor,
                "--mode",
                mode,
            ],
            output=str(study / f"actor-{actor}" / f"readout-{mode}"),
            cap_seconds=5700,
            pins=pins,
        )
        for actor, mode in (
            ("raw", "raw"),
            ("binder", "binder"),
            ("raw", "binder"),
            ("binder", "raw"),
        )
    ]
    jobs.append(
        dict(
            name="familiar-rl-transfer-native-paired-comparison",
            argv=[
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=",
                t.PYTHON,
                str(t.HERE / "compare.py"),
                "--study",
                str(study),
                "--output",
                str(study / "COMPARISON.json"),
            ],
            cap_seconds=600,
            pins=pins,
        )
    )
    receipt = dict(
        status="prepared_for_parent_dispatch",
        question="Does the one-step gain on familiar TRAIN goals transfer to frozen new "
        "diagnostic TRAIN roots, and is it specific to the actor's training execution interface?",
        jobs=jobs,
        plans=plans,
        warm_controls_independent=True,
        warm_controls_dependency_evidence=evidence,
        GPU_jobs=4,
        new_scientific_attempts=64,
        expected_GPU_hours="1.7–3.3 total; roughly25–50minutes per16-attempt B cell",
        scientific_cap_seconds_per_cell=5400,
        upper_job_caps_hours=sum(job["cap_seconds"] for job in jobs) / 3600,
        reused_control_outputs=[job["output"] for job in evidence["original_descriptors"]],
        inference="Eight diagnostic TRAIN clusters, not official held-out confirmation. "
        "Same-task seeds pair calls, not identical trajectories after policies diverge. "
        "No extra-SFT control here; transfer does not isolate RL versus another gradient step. "
        "Native audit runs within each collector after model cleanup. Incomplete cells remain "
        "unknown. No optimizer or model collection for optimization is added.",
    )
    t.f.persist(study / "PREPARED-JOBS.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=t.STUDY)
    result = prepare(parser.parse_args().study)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "GPU_jobs",
                    "warm_controls_independent",
                    "upper_job_caps_hours",
                )
            },
            indent=2,
        )
    )
