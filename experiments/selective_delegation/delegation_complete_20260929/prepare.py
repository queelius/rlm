"""CPU-only immutable plans and generic-executor descriptors; never launches a process."""

import argparse
import json
from pathlib import Path

import complete as c


def run(study: Path, fixture: Path) -> dict:
    receipt = c.read(fixture / "FIXTURE.json")
    if not receipt["passed"] or receipt["scientific_model_calls"] != 0:
        raise ValueError("qualified CPU fixture required")
    pins = {str(fixture / "FIXTURE.json"): c.sha(fixture / "FIXTURE.json")}
    for path, digest in receipt["source_sha256"].items():
        if c.sha(path) != digest:
            raise ValueError("fixture predates current source")
    for mode in c.MODES:
        plan, _, _, _, _, _ = c.build(mode, study / mode)
        c.check_plan(plan)
        c.persist(study / mode / "PLAN.json", plan)
        pins.update(plan["source_sha256"])
        paths = [
            study / mode / "PLAN.json",
            Path(plan["prepared"]) / "MANIFEST.json",
            Path(plan["prepared"]) / "tasks.jsonl",
            Path(plan["model"]) / "local-research-manifest.json",
            Path(plan["adapter"]["path"]) / "COMMIT.json",
            Path(plan["adapter"]["path"]) / "STATE.json",
            Path(plan["adapter"]["path"]) / "adapter_config.json",
        ]
        pins.update({str(path): c.sha(path) for path in paths})
    jobs = []
    for mode in c.MODES:
        argv = [c.PYTHON, str(c.HERE / "complete.py"), "--mode", mode, "--study", str(study)]
        jobs.append(
            dict(
                name=f"complete-delegation-{mode}-two-roots",
                argv=argv,
                output=str(study / mode),
                cap_seconds=1320,
                pins=pins,
            )
        )
        jobs.append(
            dict(
                name=f"complete-delegation-{mode}-native-audit",
                argv=[
                    "/usr/bin/env",
                    "CUDA_VISIBLE_DEVICES=",
                    c.PYTHON,
                    str(c.HERE / "audit.py"),
                    "--mode",
                    mode,
                    "--study",
                    str(study),
                ],
                cap_seconds=300,
                pins=pins,
            )
        )
    jobs.append(
        dict(
            name="complete-delegation-three-arm-comparison",
            argv=[
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=",
                c.PYTHON,
                str(c.HERE / "compare.py"),
                "--study",
                str(study),
            ],
            cap_seconds=300,
            pins=pins,
        )
    )
    result = dict(
        schema="complete-delegation-generic-jobs-20260929-v1",
        jobs=jobs,
        scientific_jobs=3,
        planned_roots=6,
        task="textcraft_synth.val.494",
        world_seed=42,
        sampling_seeds=list(c.SEEDS),
        expected_GPU_minutes=[15, 30],
        conservative_GPU_minutes=[45, 60],
        hard_science_cap_minutes=60,
        total_owner_and_CPU_caps_seconds=sum(job["cap_seconds"] for job in jobs),
        fixture=str(fixture / "FIXTURE.json"),
        caveat="Same exposed root, not fresh-task transfer. Only32-response ceiling removed; "
        "two delegate refusals still produce UNKNOWN and can leave later slot unattempted. "
        "No physical stock reservation, no learned router, no new host-spawn mechanism. "
        "Checkpoint is fixed actual publiccp23, never selected on these outcomes.",
    )
    c.persist(study / "PREPARED-JOBS.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=c.STUDY)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.study, args.fixture)
    print(
        json.dumps(
            {k: result[k] for k in ("scientific_jobs", "planned_roots", "expected_GPU_minutes")}
        )
    )
