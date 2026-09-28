"""Finite public-bookkeeping and second-model comparisons after accepted work."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from queue_rl_updates_20260928 import PYTHON, ROOT, SUPERVISOR_PYTHON, digest
from run_followon_queue_20260928 import GENERIC, generic

HERE = Path(__file__).resolve().parent
PHI = ROOT / "textcraft-phi-transfer-20260928-001/PREPARED-JOBS.json"
PHI_SHA = "d04c18251a1fc6f964d751d8208c2dd0933d23b968add57e52fd1127bdcb1737"
TABLE_SHAS = {
    "demand": "dc01db54dacf5eee2f58331ee00dcea2cf1d2d4276e4c82493e201215bfb58c6",
    "masked": "d28fde32106b7f37ca264a7a677f33ff3cfaa5e8f3f1165757a05a399e25935c",
}


def build_jobs():
    if digest(PHI) != PHI_SHA:
        raise ValueError("reviewed Phi preparation changed")
    phi = json.loads(PHI.read_text())
    pins = {
        str(p): digest(p)
        for p in (Path(__file__), HERE / "run_followon_queue_20260928.py", GENERIC, PHI)
    }
    jobs = []
    table_dir = HERE / "inventory_bottleneck_20260928"
    for mode in ("demand", "masked"):
        output = ROOT / "textcraft-public-demand-20260928-001" / mode
        path = output / "PLAN.json"
        if digest(path) != TABLE_SHAS[mode]:
            raise ValueError("reviewed table PLAN changed")
        plan = json.loads(path.read_text())
        job_pins = {**pins, **plan["source_sha256"], str(path): digest(path)}
        argv = [PYTHON, str(table_dir / "readout.py"), "--mode", mode]
        jobs.append(
            dict(
                name=f"public-quantity-{mode}",
                argv=argv,
                output=str(output),
                cap_seconds=3000,
                pins=job_pins,
            )
        )
        jobs.append(
            dict(
                name=f"public-quantity-{mode}-audit",
                argv=["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *argv, "--audit"],
                cap_seconds=300,
                pins=job_pins,
            )
        )
    compare = table_dir / "compare.py"
    jobs.append(
        dict(
            name="public-quantity-paired",
            cap_seconds=300,
            argv=["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", PYTHON, str(compare)],
            pins={**pins, str(compare): digest(compare)},
        )
    )
    phi_pins = {**pins, **phi["source_sha256"], **phi["artifact_sha256"]}
    by_id = {job["id"]: job for job in phi["jobs"]}
    for name in phi["suggested_queue_order"]:
        prepared = by_id[name]
        argv = prepared["argv"]
        if prepared["kind"] == "trained_readout":
            dependency = by_id[prepared["depends_on"][0]]
            audit = str(Path(dependency["output"]) / "ENDPOINT-AUDIT.json")
            # The old generic executor has no dependency API. This immutable
            # command checks the successful extra dose/projection audit before
            # exec; evaluate.py independently reauthenticates the real endpoint.
            code = (
                "import json,os; from pathlib import Path; "
                f"r=json.loads(Path({audit!r}).read_text()); "
                "assert r.get('passed') is True; "
                f"assert r['binding']['training_plan_sha256']=={dependency['plan_sha256']!r}; "
                f"os.execv({argv[0]!r},{argv!r})"
            )
            argv = [PYTHON, "-c", code]
        jobs.append(
            dict(
                name=name,
                argv=argv,
                output=prepared["output"],
                cap_seconds=prepared["cap_seconds"] + 300,
                pins=phi_pins,
            )
        )
        jobs.append(
            dict(
                name=name + "-audit",
                argv=["/usr/bin/env", "CUDA_VISIBLE_DEVICES=", *prepared["cpu_audit_argv"]],
                cap_seconds=300,
                pins=phi_pins,
            )
        )
    return jobs


def accept(name, predecessor, jobs, question, scientific_limits, launch=False):
    unique = {}
    for job in jobs:
        unique.update(job["pins"])
        if job.get("output") and list(Path(job["output"]).glob("OWNER-*.json")):
            raise ValueError("existing scientific attempt: " + job["output"])
    generic.validate_pins(unique)
    receipt = dict(
        status="accepted",
        authorized_by="September28 autonomous informative research",
        predecessor_queue=str(ROOT / predecessor),
        executor_sha256=digest(GENERIC),
        maximum_seconds=72 * 3600,
        allocation_end=os.environ["SLURM_JOB_END_TIME"],
        jobs=jobs,
        question=question,
        scientific_limits=scientific_limits,
        upper_job_caps_hours=sum(j["cap_seconds"] for j in jobs) / 3600,
    )
    if not launch:
        print(
            json.dumps(
                dict(
                    jobs=len(jobs),
                    unique_pins=len(unique),
                    cap_hours=receipt["upper_job_caps_hours"],
                )
            )
        )
        return
    output = ROOT / name
    output.mkdir(exist_ok=False)
    with (output / "ACCEPTED.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    argv = [
        SUPERVISOR_PYTHON,
        str(HERE / "run_followon_queue_20260928.py"),
        "--receipt",
        str(output / "ACCEPTED.json"),
        "--output",
        str(output / "queue"),
    ]
    with (output / "supervisor.log").open("x") as stream:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with (output / "LAUNCH.json").open("x") as stream:
        json.dump(dict(pid=child.pid, started=time.time(), argv=argv), stream, indent=2)
    print(
        json.dumps(
            dict(
                pid=child.pid,
                receipt=str(output / "ACCEPTED.json"),
                receipt_sha256=digest(output / "ACCEPTED.json"),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    accept(
        "transfer-queue-20260928-001",
        "compact-delegation-queue-20260928-001",
        build_jobs(),
        "Does public quantity bookkeeping improve deeper tasks? Separately, does "
        "the paired teaching/recipe-assistance result extend to Phi4mini?",
        "Table: four exposed goals, eight attempts/arm, host arithmetic not memory. "
        "Phi: eight exposed roots, one seed, two worlds, fixed23-update teacher endpoints. "
        "Four base episodes only descriptive. Unequal model/token/LoRA compute across "
        "families. Dedicated audits use actualPLAN counts, not legacy16-slot summaries. "
        "Future real endpoints required; missing or failed dependencies cannot load a model. "
        "Expected useful work approximately3–5h; caps are ceilings, not duration predictions.",
        parser.parse_args().launch,
    )
