"""CPU admission and two scientific job descriptors; never starts the GPU."""

import json
import sys
from types import SimpleNamespace

import payload as m


def prepare():
    fixture = m.STUDY / "runtime-fixture-003/VERIFICATION.json"
    result = m.p.read(fixture)
    if result.get("passed") is not True or result["GPU_used"]:
        raise ValueError("successful scripted CPU native/loss fixture required")
    for name, digest in result["source_sha256"].items():
        if m.p.sha(m.Path(name)) != digest:
            raise ValueError("fixture source changed")
    masks = m.prepare_masks()
    if not masks["qualified"]:
        m.persist(
            m.STUDY / "NO-RUN.json",
            dict(reason="Prospective payload signal gate failed", masks=masks),
        )
        return
    m.prepare_training(SimpleNamespace(output=m.TRAIN, collection=m.COLLECTION, hours=1))
    pins = m.source_pins()
    artifacts = [
        fixture,
        m.STUDY / "MASKS.json",
        m.STUDY / "PRIVATE-ADAPTER.json",
        m.TRAIN / "PLAN.json",
        m.COLLECTION / "PLAN.json",
        m.COLLECTION / "NATIVE-AUDIT.json",
        m.ORIGINAL / "train-0001/PLAN.json",
        m.ORIGINAL / "readout-warm/PLAN.json",
    ]
    artifact_pins = {str(path): m.p.sha(path) for path in artifacts}
    py = sys.executable
    report = dict(
        schema="payload-mask-prepared-jobs-20260928-v1",
        parent_alone_launches_GPU=True,
        source_sha256=pins,
        artifact_sha256=artifact_pins,
        warm_adapter=m.p.read(m.COLLECTION / "PLAN.json")["adapter"],
        future_endpoints="Full and masked committed adapter hashes authenticated at runtime; "
        "no invented bindings",
        allocation_contract="Existing native COORDINATOR lock; parent assigns one GPU "
        "and SLURM_JOB_END_TIME",
        jobs=[
            dict(
                id="payload-mask-train",
                kind="training",
                output=str(m.TRAIN),
                cap_seconds=3600,
                argv=[py, str(m.HERE / "train.py"), "--hours", "1"],
                cpu_audit_argv=[
                    py,
                    str(m.HERE / "analyze.py"),
                    "--kind",
                    "train",
                    "--output",
                    str(m.TRAIN / "ENDPOINT-AUDIT.json"),
                ],
                depends_on_actual_usable_full_baseline=str(m.ORIGINAL / "train-0001"),
                plan_sha256=m.p.sha(m.TRAIN / "PLAN.json"),
                adapter_sha256=None,
                estimated_GPU_seconds=[900, 2100],
            ),
            dict(
                id="payload-mask-readout",
                kind="readout",
                output=str(m.READOUT),
                cap_seconds=2700,
                argv=[py, str(m.HERE / "readout.py"), "--hours", "0.75"],
                depends_on=["payload-mask-train"],
                requires_successful_cpu_audit_of=["payload-mask-train"],
                planned_episodes=16,
                plan_sha256=None,
                adapter_sha256=None,
                missing_endpoint="Explicit conditional skip, outcomes unknown",
                audit="Original independent native replay before complete SUMMARY",
                estimated_GPU_seconds=[900, 1800],
            ),
        ],
        cpu_comparison=dict(
            argv=[
                py,
                str(m.HERE / "analyze.py"),
                "--kind",
                "compare",
                "--output",
                str(m.STUDY / "COMPARISON.json"),
            ],
            cap_seconds=300,
            reused_controls=[str(m.ORIGINAL / "readout-warm"), str(m.ORIGINAL / "readout-0001")],
        ),
        limitations="Biased loss intervention; original generated context and all-token numerical "
        "checks remain. No inference savings, gradient-component cosine, or matched-compute claim. "
        "If positive, matched-size random-token/scale control precedes specificity claims.",
    )
    m.persist(m.STUDY / "PREPARED-JOBS.json", report)
    print(
        json.dumps(
            dict(
                qualified=True,
                jobs=2,
                masked_tokens=masks["masked_nonzero_credit_tokens"],
                descriptor=str(m.STUDY / "PREPARED-JOBS.json"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    prepare()
