"""Pending-weight preparation contract and parent-dispatched compact-RL descriptors."""

from __future__ import annotations

import argparse
import json

import compact_common as c

PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)


def describe() -> dict:
    pins = c.source_pins()
    for group in ("train", "diagnostic"):
        c.f.dataset_binding(c.f.DATA / group)
        for name in ("MANIFEST.json", "tasks.jsonl"):
            path = c.f.DATA / group / name
            pins[str(path)] = c.sha(path)
    paths = [
        c.f.DATA / "MANIFEST.json",
        c.interface.TRAIN_OUTPUT / "PLAN.json",
        c.interface.TRAIN_OUTPUT / "COMPACT-CONTRACT.json",
        c.interface.data.OUTPUT / "MANIFEST.json",
        c.BASELINE / "ENVIRONMENT-PREPARED.json",
        c.BASELINE / "binder/collect-0001/PLAN.json",
        c.BASELINE / "binder/train-0001/PLAN.json",
        c.BASELINE / "binder/readout-warm/PLAN.json",
    ]
    pins.update({str(path): c.sha(path) for path in paths})
    jobs = []
    for kind, directory, cap in (
        ("collect", "collect-0001", 9300),
        ("train", "train-0001", 7500),
        ("warm", "readout-warm", 5700),
        ("updated", "readout-0001", 5700),
    ):
        jobs.append(
            dict(
                name=f"compact-rl-{kind}",
                argv=[PYTHON, str(c.HERE / "compact_stage.py"), "--kind", kind],
                output=str(c.STUDY / directory),
                cap_seconds=cap,
                pins=pins,
            )
        )
    jobs.append(
        dict(
            name="compact-rl-first-step-comparison",
            argv=[
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=",
                PYTHON,
                str(c.HERE / "compact_compare.py"),
                "--output",
                str(c.STUDY / "COMPARISON.json"),
            ],
            cap_seconds=600,
            pins=pins,
        )
    )
    return dict(
        schema="textcraft-compact-rl-pending-preparation-20260928-v1",
        status="prepared_for_parent_dispatch_only",
        execution_mode=c.MODE,
        study=str(c.STUDY),
        question="Does one native-success RL update improve its compact observed-only actor, "
        "and how does its own-interface gain differ from first-cycle full+binder learning?",
        checkpoint_pending_until_authenticated_complete=True,
        future_checkpoint_path=str(c.WARM),
        compact_sft_plan_sha256=c.interface.TRAIN_PLAN_SHA,
        compact_sft_contract_sha256=c.interface.TRAIN_CONTRACT_SHA,
        actual_adapter_sha256=None,
        checkpoint_admission="Bind only after authentic completed compact SFT owner, committed "
        "checkpoint23, exact23-step dose/rows/contract and actual weight digest validate. "
        "Scientific collection/train PLANs and WARM-ENDPOINT.json are generated then, not now.",
        datasets={
            group: dict(
                path=str(c.f.DATA / group),
                manifest_sha256=c.f.GROUP_SHA[group],
                tasks_sha256=c.read(c.f.DATA / group / "MANIFEST.json")["tasks_sha256"],
            )
            for group in ("train", "diagnostic")
        },
        collection_seeds=list(range(202609280110, 202609280114)),
        readout_seeds=[202609280900, 202609280901],
        collection_episodes=32,
        episodes_per_readout=16,
        baseline=str(c.BASELINE / "binder"),
        baseline_update=1,
        GPU_launched=False,
        environment_receipt=str(c.BASELINE / "ENVIRONMENT-PREPARED.json"),
        environment_receipt_sha256=c.sha(c.BASELINE / "ENVIRONMENT-PREPARED.json"),
        jobs=jobs,
        expected_useful_hours="2–4h; approximately50–100min collection,15–40min training, "
        "25–50min per diagnostic readout. Compact cost is not yet measured.",
        scientific_stage_caps_hours=7.5,
        upper_job_caps_hours=sum(job["cap_seconds"] for job in jobs) / 3600,
        coordinator_lock=str(
            c.c.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock"
        ),
        continuation="No sweep or continuation: one on-policy batch and freshAdamW2e-5 update. "
        "Incomplete batch, flat native rewards or fatal numerical seam skips the learned "
        "endpoint. Warm groupB remains independent. DiagnosticB never enters optimization.",
        dispatch="Prepared job list, not ACCEPTED. Parent schedules after existing GPU owners; "
        "execute all fixed stages even after conditional skips, respecting the shared lock.",
        caveat="Matched SFT updates are not equal target-token exposure; compact unseen-recipe "
        "rejection differs from full binder. Compare own-interface learning gains, never "
        "interpret raw score difference as learning benefit or isolated schema causality.",
    )


def qualify() -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        c.c.BASE, local_files_only=True, trust_remote_code=False
    )
    world = c.c.bridge.load_world()
    world_sha = c.interface.data.worlds.digest(c.interface.data.worlds.snapshot(world))
    groups = {}
    for group, phase in (("train", "collect"), ("diagnostic", "readout")):
        manifest, rows = c.f.load_dataset(c.f.DATA / group, phase)
        if manifest["world_sha256"] != world_sha:
            raise ValueError("compact native world differs from qualified fresh world42")
        lengths = [
            len(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": c.c.bridge.initial_prompt(task, "flat")}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            for task in rows
        ]
        if max(lengths) + 256 > 8192:
            raise ValueError("initial compact context cap exceeded")
        groups[group] = dict(prompt_tokens=lengths, max_prompt_plus_cap=max(lengths) + 256)
    return dict(
        world_seed=42,
        world_sha256=world_sha,
        initial_context=groups,
        GPU_used=False,
        scope="Initial compact prompt fit only; not a full-rollout context guarantee.",
    )


def prepare() -> dict:
    contract = describe()
    contract["cpu_qualification"] = qualify()
    fixture = c.STUDY / "runtime-fixture-001/VERIFICATION.json"
    checked = c.read(fixture)
    if not checked["replayed"] or checked["native_score"] != 1 or checked["GPU_used"]:
        raise ValueError("saved native compact fixture must pass before dispatch")
    contract["runtime_fixture"] = str(fixture)
    contract["runtime_fixture_sha256"] = c.sha(fixture)
    for job in contract["jobs"]:
        job["pins"][str(fixture)] = c.sha(fixture)
    c.f.persist(c.STUDY / "PREPARED-JOBS.json", contract)
    return contract


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    contract = prepare()
    print(
        json.dumps(
            {
                name: contract[name]
                for name in (
                    "status",
                    "study",
                    "GPU_launched",
                    "checkpoint_pending_until_authenticated_complete",
                    "expected_useful_hours",
                    "upper_job_caps_hours",
                    "cpu_qualification",
                )
            },
            indent=2,
        )
    )
