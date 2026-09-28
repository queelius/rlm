"""CPU-only pinned descriptors; no scheduler, subprocess dispatch or GPU launch."""

import argparse
import json
from types import SimpleNamespace

import alf_rep as r
import alf_rep_collect as collect
import alf_rep_data as data
import alf_rep_train as train

PYTHON = (
    "/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/"
    "gpu/training/.venv/bin/python"
)


def prepare():
    from transformers import AutoTokenizer

    manifest = r.read(data.PANEL / "MANIFEST.json")
    if not manifest["ready"]:
        raise ValueError("retain failed resets without substitution; panel not runnable")
    fixture = r.STUDY / "runtime-fixture-002/VERIFICATION.json"
    if not r.read(fixture)["passed"]:
        raise ValueError("actual saved native representation fixture required")
    tokenizer = AutoTokenizer.from_pretrained(r.BASE, local_files_only=True)
    client = r.original.BaseClient(None, tokenizer, r.STUDY, 0)
    initial_context = {}
    for mode in ("index", "command"):
        lengths = []
        for game in manifest["games"]:
            prompt = r.controller(mode).prompts(game["public"]["feedback"], game["public"], [], "")[
                "flat"
            ]
            lengths.append(client.token_count(prompt))
        if max(lengths) + 128 > 8192:
            raise ValueError("fresh initial context does not fit identical limits")
        initial_context[mode] = dict(prompt_tokens=lengths, max_prompt_plus_cap=max(lengths) + 128)
    train.run(
        SimpleNamespace(
            prepared=data.COMMAND_DATA,
            output=r.COMMAND_TRAIN,
            epochs=1,
            learning_rate=1e-4,
            hours=1 / 3,
            resume=True,
            prepare_only=True,
        )
    )
    pins = r.source_pins()
    for path in (
        data.PANEL / "MANIFEST.json",
        data.COMMAND_DATA / "MANIFEST.json",
        data.COMMAND_DATA / "examples.jsonl",
        r.ORIGINAL_DATA / "examples.jsonl",
        r.ORIGINAL_DATA / "MANIFEST.json",
        r.COMMAND_TRAIN / "PLAN.json",
        r.COMMAND_TRAIN / "REPRESENTATION-CONTRACT.json",
        fixture,
        r.INDEX_CHECKPOINT / "COMMIT.json",
        r.INDEX_CHECKPOINT.parent / "PLAN.json",
    ):
        pins[str(path)] = r.sha(path)
    for game in manifest["games"]:
        pins[game["game"]] = game["game_sha256"]
    jobs = [
        dict(
            name="alf-command-sft33",
            argv=[PYTHON, str(r.HERE / "alf_rep_train.py")],
            output=str(r.COMMAND_TRAIN),
            cap_seconds=1320,
            pins=pins,
        )
    ]
    for mode, actor in (
        ("index", "base"),
        ("command", "base"),
        ("index", "trained"),
        ("command", "trained"),
    ):
        output = r.STUDY / f"{mode}-{actor}"
        if mode != "command" or actor != "trained":
            collect.plan(mode, actor, output)
            pins[str(output / "PLAN.json")] = r.sha(output / "PLAN.json")
        jobs.append(
            dict(
                name=f"alf-{mode}-{actor}",
                output=str(output),
                cap_seconds=1560,
                pins=pins,
                argv=[PYTHON, str(r.HERE / "alf_rep_collect.py"), "--mode", mode, "--actor", actor],
            )
        )
    jobs.append(
        dict(
            name="alf-native-audits-and-comparison",
            cap_seconds=1200,
            pins=pins,
            argv=[
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=",
                PYTHON,
                str(r.HERE / "alf_rep_compare.py"),
            ],
        )
    )
    result = dict(
        schema="alfworld-representation-prepared-jobs-20260928-v1",
        status="prepared_for_parent_dispatch_only",
        study=str(r.STUDY),
        GPU_launched=False,
        jobs=jobs,
        original_index_checkpoint=r.checkpoint_binding("index"),
        pending_command_checkpoint=str(r.COMMAND_CHECKPOINT),
        command_adapter_sha256=None,
        initial_context_qualification=initial_context,
        actual_command_binding="Only authenticated complete33-step checkpoint admitted at readout; "
        "missing training skips command-trained without suppressing base/index controls.",
        expected_useful_hours="1–2; original index training4.8min, old24-slot flat cells7.4–12.7 "
        "native min. Exact-command replies are longer; measured latency is a key outcome.",
        scientific_GPU_cap_minutes=116,
        supervisor_plus_CPU_cap_minutes=sum(job["cap_seconds"] for job in jobs) / 60,
        coordinator_lock=str(
            r.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock"
        ),
        question="Does removing transient index targets change within-interface native-success "
        "learning gain under matched demonstrations and optimizer updates?",
        dispatch="Parent integrates after authenticated prior GPU release. Six descriptors: "
        "one training, four independent fixed cells, then CPU native replay/comparison. "
        "This preparation receipt is not an ACCEPTED queue and launches nothing.",
    )
    r.persist(r.STUDY / "PREPARED-JOBS.json", result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = prepare()
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "study",
                    "GPU_launched",
                    "scientific_GPU_cap_minutes",
                    "supervisor_plus_CPU_cap_minutes",
                )
            },
            indent=2,
        )
    )
