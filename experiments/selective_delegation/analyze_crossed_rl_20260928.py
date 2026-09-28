"""CPU-only crossed training/tool readout, including skipped and failed endpoints."""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

from run_followon_queue_20260928 import predecessor_state

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rl_resume_20260928"))
spec = importlib.util.spec_from_file_location(
    "crossed_prior_compare", HERE / "rl_resume_20260928/compare.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
p = old.p
STUDY = p.ROOT / "textcraft-rl-assist-20260928-001"
QUEUE = p.ROOT / "rl-update-queue-20260928-001"


def contrast(new, reference, keys):
    return old.paired(
        {
            key: new[key] - reference[key]
            if new.get(key) is not None and reference.get(key) is not None
            else None
            for key in keys
        }
    )


def read_if(path):
    return p.read(path) if path.exists() else None


def check_pairing(plans):
    for plan in plans[1:]:
        for field in (
            "tasks_sha256",
            "runtime_tasks_sha256",
            "selection_plan_sha256",
            "model_manifest_sha256",
            "base_dtype",
            "sampling",
            "max_global_calls",
            "max_global_output_tokens",
            "max_new_tokens",
            "input_plus_output_limit",
        ):
            if plan[field] != plans[0][field]:
                raise ValueError("crossed readout scientific contract differs: " + field)


def analyze():
    rows = p.tasks()
    keys = [(row["id"], repeat) for row in rows for repeat in (0, 1)]
    cells, values, plans, training = {}, {}, {}, {}
    for learned in ("raw", "binder"):
        training[learned] = read_if(STUDY / learned / "train-0001/SUMMARY.json")
    for learned in ("warm", "raw", "binder"):
        for executed in ("raw", "binder"):
            name = learned + "_" + executed
            if learned == "warm":
                directory, expected = STUDY / executed / "readout-warm", str(p.WARM)
            else:
                suffix = "" if learned == executed else "-as-" + executed
                directory = STUDY / learned / ("readout-0001" + suffix)
                trained = training[learned] or {}
                expected = trained.get("endpoint") if trained.get("endpoint_usable") else None
            summary = read_if(directory / "SUMMARY.json")
            values[name] = {}
            cells[name] = dict(
                available=False,
                path=str(directory),
                skip=read_if(directory / "CONDITIONAL-SKIP.json"),
                summary=summary,
            )
            if not summary or not summary.get("complete") or summary.get("failure"):
                continue
            plan, outcomes, stats = old.load_cell(directory)
            if not expected or plan["adapter"]["path"] != expected:
                raise ValueError("crossed cell used wrong trained/unchanged actor: " + name)
            if plan["execution_mode"] != executed or plan["jobs"] != p.jobs(
                rows, executed, "readout", 1
            ):
                raise ValueError("crossed readout task/seed/interface contract changed")
            plans[name], values[name], cells[name] = plan, outcomes, dict(stats, available=True)
    check_pairing(list(plans.values()))
    gains = {
        learned + "_trained_used_" + executed: contrast(
            values[learned + "_" + executed], values["warm_" + executed], keys
        )
        for learned in ("raw", "binder")
        for executed in ("raw", "binder")
    }
    return dict(
        schema="textcraft-crossed-learning-execution-20260928-v1",
        created=time.time(),
        source_sha256=p.sha(Path(__file__)),
        predecessor=predecessor_state(QUEUE),
        cells=cells,
        training=training,
        learning_gain_vs_own_tool_baseline=gains,
        binder_trained_minus_raw_trained_same_tool={
            executed: contrast(values["binder_" + executed], values["raw_" + executed], keys)
            for executed in ("raw", "binder")
        },
        tool_effect_at_fixed_weights={
            learned: contrast(values[learned + "_binder"], values[learned + "_raw"], keys)
            for learned in ("warm", "raw", "binder")
        },
        limitation="Eight exposed TRAIN task identities with two held sampling seeds, "
        "not held-out task transfer. Missing endpoints stay unknown. Compare trained "
        "weights against unchanged weights using the SAME tools before claiming learning. "
        "Same update count does not imply same credited tokens. No recursion claim.",
    )


def markdown(report):
    def score(key):
        cell = report["cells"][key]
        return f"{cell['successes']}/{cell['observed']}" if cell["available"] else "Not available"

    lines = [
        "# Did the model improve, or did only the tools improve?",
        "",
        "Each available cell contains the same eight training goals attempted twice.",
        "These goals are familiar to the model; this is a small mechanism test.",
        "",
        "| Model weights | Ordinary tools | Ingredient assistance |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("warm", "No reward update"),
        ("raw", "Reward training with ordinary tools"),
        ("binder", "Reward training with ingredient assistance"),
    ):
        lines.append(f"| {label} | {score(key + '_raw')} | {score(key + '_binder')} |")
    lines.extend(
        [
            "",
            "Read down a column to assess learning with the tools held fixed. "
            "Read across a row to assess the tool change with the model held fixed.",
            "",
            "A missing training endpoint is not a zero-scoring model. The JSON preserves "
            "missing cells, paired wins/losses, costs and task-cluster intervals.",
            "",
            report["limitation"],
            "",
        ]
    )
    return "\n".join(lines)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    p.c.save(
        args.output / "INVOCATION.json",
        dict(
            pid=os.getpid(),
            started=time.time(),
            source_sha256=p.sha(Path(__file__)),
            GPU_actions=False,
            queue=str(QUEUE),
        ),
    )
    if args.wait:
        deadline = int(os.environ["SLURM_JOB_END_TIME"]) - 600
        while time.time() < deadline and not predecessor_state(QUEUE)["settled"]:
            time.sleep(15)
    report = analyze()
    p.c.save(args.output / "CROSSED.json", report)
    with (args.output / "FINDINGS.md").open("x") as stream:
        stream.write(markdown(report))
    print(markdown(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
