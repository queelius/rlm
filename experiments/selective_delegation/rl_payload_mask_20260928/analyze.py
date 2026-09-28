"""CPU endpoint audit and paired masked/full/warm comparison; missing stays unknown."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import payload as m


def audit_train() -> dict:
    binding = m.endpoint(m.TRAIN)
    if binding is None:
        return dict(passed=False, status="no_usable_masked_endpoint", outcome_unknown=True)
    masks = m.masks_record()
    _, credits, _, _ = m.private(m.LEGACY / "train.py").load_batch(
        m.COLLECTION, {"collection_plan_sha256": masks["collection_plan_sha256"]}
    )
    nonzero = [c for c in credits if c.advantage]
    bits = {cid: row["mask"] for cid, row in masks["records"].items()}
    before = m.p.read(m.TRAIN / "BEFORE_LOGPS.json")["logps"]
    after = m.p.read(m.TRAIN / "AFTER_LOGPS.json")["logps"]
    update = m.p.read(m.TRAIN / "UPDATE.json")
    if set(before) != set(bits) or set(after) != {c.call_id for c in nonzero}:
        raise ValueError("original before/all and after/nonzero token coverage differs")
    for values in (before, after):
        for cid, row in values.items():
            if len(row) != len(bits[cid]) or not all(math.isfinite(v) for v in row):
                raise ValueError("original replay tokens omitted or nonfinite")
    a, b = (m.masked_batch_objective(nonzero, value, bits) for value in (before, after))
    if not math.isclose(update["objective_before_eval"], a, abs_tol=1e-8):
        raise ValueError("before objective was not the declared masked objective")
    if not math.isclose(update["objective_after"], b, abs_tol=1e-8):
        raise ValueError("after objective was not the declared masked objective")
    counts = m.loss_counts(nonzero, bits)
    if any(update[k] != v for k, v in counts.items()):
        raise ValueError("retained/masked loss count differs")
    if update["original_replay_tokens"] != masks["original_nonzero_credit_tokens"]:
        raise ValueError("full original numerical-check count was changed")
    if not m.p.read(m.TRAIN / "FIRST-REPLAY.json")["passed"]:
        raise ValueError("missing successful early real replay")
    movement = {}
    for name, bit in (("retained", 1), ("removed_payload", 0)):
        deltas = [
            new - old
            for c in nonzero
            for old, new, flag in zip(
                before[c.call_id], after[c.call_id], bits[c.call_id], strict=True
            )
            if flag == bit
        ]
        movement[name] = dict(tokens=len(deltas), mean_sampled_token_logp_delta=mean(deltas))
    return dict(
        passed=True,
        binding=binding,
        masked_objective_before_eval=a,
        masked_objective_after=b,
        training_mode_objective_before=update["objective_before"],
        loss_counts=counts,
        original_replay_tokens=update["original_replay_tokens"],
        sampled_probability_movement=movement,
        caveat="Same-prefix token movement only; no KL, gradient-conflict or task-benefit claim",
        input_sha256={
            str(path): m.p.sha(path)
            for path in (
                m.TRAIN / "UPDATE.json",
                m.TRAIN / "BEFORE_LOGPS.json",
                m.TRAIN / "AFTER_LOGPS.json",
                m.STUDY / "MASKS.json",
            )
        },
    )


def compare() -> dict:
    directories = {
        "warm": m.ORIGINAL / "readout-warm",
        "full": m.ORIGINAL / "readout-0001",
        "masked": m.READOUT,
    }
    missing = [
        name
        for name, d in directories.items()
        if not (d / "SUMMARY.json").exists() or not m.p.read(d / "SUMMARY.json").get("complete")
    ]
    if missing:
        return dict(
            status="incomplete",
            missing_cells=missing,
            outcome_unknown=True,
            caveat="Missing baseline or masked readouts are not failures or zero rewards",
        )
    original = m.private(m.LEGACY / "compare.py")
    cells = {name: original.load_cell(d) for name, d in directories.items()}
    reference = cells["warm"][0]
    identities = [(j["task_id"], j["repeat"], j["seed"]) for j in reference["jobs"]]
    for plan, _, _ in cells.values():
        if [(j["task_id"], j["repeat"], j["seed"]) for j in plan["jobs"]] != identities:
            raise ValueError("task/repeat/seed pairing differs")
        for key in (
            "tasks_sha256",
            "sampling",
            "base_dtype",
            "max_global_calls",
            "max_global_output_tokens",
            "max_new_tokens",
            "input_plus_output_limit",
        ):
            if plan[key] != reference[key]:
                raise ValueError("scientific readout contract differs: " + key)
    for name, directory in (("masked", m.TRAIN), ("full", m.ORIGINAL / "train-0001")):
        binding = m.endpoint(directory)
        if binding is None or cells[name][0]["adapter"]["sha256"] != binding["sha256"]:
            raise ValueError("readout does not match its actual fixed endpoint")
    if reference["adapter"] != m.p.read(m.COLLECTION / "PLAN.json")["adapter"]:
        raise ValueError("warm readout actor differs from collection")
    results = {}
    for right, left in (("masked", "full"), ("masked", "warm"), ("full", "warm")):
        results[f"{right}_minus_{left}"] = original.paired(
            {k: cells[right][1][k] - cells[left][1][k] for k in cells[left][1]}
        )
    return dict(
        status="complete",
        paired=results,
        cells={k: v[2] for k, v in cells.items()},
        masked_endpoint_audit=audit_train(),
        caveat="Eight exposed TRAIN clusters; same-batch biased loss intervention. If payload-off "
        "wins, matched-size random-token or gradient-scale control is needed before attributing "
        "specificity to overwritten fields. No such extra arm has been run or queued.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("train", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_train() if args.kind == "train" else compare()
    m.p.c.save(args.output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"cells", "input_sha256"}}, indent=2
        )
    )
