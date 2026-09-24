"""Additive native audit of the interrupted positive-credit readout; no GPU use."""

import argparse
import json
from pathlib import Path

import analyze_textcraft_positive_partial as audit
import textcraft_positive_replay as experiment

c, read, sha = experiment.c, audit.read, experiment.sha


def run(report):
    from transformers import AutoTokenizer

    paths = (experiment.REFERENCE_READOUT, experiment.READOUT)
    plans = [read(p / "PLAN.json") for p in paths]
    if (
        plans[0]["adapter"] != experiment.signed.endpoint_binding()
        or plans[1]["adapter"] != experiment.endpoint_binding()
    ):
        raise ValueError("fixed committed training endpoint identity differs")
    experiment.reference.comparison.match_slots(plans)
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    arms = [audit.analyze(p, tokenizer, expected_task_count=16) for p in paths]
    for arm in arms:
        arm.pop("paired", None)
        arm.pop("depth_strata", None)
    rows = [
        {
            (r["task_id"], r["repeat"]): r
            for r in (read(p) for p in (output / "episodes").glob("*.json"))
        }
        for output in paths
    ]
    effect = experiment.reference.comparison.profiles.compare(plans[0]["jobs"], *rows)
    low = (
        sum((r["right"] if r["right"] is not None else 0) - r["left"] for r in effect["rows"]) / 32
    )
    high = (
        sum((r["right"] if r["right"] is not None else 1) - r["left"] for r in effect["rows"]) / 32
    )
    missing = [
        j
        for j in plans[1]["jobs"]
        if not rows[1].get((j["task_id"], j["repeat"]), {}).get("observed")
    ]
    result = dict(
        schema="positive-credit-partial-native-recovery-v1",
        arms=arms,
        positive_minus_signed=effect,
        difference_identification_bounds=[low, high],
        unavailable_jobs=missing,
        all32_primary=True,
        root_cause="Unknown-episode spec omitted condition/prompt_profile. "
        "Raw calls/requests agree; "
        "original checker fell back to flat instead of named condition. Additive auditor "
        "projects authoritative job metadata, retaining all original identity checks.",
        interpretation="No imputed losses or completed-panel CI. "
        "Interrupted episode remains unknown; "
        "bounds are identification bounds, not confidence intervals.",
        source_sha256={str(Path(m.__file__)): sha(Path(m.__file__)) for m in (audit, experiment)},
        wrapper_sha256=sha(Path(__file__)),
    )
    c.save(report, result)
    brief = dict(
        groups=[a["groups"] for a in arms],
        paired={k: v for k, v in effect.items() if k != "rows"},
        difference_identification_bounds=[low, high],
        missing=[j["episode_id"] for j in missing],
        costs=[a["physical_cost"] for a in arms],
    )
    with report.with_suffix(".md").open("x") as f:
        f.write(
            "# Positive-credit readout: bounded partial audit\n\n"
            + result["root_cause"]
            + "\n\n"
            + result["interpretation"]
            + "\n\n```json\n"
            + json.dumps(brief, indent=2)
            + "\n```\n"
        )
    print(json.dumps(brief))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    run(parser.parse_args().report)
