"""CPU-only second-seed paired native audit; omit irrelevant within-arm policy contrasts."""

import argparse
import json
from pathlib import Path

import textcraft_teacher_seed as s


def single_condition(arm):
    result = dict(arm)
    result.pop("paired", None)
    result.pop("depth_strata", None)
    return result


def analyze(report):
    from transformers import AutoTokenizer

    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError("immutable report already exists")
    plans = [s.public.read(s.EVAL[t] / "PLAN.json") for t in s.TEACHERS]
    for teacher, plan in zip(s.TEACHERS, plans, strict=True):
        if plan["adapter"] != s.endpoint(teacher):
            raise ValueError("teacher endpoint changed")
    s.comparison.match_slots(plans)
    tokenizer = AutoTokenizer.from_pretrained(
        plans[0]["model"], local_files_only=True, trust_remote_code=False
    )
    audit, profiles = s.comparison.profiles.audit, s.comparison.profiles
    arms = [
        single_condition(audit.analyze(s.EVAL[t], tokenizer, expected_task_count=16))
        for t in s.TEACHERS
    ]
    rows = [
        {
            (r["task_id"], r["repeat"]): r
            for r in (s.public.read(p) for p in (s.EVAL[t] / "episodes").glob("*.json"))
        }
        for t in s.TEACHERS
    ]
    effect = profiles.compare(plans[0]["jobs"], rows[0], rows[1])
    result = dict(
        schema="textcraft-second-training-seed-comparison-v2",
        training_seed=s.SEED,
        arms=arms,
        public_minus_privileged=effect,
        parents=16,
        repeats=2,
        planned_per_arm=32,
        bootstrap_draws=20000,
        bootstrap_seed=2026092206,
        caveat=plans[0]["caveat"],
        native_source_sha256=s.c.inputs.sha(Path(s.__file__)),
        analyzer_source_sha256=s.c.inputs.sha(Path(__file__)),
        training_plan_pins_sha256=s.c.inputs.sha(s.PLANS),
        omitted_summaries="Within-arm paired/depth_strata require multiplepolicies; "
        "each teacher is a single-condition arm; native audits/costs preserved",
    )
    s.c.save(report, result)
    with report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Second training-seed teacher replication\n\n"
            + result["caveat"]
            + "\n\n```json\n"
            + json.dumps(effect, indent=2)
            + "\n```\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    analyze(parser.parse_args().report)
