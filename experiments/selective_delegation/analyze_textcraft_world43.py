"""Native within-world43 comparison of two fixed teacher adapters, not matched worlds."""

import argparse
import json
from pathlib import Path

import analyze_textcraft_profiles as profiles
import eval_textcraft_world43 as reader


def analyze(left, right):
    from transformers import AutoTokenizer

    c, audit = reader.c, profiles.audit
    plans = [audit.read(p / "PLAN.json") for p in (left, right)]
    for plan, teacher in zip(plans, ("privileged", "public"), strict=True):
        if (
            plan["schema"] != "textcraft-world43-transfer-v1"
            or plan["teacher"] != teacher
            or plan["world_sha256"] != reader.panel.WORLD_SHA
            or plan["profile"] != "original"
        ):
            raise ValueError("fixed within-world43 teacher comparison required")
        expected = (
            reader.shared.TRAINING_PLAN_SHA256
            if teacher == "privileged"
            else reader.public.PLAN_SHA
        )
        if plan["adapter"]["training_plan_sha256"] != expected:
            raise ValueError("wrong fixed teacher training endpoint")
        path = Path(plan["adapter"]["path"])
        if c.inputs.sha(path / "COMMIT.json") != plan["adapter"]["commit_sha256"]:
            raise ValueError("committed endpoint changed")
    for key in (
        "tasks_sha256",
        "manifest_sha256",
        "sampling",
        "seeds",
        "seed_rule",
        "model_manifest_sha256",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "budget_seconds",
    ):
        audit.require(plans[0][key] == plans[1][key], "unmatched world43 contract: " + key)
    def normalize(jobs):
        return [{k: v for k, v in j.items() if k != "condition"} for j in jobs]
    audit.require(normalize(plans[0]["jobs"]) == normalize(plans[1]["jobs"]), "unmatched slots")
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    world = reader.panel.worlds()[1]
    reports = [audit.analyze(p, tokenizer, world=world) for p in (left, right)]
    rows = [
        {
            (r["task_id"], r["repeat"]): r
            for r in (audit.read(p) for p in (output / "episodes").glob("*.json"))
        }
        for output in (left, right)
    ]
    contrast = profiles.compare(plans[0]["jobs"], *rows)
    return dict(
        public_minus_privileged=contrast,
        arms=reports,
        caveat="All8fixedrootgoals/two correlated seeds retained. Native within-world43 teacher "
        "comparison; reconstructed inventories, dependency graphs and task difficulty differ "
        "from42. Parent bootstrap is exploratory, not independent recipes/worlds. Missing "
        "outcomes unknown. Each arm45min; only new physical costs are charged.",
        source_sha256={str(Path(__file__)): c.inputs.sha(Path(__file__))},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privileged-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise ValueError("immutable report exists")
    result = analyze(args.privileged_output, args.public_output)
    reader.c.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Changed-recipe TextCraft teacher comparison\n\n" + result["caveat"] + "\n\n"
        )
        stream.write(
            json.dumps({k: v for k, v in result["public_minus_privileged"].items() if k != "rows"})
            + "\n\n"
        )
        for arm in result["arms"]:
            stream.write(json.dumps(arm["groups"]) + "\n\n")
