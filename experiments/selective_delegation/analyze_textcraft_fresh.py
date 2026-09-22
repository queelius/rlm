"""Native fresh16 teacher comparison with full unknown denominators, not independent worlds."""

import argparse
import json
from pathlib import Path

import analyze_textcraft_profiles as profiles
import eval_textcraft_fresh as reader


def validate_pair(plans):
    audit = profiles.audit
    for plan, teacher in zip(plans, ("privileged", "public"), strict=True):
        audit.require(
            plan["schema"] == "textcraft-fresh16-teacher-readout-v1"
            and plan["teacher"] == teacher
            and plan["profile"] == "original"
            and plan["world_sha256"] == reader.WORLD_SHA
            and plan["tasks_sha256"] == reader.TASKS_SHA
            and plan["manifest_sha256"] == reader.MANIFEST_SHA,
            "fixed fresh16 original-world teacher comparison required",
        )
        audit.require(
            len(plan["jobs"]) == 32
            and len({j["task_id"] for j in plan["jobs"]}) == 16
            and {j["repeat"] for j in plan["jobs"]} == {0, 1}
            and all(
                j["policy"] == "flat" and j["prompt_profile"] == "original" for j in plan["jobs"]
            ),
            "fixed32 fresh slots required",
        )
        expected = (
            reader.shared.TRAINING_PLAN_SHA256
            if teacher == "privileged"
            else reader.public.PLAN_SHA
        )
        audit.require(
            plan["adapter"]["training_plan_sha256"] == expected, "fixed teacher endpoint differs"
        )
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
        "truncation",
    ):
        audit.require(plans[0][key] == plans[1][key], "unmatched fresh contract: " + key)

    def normalize(jobs):
        return [{k: v for k, v in j.items() if k != "condition"} for j in jobs]

    audit.require(normalize(plans[0]["jobs"]) == normalize(plans[1]["jobs"]), "unmatched slots")


def analyze(left, right):
    from transformers import AutoTokenizer

    audit, c = profiles.audit, reader.c
    plans = [audit.read(p / "PLAN.json") for p in (left, right)]
    validate_pair(plans)
    for plan in plans:
        binding = reader.endpoint(plan["teacher"])
        audit.require(plan["adapter"] == binding, "saved fixed endpoint identity differs")
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    reports = [audit.analyze(p, tokenizer, expected_task_count=16) for p in (left, right)]
    rows = [
        {
            (r["task_id"], r["repeat"]): r
            for r in (audit.read(p) for p in (output / "episodes").glob("*.json"))
        }
        for output in (left, right)
    ]
    jobs = plans[0]["jobs"]
    tasks = {
        r["id"]: r
        for r in map(
            json.loads, (Path(plans[0]["prepared"]) / "tasks.jsonl").read_text().splitlines()
        )
    }
    strata = {
        str(depth): profiles.compare(
            [j for j in jobs if tasks[j["task_id"]]["misc"]["max_depth"] == depth], *rows
        )
        for depth in (2, 3, 4)
    }
    for report in reports:
        report.pop("paired", None)
        report.pop("depth_strata", None)
    return dict(
        schema="textcraft-fresh16-teacher-comparison-v1",
        public_minus_privileged=profiles.compare(jobs, *rows),
        depth_strata=strata,
        arms=reports,
        method=dict(
            parent_count=16,
            repeats=2,
            planned_per_arm=32,
            bootstrap_draws=20000,
            bootstrap_seed=2026092206,
            bootstrap_unit="Root task, both seeds retained; shared world/prerequisites "
            "are not independent units",
            unknown="No imputation; full-panel bounds; "
            "complete-panel effect/CI unavailable if any paired outcome unknown",
        ),
        caveat="Outcome-blind fresh16roots in sameworld42; currentR exclusion inventory only. "
        "14roots share TRAIN32prerequisites. Original prompt fixed, old048 versus new056 "
        "checkpoint23 only. ParentCI exploratory; not teacher-order causal proof. "
        "Eacharm60min; missing/transport outcomes remain unknown, not losses.",
        source_sha256={
            str(Path(m.__file__).resolve()): c.inputs.sha(Path(m.__file__))
            for m in (reader, profiles, audit)
        },
        analyzer_sha256=c.inputs.sha(Path(__file__)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privileged-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    result = analyze(args.privileged_output, args.public_output)
    reader.c.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write("# Fresh-root TextCraft teacher comparison\n\n" + result["caveat"] + "\n\n")
        stream.write(
            json.dumps({k: v for k, v in result["public_minus_privileged"].items() if k != "rows"})
            + "\n\n"
        )
        for arm in result["arms"]:
            stream.write(json.dumps(arm["groups"]) + "\n\n")
