"""Native three-arm fresh-root comparison; unavailable outcomes remain unknown."""

import argparse
import json
import math
from pathlib import Path

import analyze_textcraft_fresh as teachers
import analyze_textcraft_profiles as profiles
import eval_textcraft_fresh_base as reader

audit = profiles.audit


def actual_time_limits(output):
    calls = [audit.read(path) for path in (output / "calls").glob("*.json")]
    effective = []
    potential_time_stops = 0
    for call in calls:
        limit = call.get("effective_max_time")
        if call["available"]:
            audit.require(
                isinstance(limit, (int, float)) and math.isfinite(limit) and 0 < limit <= 90,
                "actual native per-call time cap changed",
            )
        if limit is not None:
            effective.append(limit)
        if (
            call["available"]
            and call["finish_reason"] == "length_or_time"
            and len(call["output_token_ids"]) < call["request"]["cap"]
        ):
            potential_time_stops += 1
    return dict(
        recorded_calls=len(calls),
        calls_with_effective_limit=len(effective),
        minimum_effective_seconds=min(effective) if effective else None,
        deadline_reduced_limits=sum(t < 90 for t in effective),
        potential_time_limited_non_eos_calls=potential_time_stops,
        note="Nominal90seconds/request unchanged. Actual deadline-reduced limits and "
        "non-EOS short outputs retained, not silently treated as identical realized time.",
    )


def validate_base(base, teacher):
    audit.require(
        base["schema"] == "textcraft-fresh16-base-readout-v1"
        and not base.get("adapter")
        and not base.get("adapters")
        and not base.get("fixed_adapter"),
        "base must have no adapter",
    )
    for key in (
        "tasks_sha256",
        "manifest_sha256",
        "world_sha256",
        "sampling",
        "seeds",
        "seed_rule",
        "model",
        "model_manifest_sha256",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "truncation",
        "profile",
    ):
        audit.require(base[key] == teacher[key], "unmatched fresh contract: " + key)

    def normalize(jobs):
        return [{k: v for k, v in j.items() if k != "condition"} for j in jobs]

    audit.require(normalize(base["jobs"]) == normalize(teacher["jobs"]), "unmatched fresh slots")
    if base["budget_seconds"] != teacher["budget_seconds"]:
        audit.require(
            base["budget_seconds"] == 7200
            and teacher["budget_seconds"] == 3600
            and base.get("collection_wall_cap_contract") == reader.COLLECTION_CONTRACT
            and reader.c.inputs.sha(reader.TEACHER_REPORT) == reader.TEACHER_REPORT_SHA,
            "undeclared or invalid unequal collection wall caps",
        )


def analyze(base, privileged, public):
    from transformers import AutoTokenizer

    outputs = (base, privileged, public)
    plans = [audit.read(p / "PLAN.json") for p in outputs]
    teachers.validate_pair(plans[1:])
    for plan in plans[1:]:
        validate_base(plans[0], plan)
        audit.require(
            plan["adapter"] == reader.fresh.endpoint(plan["teacher"]),
            "fixed teacher checkpoint identity differs",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        reader.c.BASE, local_files_only=True, trust_remote_code=False
    )
    reports = [audit.analyze(p, tokenizer, expected_task_count=16) for p in outputs]
    realized_limits = [actual_time_limits(p) for p in outputs]
    rows = [
        {
            (r["task_id"], r["repeat"]): r
            for r in (audit.read(p) for p in (output / "episodes").glob("*.json"))
        }
        for output in outputs
    ]
    for report in reports:
        report.pop("paired", None)
        report.pop("depth_strata", None)
    return dict(
        schema="textcraft-fresh16-base-teachers-comparison-v1",
        arms=reports,
        collection_wall_caps=dict(
            seconds_by_arm=[p["budget_seconds"] for p in plans],
            prospective_declaration=plans[0].get("collection_wall_cap_contract"),
            completed_teacher_report_sha256=reader.TEACHER_REPORT_SHA,
        ),
        actual_native_time_limits=realized_limits,
        comparisons={
            name: profiles.compare(plans[0]["jobs"], rows[0], rows[index])
            for index, name in ((1, "privileged_minus_base"), (2, "public_minus_base"))
        },
        method="All32 planned slots/arm, native replay before scoring;16 root clusters with both "
        "seeds retained,20000 bootstrap draws seed2026092206 only if complete. "
        "Shared world/recipes "
        "are not independent units. Known-pair wins/losses descriptive if any unknown.",
        caveat="Checkpoint comparison on fixed fresh roots, not causal teacher-order isolation. "
        "No-adapter base,048 privilegedSFT and056 publicSFT use identical per-episode contracts. "
        "Declared collection wall caps may differ, not per-episode reasoning limits. "
        "All32 observed per arm permits matched primary accuracy; any missing/unavailable "
        "base outcome stays unknown with full-panel bounds and no complete-panel CI. "
        "Native service costs, effective time limits and actual failures remain audited.",
        source_sha256={
            str(Path(m.__file__).resolve()): reader.c.inputs.sha(Path(m.__file__))
            for m in (reader, teachers, profiles, audit)
        },
        analyzer_sha256=reader.c.inputs.sha(Path(__file__)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("base", "privileged", "public"):
        parser.add_argument("--" + arm + "-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    result = analyze(args.base_output, args.privileged_output, args.public_output)
    reader.c.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Fresh-root TextCraft: base versus teaching\n\n"
            + result["method"]
            + "\n\n"
            + result["caveat"]
            + "\n\n"
        )
        for name, comparison in result["comparisons"].items():
            stream.write(
                name
                + ": "
                + json.dumps({k: v for k, v in comparison.items() if k != "rows"})
                + "\n\n"
            )
        for arm in result["arms"]:
            stream.write(json.dumps(arm["groups"]) + "\n\n")
