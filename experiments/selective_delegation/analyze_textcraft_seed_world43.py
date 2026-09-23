"""Native second-seed world43 audit with explicit byte-identical owner-path binding."""

import argparse
import json
from pathlib import Path

import textcraft_seed_world43 as reader


def analyze(report):
    from transformers import AutoTokenizer

    c = reader.c
    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError(report)
    plans = [
        json.loads((reader.OUTPUTS[t] / "PLAN.json").read_text()) for t in reader.seed.TEACHERS
    ]
    template = json.loads(reader.TEMPLATE.read_text())

    def normalize(jobs):
        return [{k: v for k, v in j.items() if k != "condition"} for j in jobs]

    for teacher, plan in zip(reader.seed.TEACHERS, plans, strict=True):
        if (
            plan["adapter"] != reader.seed.endpoint(teacher)
            or plan["training_seed"] != reader.seed.SEED
            or normalize(plan["jobs"]) != normalize(template["jobs"])
        ):
            raise ValueError("fixed endpoint/slots changed")
        for key in (
            "world_sha256",
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
        ):
            if plan[key] != template[key]:
                raise ValueError("changed world43 contract: " + key)
    tokenizer = AutoTokenizer.from_pretrained(
        c.BASE, local_files_only=True, trust_remote_code=False
    )
    world, arms, rows = reader.panel.worlds()[1], [], []
    for teacher in reader.seed.TEACHERS:
        output = reader.OUTPUTS[teacher]
        arm = reader.profiles.audit.analyze(output, tokenizer, world=world)
        for key in ("paired", "depth_strata"):
            arm.pop(key, None)
        arms.append(arm)
        rows.append(
            {
                (r["task_id"], r["repeat"]): r
                for r in (json.loads(p.read_text()) for p in (output / "episodes").glob("*.json"))
            }
        )
    result = dict(
        schema="textcraft-world43-second-seed-analysis-v2",
        arms=arms,
        public_minus_privileged=reader.profiles.compare(plans[0]["jobs"], *rows),
        parents=8,
        repeats=2,
        planned_per_arm=16,
        caveat=plans[0]["caveat"],
        owner_path_amendment="Actual owner collector path pinned to byte-identical "
        "source062 collector; original PLAN unchanged",
        source_sha256=c.inputs.sha(Path(__file__)),
    )
    c.save(report, result)
    with report.with_suffix(".md").open("x") as stream:
        stream.write(
            "# Second-seed changed-world teacher comparison\n\n" + result["caveat"] + "\n\n"
        )
        stream.write(
            json.dumps(
                {k: v for k, v in result["public_minus_privileged"].items() if k != "rows"},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    analyze(parser.parse_args().report)
