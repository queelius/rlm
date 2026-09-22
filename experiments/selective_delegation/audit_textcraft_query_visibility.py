"""Exact identifier visibility in saved public frames; not an action-legality audit."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)
    elif isinstance(value, str):
        yield value


def inspect(path):
    queries, tasks, rows = [], set(), 0
    for row in map(json.loads, path.read_text().splitlines()):
        rows += 1
        tasks.add(row["task_id"])
        frame = json.loads(row["prompt"].splitlines()[-1])
        if not {"target_items", "current_inventory", "history"} <= frame.keys():
            raise ValueError("unexpected public-frame schema")
        action = json.loads(row["target"])
        if action["action"] != "get_info":
            continue
        visible = set(strings(frame))
        queries.append(
            dict(
                task_id=row["task_id"],
                step=row["step"],
                items=action["items"],
                absent=[item for item in action["items"] if item not in visible],
                boundary_absent=[
                    item
                    for item in action["items"]
                    if not re.search(
                        r"(?<![A-Za-z0-9_])" + re.escape(item) + r"(?![A-Za-z0-9_])",
                        row["prompt"],
                    )
                ],
                prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
            )
        )
    return dict(
        rows=rows,
        tasks=sorted(tasks),
        query_rows=len(queries),
        query_items=sum(len(q["items"]) for q in queries),
        absent_items=sum(len(q["absent"]) for q in queries),
        absent_query_rows=sum(bool(q["absent"]) for q in queries),
        whole_prompt_boundary_absent_items=sum(len(q["boundary_absent"]) for q in queries),
        whole_prompt_boundary_absent_query_rows=sum(bool(q["boundary_absent"]) for q in queries),
        whole_prompt_boundary_absent_tasks=sorted(
            {q["task_id"] for q in queries if q["boundary_absent"]}
        ),
        tasks_with_absent=sorted({q["task_id"] for q in queries if q["absent"]}),
        queries=queries,
        rows_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Focused projection fixture: include nested history and dictionary keys, but
    # never the current row's target or future feedback. Do not substring-match.
    frame = {"target_items": {"root": 1}, "history": [{"reply": ["prior"]}]}
    assert {"root", "prior"} <= set(strings(frame))
    assert "future" not in set(strings(frame))
    assert "prior" not in set(strings({"reply": "crafted prior successfully"}))
    paths = {
        "privileged_order_047": args.root / "textcraft-train-inputs-001/rows.jsonl",
        "public_discovery_055": args.root / "textcraft-public-discovery-prototype-001/rows.jsonl",
    }
    results = {name: inspect(path) for name, path in paths.items()}
    if len({tuple(r["tasks"]) for r in results.values()}) != 1:
        raise ValueError("task sets differ")
    result = dict(
        method="Parse only each saved prompt's final JSON public frame. Recursively collect "
        "all dictionary keys and complete string values, including nested history. Compare "
        "each get_info target by exact string equality. No substring matching, current "
        "target/feedback, external world or model outcome enters visibility.",
        companion_method="Search the entire prior prompt, including free-text feedback and "
        "instructions, using (?<![A-Za-z0-9_]) + re.escape(item) + (?![A-Za-z0-9_]). "
        "This broadens visibility without reading the current target/feedback into the prompt.",
        limitation="Absence does not make a query illegal or prove label/input leakage. "
        "Arbitrary identifiers are allowed; this diagnoses a teacher-supplied identifier "
        "not explicitly present as a prior frame string/key. Not a knowledge audit.",
        inputs={name: str(path) for name, path in paths.items()},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        projection_fixture_passed=True,
        results=results,
    )
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                name: {k: v for k, v in row.items() if k != "queries"}
                for name, row in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
