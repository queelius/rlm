"""Score only literal, dependency-free annotated first questions in helper-eval-001."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
MUSIQUE = Path(
    "/project/alex_phd/research-cache/repos/musique-922ac98f19a201998dbdae6d7f2887a5258dbdeb"
)
CONDITIONS = ("base_helper", "trained_helper", "format_reminder")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_paths(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.name.encode())
        hasher.update(sha256(path).encode())
    return hasher.hexdigest()


def normalize_question(value: str) -> str:
    """Only collapse Unicode whitespace; do not rewrite punctuation, case, or wording."""
    return " ".join(value.split())


def parse_strict_answer(text: str | None) -> str | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"answer"}:
        return None
    return value["answer"] if isinstance(value["answer"], str) else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def pair_counts(rows: list[dict], left: str, right: str) -> dict:
    states = Counter(
        (row["conditions"][left]["exact"], row["conditions"][right]["exact"]) for row in rows
    )
    return {
        "left_wins": states[(True, False)],
        "right_wins": states[(False, True)],
        "both_exact": states[(True, True)],
        "both_not_exact": states[(False, False)],
    }


def run(root: Path) -> dict:
    sys.path.insert(0, str(MUSIQUE))
    from metrics.answer import compute_exact  # noqa: PLC0415

    inputs = root / "inputs-001/cases.jsonl"
    episodes_dir = root / "helper-eval-001/episodes"
    calls_dir = root / "helper-eval-001/calls"
    plan = root / "helper-eval-001/PLAN.json"
    exposure = root / "analysis-document-exposure-001.json"
    cases = {row["id"]: row for row in read_jsonl(inputs)}
    exposure_rows = json.loads(exposure.read_text())["panels"]["heldvalidation32"]["parents_detail"]
    exposure_by_parent = {row["parent_id"]: row["exact_title_text"] for row in exposure_rows}
    episode_paths = sorted(episodes_dir.glob("*.json"))
    groups = defaultdict(dict)
    for path in episode_paths:
        row = json.loads(path.read_text())
        groups[(row["case_id"], row["repeat"])][row["condition"]] = row

    excluded, eligible = Counter(), []
    consumed_calls = []
    for (case_id, repeat), group in sorted(groups.items()):
        if set(group) != set(CONDITIONS):
            excluded["missing_condition"] += 1
            continue
        questions = []
        for condition in CONDITIONS:
            plan_questions = group[condition].get("plan", {}).get("subquestions", [])
            if not plan_questions or not isinstance(plan_questions[0], str):
                questions = []
                break
            questions.append(normalize_question(plan_questions[0]))
        if not questions:
            excluded["missing_first_question"] += 1
            continue
        if len(set(questions)) != 1:
            excluded["conditions_differ"] += 1
            continue
        reference = [
            step
            for step in cases[case_id]["metadata"]["question_decomposition"]
            if re.search(r"#\d+", step["question"]) is None
            and normalize_question(step["question"]) == questions[0]
        ]
        if len(reference) != 1:
            excluded["not_unique_literal_dependency_free_reference"] += 1
            continue
        row = {
            "case_id": case_id,
            "repeat": repeat,
            "reference_step_id": str(reference[0]["id"]),
            "exact_train_document_exposure": exposure_by_parent[case_id],
            "conditions": {},
        }
        for condition in CONDITIONS:
            episode = group[condition]
            call_id = episode.get("call_ids", [None])[0]
            call_path = calls_dir / f"{call_id}.json" if call_id else None
            call = json.loads(call_path.read_text()) if call_path and call_path.exists() else {}
            if call_path and call_path.exists():
                consumed_calls.append(call_path)
            answer = parse_strict_answer(call.get("text")) if call.get("available") else None
            row["conditions"][condition] = {
                "planned": True,
                "call_present": bool(call_path and call_path.exists()),
                "available": bool(call.get("available")),
                "strict_json_valid": answer is not None,
                "exact": bool(compute_exact(reference[0]["answer"], answer)) if answer else False,
            }
        eligible.append(row)
    summaries = {}
    for condition in CONDITIONS:
        summaries[condition] = {
            key: sum(row["conditions"][condition][key] for row in eligible)
            for key in ("planned", "call_present", "available", "strict_json_valid", "exact")
        }
    return {
        "schema": "helper-annotation-compatible-diagnostic-v1",
        "scope": "First helper only; literal dependency-free annotated question match only.",
        "method": {
            "question_match": "collapse whitespace only, then literal equality",
            "required_pairing": "all three frozen conditions must share the same first question",
            "answer_score": "MuSiQue official compute_exact against only the annotated step answer",
            "not_done": (
                "no aliases, semantic matching, judging, later-step scoring, or outcome filtering"
            ),
        },
        "coverage": {
            "planned_root_condition_rows": 64 * len(CONDITIONS),
            "recorded_root_groups": len(groups),
            "all_three_condition_groups": sum(
                set(group) == set(CONDITIONS) for group in groups.values()
            ),
            "eligible_root_groups": len(eligible),
            "eligible_unique_parents": len({row["case_id"] for row in eligible}),
            "excluded": dict(excluded),
        },
        "planned_denominator_per_condition": len(eligible),
        "conditions": summaries,
        "paired_exact": {
            "trained_helper_minus_base_helper": pair_counts(
                eligible, "trained_helper", "base_helper"
            ),
            "format_reminder_minus_base_helper": pair_counts(
                eligible, "format_reminder", "base_helper"
            ),
        },
        "rows": eligible,
        "sources_sha256": {
            "analyze_helper_annotation_compatible.py": sha256(Path(__file__)),
            "inputs-001/cases.jsonl": sha256(inputs),
            "helper-eval-001/PLAN.json": sha256(plan),
            "analysis-document-exposure-001.json": sha256(exposure),
            "helper_eval_episode_receipts_aggregate": digest_paths(episode_paths),
            "consumed_first_helper_calls_aggregate": digest_paths(consumed_calls),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = json.dumps(run(args.root), indent=2, sort_keys=True) + "\n"
    with args.report.open("x") as handle:
        handle.write(result)
    print(result, end="")
