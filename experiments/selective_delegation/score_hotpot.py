"""Strict, official HotpotQA regrade of saved planner-evaluation final calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

EVALUATOR = Path("/project/alex_phd/research-cache/repos/") / (
    "hotpot-3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py"
)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tree_sha256(root: Path) -> str:
    """Content identity for the saved receipt subset, without exporting receipt text."""
    hasher = hashlib.sha256()
    for path in sorted(root.glob("*.json")):
        hasher.update(path.name.encode() + b"\0" + sha256(path).encode() + b"\n")
    return hasher.hexdigest()


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def official_score(prediction: str, answer: str) -> tuple[float, float]:
    """HotpotQA answer EM/F1, including its special yes/no/noanswer F1 rule."""
    predicted, gold = normalize_answer(prediction), normalize_answer(answer)
    em = float(predicted == gold)
    if predicted in {"yes", "no", "noanswer"} or gold in {"yes", "no", "noanswer"}:
        return em, em
    predicted_tokens, gold_tokens = predicted.split(), gold.split()
    overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
    if not overlap:
        return em, 0.0
    precision, recall = overlap / len(predicted_tokens), overlap / len(gold_tokens)
    return em, 2 * precision * recall / (precision + recall)


def parse_answer(text: str) -> str:
    """Match the saved evaluator's strict JSON answer-object contract."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate answer field")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    if (
        not isinstance(value, dict)
        or set(value) != {"answer"}
        or not isinstance(value["answer"], str)
    ):
        raise ValueError("final must be exactly one string answer field")
    return value["answer"]


def _load_cases(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows) or any(
        row["split"] != "transfer" for row in rows
    ):
        raise ValueError("cases require unique transfer parents")
    if any(row.get("dataset") != "hotpotqa" for row in rows):
        raise ValueError("Hotpot scorer accepts only HotpotQA prepared cases")
    return {row["id"]: row for row in rows}


def _final_call_id(episode: dict) -> str | None:
    call_ids = episode.get("call_ids", episode.get("new_call_ids", []))
    if not isinstance(call_ids, list) or not call_ids:
        return None
    result = call_ids[-1]
    return result if isinstance(result, str) and result.endswith("-final") else None


def _markdown(report: dict) -> str:
    lines = [
        "# HotpotQA official regrade",
        "",
        "The reported EM/F1 below is authoritative for this transfer diagnostic. Native saved "
        "MuSiQue scores are retained in source receipts only as non-comparable diagnostics.",
        "",
        "| Condition | Planned | Recorded | Valid final | Protocol | Unavailable final | "
        "Missing episode | EM | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in report["condition_order"]:
        row = report["conditions"][condition]
        lines.append(
            f"| {condition} | {row['planned']} | {row['recorded_episode']} | "
            f"{row['valid_final']} | {row['protocol_failure']} | {row['unavailable_final']} | "
            f"{row['missing_episode']} | "
            f"{row['em']:.4f} | {row['f1']:.4f} |"
        )
    if not report["comparisons"]:
        lines.extend(["", "Unpaired single-condition report; no effect estimate."])
    for paired in report["comparisons"].values():
        left, right = paired["first_condition"], paired["second_condition"]
        lines.extend(
            [
                "",
                f"Paired `{left}` versus `{right}` over all planned slots: "
                f"{left}-only correct {paired.get('first_only_correct', 0)}; {right}-only correct "
                f"{paired.get('second_only_correct', 0)}; "
                f"both correct {paired.get('both_correct', 0)}; "
                f"both incorrect {paired.get('both_incorrect', 0)}.",
            ]
        )
    lines.extend(
        [
            "",
            "Scoring exactly follows cached HotpotQA answer normalization: lowercase, remove ASCII "
            "punctuation and articles, collapse whitespace. If either normalized answer is `yes`, "
            "`no`, or `noanswer`, F1 is exact-match-only; otherwise it is token-overlap F1. "
            "Malformed, unavailable, and missing finals contribute zero under the planned "
            "denominator. Incomplete runs give lower bounds, not completed-run estimates. "
            "Protocol and unavailable counts are separate; see source statuses in REPORT.json.",
        ]
    )
    return "\n".join(lines) + "\n"


def score(cases_path: Path, evaluation: Path, output: Path, *, comparison_output=None) -> dict:
    """Regrade one to three conditions, optionally joining disjoint matching saved runs."""
    cases_path, evaluation, output = map(Path, (cases_path, evaluation, output))
    if output.exists():
        raise FileExistsError(output)
    cases = _load_cases(cases_path)
    plan_path = evaluation / "PLAN.json"
    plan = json.loads(plan_path.read_text())
    sources = [(evaluation, plan)]
    if comparison_output is not None:
        comparison_output = Path(comparison_output)
        other = json.loads((comparison_output / "PLAN.json").read_text())
        if (
            not plan.get("conditions")
            or not other.get("conditions")
            or set(plan["conditions"]) & set(other["conditions"])
        ):
            raise ValueError("comparison requires disjoint nonempty condition sets")
        for field in ("cases_sha256", "case_ids", "repeats", "seed", "execution"):
            if field not in plan or field not in other or plan[field] != other[field]:
                raise ValueError("comparison contract differs: " + field)
        for field in (
            "source_sha256",
            "model",
            "model_manifest_sha256",
            "split",
            "mode",
            "temperature",
            "top_p",
            "top_k",
            "caps",
            "max_context",
            "helper_budget_policy",
            "architecture",
        ):
            if plan.get(field) != other.get(field):
                raise ValueError("comparison contract differs: " + field)
        sources.append((comparison_output, other))
    conditions = [c for _, p in sources for c in p.get("conditions", [])]
    repeats = plan.get("repeats")
    if not 1 <= len(conditions) <= 3 or len(set(conditions)) != len(conditions):
        raise ValueError("evaluation plan must declare one to three distinct conditions")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("evaluation plan requires positive repeats")
    if plan.get("cases_sha256") and plan["cases_sha256"] != sha256(cases_path):
        raise ValueError("cases differ from immutable evaluation plan")
    parents = plan.get("case_ids", list(cases))
    if (
        not parents
        or len(set(parents)) != len(parents)
        or any(parent not in cases for parent in parents)
    ):
        raise ValueError("invalid planned parent inventory")
    episodes, calls, provenance = [], {}, []
    for source, source_plan in sources:
        binding = {}
        for path, expected_hash in source_plan.get("dependencies", {}).items():
            if sha256(Path(path)) != expected_hash:
                raise ValueError("source dependency hash changed: " + path)
            binding[path] = expected_hash
        if source_plan.get("adapter"):
            adapter = Path(source_plan["adapter"])
            for name, expected_hash in source_plan.get("adapter_files_sha256", {}).items():
                if sha256(adapter / name) != expected_hash:
                    raise ValueError("adapter hash changed: " + name)
                binding[str(adapter / name)] = expected_hash
            if source_plan.get("training_plan_sha256"):
                training_plan = adapter.parent / "PLAN.json"
                if sha256(training_plan) != source_plan["training_plan_sha256"]:
                    raise ValueError("training plan hash changed")
                binding[str(training_plan)] = source_plan["training_plan_sha256"]
        if source_plan.get("model_manifest_sha256"):
            manifest = Path(source_plan["model"]) / "local-research-manifest.json"
            if sha256(manifest) != source_plan["model_manifest_sha256"]:
                raise ValueError("base model manifest hash changed")
            binding[str(manifest)] = source_plan["model_manifest_sha256"]
        for path in (source / "episodes").glob("*.json"):
            row = json.loads(path.read_text())
            if row.get("condition") not in source_plan["conditions"]:
                raise ValueError("episode condition outside source plan")
            episodes.append(row)
        for path in (source / "calls").glob("*.json"):
            row = json.loads(path.read_text())
            cid = row["call_id"]
            if (
                cid in calls
                or path.stem != cid
                or row.get("condition", source_plan["conditions"][0])
                not in source_plan["conditions"]
            ):
                raise ValueError("duplicate call or source condition mismatch")
            if "request" in row:
                digest = hashlib.sha256(
                    json.dumps(row["request"], sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                if digest != row.get("request_digest"):
                    raise ValueError("native request digest changed")
            calls[cid] = row
        provenance.append(
            {
                "evaluation": str(source.resolve()),
                "plan_sha256": sha256(source / "PLAN.json"),
                "episodes_sha256": tree_sha256(source / "episodes"),
                "calls_sha256": tree_sha256(source / "calls"),
                "verified_bindings": binding,
            }
        )
    expected = {
        (case_id, repeat, condition)
        for case_id in parents
        for repeat in range(repeats)
        for condition in conditions
    }
    indexed = {}
    for episode in episodes:
        key = (episode.get("case_id"), episode.get("repeat"), episode.get("condition"))
        if key not in expected or key in indexed:
            raise ValueError("episode is outside the planned Hotpot panel or duplicated")
        indexed[key] = episode
    outcome, statuses = {}, {}
    summaries = {}
    for condition in conditions:
        counts = Counter()
        em_total = f1_total = 0.0
        for case_id in parents:
            for repeat in range(repeats):
                key = (case_id, repeat, condition)
                episode = indexed.get(key)
                if episode is None:
                    counts["missing_episode"] += 1
                    statuses[key] = "missing_episode"
                    outcome[key] = (0.0, 0.0)
                    continue
                counts["recorded_episode"] += 1
                final_id = _final_call_id(episode)
                if final_id and final_id != episode["episode_id"] + "-final":
                    raise ValueError("final receipt identity does not belong to episode")
                final = calls.get(final_id) if final_id else None
                if final and (
                    final.get("condition", condition) != condition
                    or final.get("role", "final") != "final"
                ):
                    raise ValueError("episode final condition mismatch")
                if not final or not final.get("available", False):
                    status = episode.get("status", "missing_final")
                    if status in ("invalid_plan", "invalid_dependency", "invalid_helper"):
                        counts["protocol_failure"] += 1
                    else:
                        counts["unavailable_final"] += 1
                        status = "unavailable_final" if final else "missing_final"
                    counts[
                        "no_final_called"
                        if final_id is None
                        else "missing_final_receipt"
                        if final is None
                        else "unavailable_final_receipt"
                    ] += 1
                    statuses[key] = status
                    outcome[key] = (0.0, 0.0)
                    continue
                try:
                    answer = parse_answer(final.get("text"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    counts["invalid_final"] += 1
                    counts["protocol_failure"] += 1
                    statuses[key] = "invalid_final"
                    outcome[key] = (0.0, 0.0)
                    continue
                em, f1 = official_score(answer, cases[case_id]["answer"])
                counts["valid_final"] += 1
                statuses[key] = "scored"
                em_total += em
                f1_total += f1
                outcome[key] = (em, f1)
        planned = len(parents) * repeats
        summaries[condition] = {
            "planned": planned,
            "recorded_episode": counts["recorded_episode"],
            "missing_episode": counts["missing_episode"],
            "unavailable_final": counts["unavailable_final"],
            "invalid_final": counts["invalid_final"],
            "valid_final": counts["valid_final"],
            "protocol_failure": counts["protocol_failure"],
            "no_final_called": counts["no_final_called"],
            "missing_final_receipt": counts["missing_final_receipt"],
            "unavailable_final_receipt": counts["unavailable_final_receipt"],
            "status_counts": dict(
                Counter(statuses[p, r, condition] for p in parents for r in range(repeats))
            ),
            "source_status_counts": dict(
                Counter(
                    indexed.get((p, r, condition), {}).get(
                        "status",
                        "unspecified" if (p, r, condition) in indexed else "missing_episode",
                    )
                    for p in parents
                    for r in range(repeats)
                )
            ),
            "em": em_total / planned,
            "f1": f1_total / planned,
        }
    comparisons = {}
    for left, right in combinations(conditions, 2):
        counts = Counter()
        f1_difference = 0.0
        for case_id in parents:
            for repeat in range(repeats):
                left_key, right_key = (case_id, repeat, left), (case_id, repeat, right)
                left_em, left_f1 = outcome[left_key]
                right_em, right_f1 = outcome[right_key]
                category = (
                    "both_correct"
                    if left_em and right_em
                    else "first_only_correct"
                    if left_em
                    else "second_only_correct"
                    if right_em
                    else "both_incorrect"
                )
                counts[category] += 1
                if category in ("first_only_correct", "second_only_correct"):
                    pair_status = (statuses[left_key], statuses[right_key])
                    failure_kind = (
                        "missing_involved"
                        if any(
                            s.startswith("missing") or s == "unavailable_final" for s in pair_status
                        )
                        else "protocol_involved"
                        if any(s.startswith("invalid") for s in pair_status)
                        else "both_scored"
                    )
                    counts[category + "_" + failure_kind] += 1
                f1_difference += left_f1 - right_f1
        paired = {
            "first_condition": left,
            "second_condition": right,
            **dict(counts),
            "mean_f1_difference_first_minus_second": f1_difference / (len(parents) * repeats),
        }
        if [left, right] == ["base", "sft"]:
            paired.update(
                base_only_correct=counts["first_only_correct"],
                sft_only_correct=counts["second_only_correct"],
            )
        comparisons[right + "_minus_" + left] = paired
    report = {
        "schema": "selective-delegation-hotpot-official-regrade-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition_order": conditions,
        "conditions": summaries,
        "paired_comparison": next(iter(comparisons.values())) if len(conditions) == 2 else None,
        "comparisons": comparisons,
        "planned_case_ids": parents,
        "denominator_note": "Every planned transfer parent and planned repeat contributes; "
        "missing, unavailable, and malformed finals score zero and remain explicit.",
        "scoring": {
            "authoritative": "cached official HotpotQA answer EM/F1 implementation",
            "evaluator_path": str(EVALUATOR),
            "evaluator_sha256": sha256(EVALUATOR),
            "yes_no_noanswer_f1": "F1 is zero unless normalized prediction exactly equals "
            "normalized gold.",
            "native_muSiQue_scores": "retained only in saved evaluation receipts; not comparable "
            "or reported as Hotpot performance",
        },
        "sources": {
            "cases_path": str(cases_path.resolve()),
            "cases_sha256": sha256(cases_path),
            "evaluation_plan_path": str(plan_path.resolve()),
            "evaluation_plan_sha256": sha256(plan_path),
            "episodes_sha256": tree_sha256(evaluation / "episodes"),
            "calls_sha256": tree_sha256(evaluation / "calls"),
            "evaluations": provenance,
        },
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("score_hotpot.py", "test_score_hotpot.py")
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "REPORT.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (output / "REPORT.md").open("x") as stream:
        stream.write(_markdown(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path)
    arguments = parser.parse_args()
    result = score(
        arguments.cases,
        arguments.evaluation,
        arguments.output,
        comparison_output=arguments.comparison_output,
    )
    print(json.dumps({"conditions": result["conditions"], "paired": result["paired_comparison"]}))
