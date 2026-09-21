"""Paired, component-clustered accounting for the TRAIN direct control."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import eval_planner
import probe

SEED = 2026092191


def direct_status(call: dict, grade: dict) -> str:
    """Keep unavailable requests distinct from returned non-JSON answers."""
    if not call["available"]:
        return "unavailable"
    return "scored" if grade["valid"] else "invalid_json"


def candidate_independent_seed(slots: dict, case_id: str, setting: int) -> int:
    """Obtain the saved final seed from episode metadata, not the grade payload."""
    seeds = {slots[case_id, setting, candidate]["seed"] for candidate in range(4)}
    if len(seeds) != 1:
        raise ValueError("historical final seed is candidate-dependent")
    return next(iter(seeds))


def parent_deltas(
    parents: list[str], direct: dict, executed: dict, plan_only: dict
) -> dict[str, dict[str, float]]:
    """Collapse candidate-correlated slots before any uncertainty calculation."""
    output = {"executed_minus_direct": {}, "direct_minus_plan_only": {}}
    for parent in parents:
        keys = [(parent, setting, candidate) for setting in range(5) for candidate in range(4)]
        if any(key not in direct or key not in executed or key not in plan_only for key in keys):
            continue
        output["executed_minus_direct"][parent] = mean(executed[key] - direct[key] for key in keys)
        output["direct_minus_plan_only"][parent] = mean(
            direct[key] - plan_only[key] for key in keys
        )
    return output


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cost(calls: list[dict]) -> dict:
    return analyze_helper.measured(calls)


def _read(path: Path, hashes: dict[str, str]) -> dict:
    hashes[str(path)] = _sha(path)
    return json.loads(path.read_text())


def _historical_slots(credit_output: Path, parents: list[str], hashes: dict[str, str]) -> dict:
    """Recover the 320 actual helper and plan-only outcomes, never an oracle choice."""
    slots = {}
    for path in sorted((credit_output / "episodes").glob("*.json")):
        row = _read(path, hashes)
        if row.get("episode_id", "").endswith("-factual_replay"):
            continue
        identity = row["episode_id"]
        prefix, mode = identity.rsplit("-", 1)
        if mode != "plan_only":
            continue
        case_id, repeat, policy = prefix.rsplit("-", 2)
        key = case_id, int(repeat.removeprefix("r")), int(policy.removeprefix("c"))
        if key in slots or key[0] not in parents or not row["observed"]:
            raise ValueError("unexpected or unavailable execution-credit slot")
        slots[key] = {"seed": row["seed"], "executed": row["old"], "plan_only": row["new"]}
    if len(slots) != len(parents) * 5 * 4:
        raise ValueError("execution-credit logical inventory is incomplete")
    return slots


def analyze(
    direct_output: Path, credit_output: Path, *, draws: int = 20000, seed: int = SEED
) -> dict:
    """Analyze completed receipts; rejects partial, unpaired, or altered inputs."""
    direct_output, credit_output = Path(direct_output).resolve(), Path(credit_output).resolve()
    hashes: dict[str, str] = {}
    plan = _read(direct_output / "PLAN.json", hashes)
    credit_plan = _read(credit_output / "PLAN.json", hashes)
    if (
        plan.get("schema") != "training-direct-control-v1"
        or plan["source_execution_credit_plan_sha256"] != _sha(credit_output / "PLAN.json")
        or plan["planned_unique_direct_calls"] != 80
        or plan["planned_logical_slots"] != 320
        or plan["settings"] != 5
        or plan["candidates"] != 4
        or plan["parents"] != credit_plan["parents"]
        or draws < 1
    ):
        raise ValueError("unexpected immutable direct-control contract")
    cases_path = Path(plan["cases"])
    if _sha(cases_path) != plan["cases_sha256"]:
        raise ValueError("case source differs from direct PLAN")
    hashes[str(cases_path)] = plan["cases_sha256"]
    cases = {
        row["id"]: row for row in (json.loads(line) for line in cases_path.read_text().splitlines())
    }
    parents = plan["parents"]
    if any(parent not in cases for parent in parents):
        raise ValueError("parent absent from cases")
    history = _historical_slots(credit_output, parents, hashes)
    expected = {(p, setting) for p in parents for setting in range(5)}
    direct_rows, calls, status = {}, [], Counter()
    for path in sorted((direct_output / "episodes").glob("*.json")):
        row = _read(path, hashes)
        case_id, repeat, suffix = row["episode_id"].rsplit("-", 2)
        key = case_id, int(repeat.removeprefix("s"))
        if suffix != "direct" or key not in expected or key in direct_rows:
            raise ValueError("unexpected direct episode identity")
        if row["seed"] != candidate_independent_seed(history, *key):
            raise ValueError("direct seed differs from candidate-independent historical seed")
        if row["call_ids"] != [row["episode_id"] + "-final"]:
            raise ValueError("direct episode has unexpected physical calls")
        call_path = direct_output / "calls" / (row["call_ids"][0] + ".json")
        call = _read(call_path, hashes)
        request = call["request"]
        if (
            call["call_id"] != row["call_ids"][0]
            or request["seed"] != row["seed"]
            or request["sampling"] != plan["sampling"]
            or request["model"] != plan["model"]
            or request["role"] != "final"
            or request["condition"] != "base"
            or request["adapter_enabled"]
            or request["adapter_sha256"] is not None
            or request["prompt"] != eval_planner.direct_prompt(cases[case_id])
            or probe.runtime.digest(request) != call["request_digest"]
        ):
            raise ValueError("direct native request differs from frozen contract")
        if call["available"] and (
            call["usage"]["prompt_tokens"] != len(call["input_token_ids"])
            or call["usage"]["completion_tokens"] != len(call["output_token_ids"])
        ):
            raise ValueError("direct native token receipt differs")
        grade = probe.grade(call["text"] if call["available"] else "", cases[case_id])
        if grade != row["new"] or row["observed"] != call["available"]:
            raise ValueError("direct grade/availability differs from saved episode")
        direct_rows[key] = {"grade": grade, "call": call}
        calls.append(call)
        status[direct_status(call, grade)] += 1
    if set(direct_rows) != expected:
        raise ValueError("direct physical-call inventory is incomplete")
    if {path.stem for path in (direct_output / "calls").glob("*.json")} != {
        call["call_id"] for call in calls
    }:
        raise ValueError("unlinked direct calls")
    direct, executed, plan_only, direct_f1, executed_f1, plan_only_f1 = ({}, {}, {}, {}, {}, {})
    changes = {"executed_minus_direct": Counter(), "direct_minus_plan_only": Counter()}
    for (parent, setting, candidate), historic in history.items():
        physical = direct_rows[parent, setting]
        grade = physical["grade"]
        for name, left, right in (
            ("executed_minus_direct", historic["executed"], grade),
            ("direct_minus_plan_only", grade, historic["plan_only"]),
        ):
            delta = int(left["correct"]) - int(right["correct"])
            validity = "both_valid" if left["valid"] and right["valid"] else "protocol_involved"
            changes[name][
                ("win" if delta > 0 else "loss" if delta < 0 else "tie") + "_" + validity
            ] += 1
        key = parent, setting, candidate
        direct[key], executed[key], plan_only[key] = (
            int(grade["correct"]),
            int(historic["executed"]["correct"]),
            int(historic["plan_only"]["correct"]),
        )
        direct_f1[key], executed_f1[key], plan_only_f1[key] = (
            grade["f1"],
            historic["executed"]["f1"],
            historic["plan_only"]["f1"],
        )
    em = parent_deltas(parents, direct, executed, plan_only)
    f1 = parent_deltas(parents, direct_f1, executed_f1, plan_only_f1)
    clusters = analyze_helper.component_clusters([cases[parent] for parent in parents])
    interval = {
        name: analyze_helper.clustered_interval(values, clusters, draws, seed)
        for name, values in {**em, **{name + "_f1": values for name, values in f1.items()}}.items()
    }
    return {
        "method": {
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "component_clusters": clusters,
            "parents": len(parents),
            "settings_per_parent": 5,
            "candidate_slots_per_setting": 4,
            "estimand": (
                "parent-weighted paired difference; candidates are correlated logical slots"
            ),
        },
        "planned": {"physical_direct_calls": 80, "logical_slots": 320},
        "observed": {"physical_direct_calls": len(calls), "logical_slots": len(direct)},
        "direct": {
            "correct": sum(direct.values()),
            "f1": mean(direct_f1.values()),
            "physical_status": dict(status),
            "invalid_json_physical_calls": status["invalid_json"],
            "unavailable_physical_calls": status["unavailable"],
            "new_physical_cost": _cost(calls),
        },
        "historical": {
            "executed_correct": sum(executed.values()),
            "plan_only_correct": sum(plan_only.values()),
            "executed_f1": mean(executed_f1.values()),
            "plan_only_f1": mean(plan_only_f1.values()),
            "reused_execution_credit_cost": _read(
                Path(credit_output).parent / "analysis-training-execution-credit-001.json", hashes
            )["new_physical_cost"],
        },
        "paired_differences": interval,
        "changes": {name: dict(counts) for name, counts in changes.items()},
        "source_hashes": hashes,
        "analyzer_sha256": _sha(Path(__file__)),
        "caveat": "TRAIN-only, observational comparison. Direct has one physical response per "
        "parent/setting copied to four logical candidate slots; no result is an oracle selection, "
        "and reused historical helper/plan-only calls are not newly incurred direct-control cost.",
    }


def _markdown(report: dict) -> str:
    paired = report["paired_differences"]
    return "\n".join(
        [
            "# TRAIN direct-control findings",
            "",
            report["caveat"],
            "",
            f"Direct: {report['direct']['correct']}/320 logical slots; "
            f"physical statuses {report['direct']['physical_status']}.",
            f"Executed helper: {report['historical']['executed_correct']}/320; "
            f"plan-only: {report['historical']['plan_only_correct']}/320.",
            "",
            f"Executed minus direct: {paired['executed_minus_direct']}. "
            f"Direct minus plan-only: {paired['direct_minus_plan_only']}.",
            "",
            "The first contrast tests helper execution over direct; the second tests whether "
            "direct recovers plan-only harm. These correlated TRAIN slots do not establish "
            "a deployable policy or held-out gain.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-output", type=Path, required=True)
    parser.add_argument("--credit-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise ValueError("immutable report already exists")
    report = analyze(args.direct_output, args.credit_output)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.report.with_suffix(".md").write_text(_markdown(report))
    print(json.dumps({"report": str(args.report), "direct": report["direct"]["correct"]}))


if __name__ == "__main__":
    main()
