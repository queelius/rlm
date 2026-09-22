"""Audit saved TextCraft base044 versus public056 original readouts.

This is deliberately a narrow, receipt-bound comparison. It verifies the 16
flat/original planned slots and matched first root request; it does not claim
that the two collectors have equivalent implementations.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
BASE_REPORT = ROOT / "analysis-textcraft-pilot-001.json"
PUBLIC_REPORT = ROOT / "analysis-textcraft-public-teacher-001.json"
BASE_DIR = ROOT / "textcraft-pilot-001"
PUBLIC_DIR = ROOT / "textcraft-public-discovery-readout-001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def bound_raw(report: dict, path: Path) -> dict:
    """Return a raw-file binding; reject disagreement with an existing audit map."""
    digest = sha256(path)
    reported = report["sha256"].get(str(path))
    if reported is not None and reported != digest:
        raise ValueError(f"audit report hash disagrees for {path}")
    return {
        "path": str(path),
        "sha256": digest,
        "bound_by_existing_report": reported is not None,
    }


def success_bounds(rows: list[dict]) -> dict:
    """Compute full-planned bounds, deriving unknown count from actual slots."""
    planned = len(rows)
    known = [row for row in rows if row["status"] == "known"]
    unknown = planned - len(known)
    if any(row["public"] is None for row in rows):
        raise ValueError("public outcome unavailable")
    if any(row["base"] is None for row in known):
        raise ValueError("known row lacks base outcome")
    public_total = sum(row["public"] for row in rows)
    base_observed = sum(row["base"] for row in known)
    return {
        "planned": planned,
        "known": len(known),
        "unknown": unknown,
        "public": [public_total / planned, public_total / planned],
        "base": [base_observed / planned, (base_observed + unknown) / planned],
        "public_minus_base": [
            (public_total - base_observed - unknown) / planned,
            (public_total - base_observed) / planned,
        ],
    }


def assert_plan_contract(base_plan: dict, public_plan: dict) -> dict:
    """Verify the explicit shared planning contract for the matched readout."""
    shared_keys = (
        "prepared",
        "manifest_sha256",
        "tasks_sha256",
        "model",
        "model_manifest_sha256",
        "seeds",
        "sampling",
        "max_global_calls",
        "max_global_output_tokens",
        "max_new_tokens",
        "input_plus_output_limit",
        "truncation",
        "root_depth",
        "seed_rule",
    )
    mismatches = [key for key in shared_keys if base_plan[key] != public_plan[key]]
    if mismatches:
        raise ValueError(f"PLAN contract differs: {mismatches}")
    if base_plan["max_global_calls"] != 96:
        raise ValueError("unexpected frozen global call cap")
    if base_plan["max_global_output_tokens"] != 8192:
        raise ValueError("unexpected frozen global output-token cap")
    if base_plan["input_plus_output_limit"] != 8192:
        raise ValueError("unexpected frozen context cap")
    prepared = Path(base_plan["prepared"])
    tasks = prepared / "tasks.jsonl"
    manifest = prepared / "MANIFEST.json"
    if sha256(tasks) != base_plan["tasks_sha256"]:
        raise ValueError("PLAN task bytes do not match tasks_sha256")
    if sha256(manifest) != base_plan["manifest_sha256"]:
        raise ValueError("PLAN manifest bytes do not match manifest_sha256")
    return {key: base_plan[key] for key in shared_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    base_report, public_report = read(BASE_REPORT), read(PUBLIC_REPORT)
    base_plan_path, public_plan_path = BASE_DIR / "PLAN.json", PUBLIC_DIR / "PLAN.json"
    base_plan, public_plan = read(base_plan_path), read(public_plan_path)
    for report, plan_path in ((base_report, base_plan_path), (public_report, public_plan_path)):
        if report["sha256"].get(str(plan_path)) != sha256(plan_path):
            raise ValueError("report does not bind PLAN")
    plan_contract = assert_plan_contract(base_plan, public_plan)
    base_jobs = {job["episode_id"]: job for job in base_plan["jobs"] if job["policy"] == "flat"}
    public_jobs = {
        job["episode_id"]: job
        for job in public_plan["jobs"]
        if job["condition"] == "trained_original"
    }
    expected_base_ids = {public_id.removesuffix("-original") for public_id in public_jobs}
    if len(base_jobs) != 16 or len(public_jobs) != 16 or set(base_jobs) != expected_base_ids:
        raise ValueError("expected exactly 16 matched flat/original planned slots")
    rows, contracts, raw_bindings = [], [], []
    for public_id, public_job in sorted(public_jobs.items()):
        base_id = public_id.removesuffix("-original")
        base_job = base_jobs[base_id]
        identity = ("task_id", "repeat", "seed")
        if any(base_job[key] != public_job[key] for key in identity):
            raise ValueError("planned task identity differs")
        base_episode_path = BASE_DIR / "episodes" / f"{base_id}.json"
        public_row = public_report["episode_rows"][public_id]
        row = {
            "episode_id": public_id,
            "task_id": public_job["task_id"],
            "repeat": public_job["repeat"],
            "seed": public_job["seed"],
            "public": public_row["native_score"],
        }
        if not base_episode_path.exists():
            row["base"] = None
            row["status"] = "base_missing_unknown"
            rows.append(row)
            continue
        base_episode = read(base_episode_path)
        raw_bindings.append({"base_episode": bound_raw(base_report, base_episode_path)})
        row["base"] = base_episode["native_score"] if base_episode["observed"] else None
        row["status"] = "known" if base_episode["observed"] else "base_unavailable_unknown"
        if base_episode["observed"]:
            base_call_path = BASE_DIR / "calls" / f"{base_id}-c000.json"
            public_call_path = PUBLIC_DIR / "calls" / f"{public_id}-c000.json"
            base_call = read(base_call_path)["request"]
            public_call = read(public_call_path)["request"]
            fields = (
                "task_id",
                "seed",
                "policy",
                "role",
                "depth",
                "cap",
                "model",
                "model_manifest_sha256",
                "context_limit",
                "truncation",
                "sampling",
            )
            if any(base_call[field] != public_call[field] for field in fields):
                raise ValueError("first-call native contract differs")
            if base_call["prompt"] != public_call["prompt"] or (
                base_call["input_token_ids"] != public_call["input_token_ids"]
            ):
                raise ValueError("first-call prompt contract differs")
            if base_call["adapter_enabled"] or not public_call["adapter_enabled"]:
                raise ValueError("adapter identity does not match base/public comparison")
            contracts.append(
                {
                    "task_id": public_job["task_id"],
                    "repeat": public_job["repeat"],
                    "seed": public_job["seed"],
                    "base_call": f"{base_id}-c000",
                    "public_call": f"{public_id}-c000",
                    "output_cap": base_call["cap"],
                }
            )
            raw_bindings.append(
                {
                    "base_first_call": bound_raw(base_report, base_call_path),
                    "public_first_call": bound_raw(public_report, public_call_path),
                }
            )
        rows.append(row)
    bounds = success_bounds(rows)
    if (bounds["planned"], bounds["known"], bounds["unknown"]) != (16, 13, 3):
        raise ValueError("frozen readout did not have the expected 13 known / 3 unknown slots")
    known = [row for row in rows if row["status"] == "known"]
    wins = sum(row["public"] > row["base"] for row in known)
    losses = sum(row["public"] < row["base"] for row in known)
    receipt = {
        "schema": "textcraft_base044_public056_audit_v2",
        "source_reports": {
            str(BASE_REPORT): sha256(BASE_REPORT),
            str(PUBLIC_REPORT): sha256(PUBLIC_REPORT),
        },
        "plans": {
            str(base_plan_path): sha256(base_plan_path),
            str(public_plan_path): sha256(public_plan_path),
        },
        "verified_plan_contract": plan_contract,
        "matched_job_count": 16,
        "native_first_call_contracts": contracts,
        "raw_receipt_bindings": raw_bindings,
        "rows": rows,
        "paired_known": {
            "pairs": len(known),
            "wins": wins,
            "losses": losses,
            "ties": len(known) - wins - losses,
        },
        "full_planned_success_bounds": bounds,
        "caveat": (
            "Base044 ended at owner cap: its three missing episodes remain unknown, not losses. "
            "This verifies PLAN/job identity, task and manifest bytes, and raw first-call parity "
            "where receipts exist; it does not establish full collector implementation equivalence."
        ),
        "script_sha256": sha256(Path(__file__)),
    }
    with args.report.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
