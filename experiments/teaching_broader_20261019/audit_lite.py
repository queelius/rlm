"""Lightweight independent audit for four-task broader-structure readouts.

The sealed October audit requires eight parent tasks per output, so it cannot run on these
panels. This audit recomputes each episode's success from its saved final inventory against the
task's target items, cross-checks the collector's native_score, verifies that every planned
episode was recorded exactly once, and totals physical cost from saved call records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(output: Path) -> dict:
    plan = json.loads((output / "PLAN.json").read_text())
    prepared = Path(plan["prepared"])
    tasks = {}
    for line in (prepared / "tasks.jsonl").read_text().splitlines():
        task = json.loads(line)
        tasks[task["id"]] = task
    if sha(prepared / "tasks.jsonl") != plan["tasks_sha256"]:
        raise ValueError("frozen tasks changed")
    planned = {j["episode_id"]: j for j in plan["jobs"]}
    rows = {p.stem: json.loads(p.read_text()) for p in (output / "episodes").glob("*.json")}
    calls = [json.loads(p.read_text()) for p in (output / "calls").glob("*.json")]
    problems = []
    if set(rows) != set(planned):
        problems.append(f"recorded episodes {sorted(set(rows) ^ set(planned))} differ from plan")
    per_task = {}
    recomputed_wins = 0
    for episode_id, row in sorted(rows.items()):
        job = planned.get(episode_id, {})
        if job and (row["task_id"] != job["task_id"] or row["seed"] != job["seed"]):
            problems.append(f"{episode_id}: task/seed differ from plan")
        targets = tasks[row["task_id"]]["misc"]["target_items"]
        inventory = row.get("final_inventory") or {}
        recomputed = all(inventory.get(item, 0) >= qty for item, qty in targets.items())
        native = row.get("native_score") == 1
        if row.get("observed") and recomputed != native:
            problems.append(f"{episode_id}: recomputed success {recomputed} != native_score {row.get('native_score')}")
        recomputed_wins += bool(row.get("observed") and recomputed)
        t = per_task.setdefault(row["task_id"], dict(attempts=0, wins=0, statuses=Counter()))
        t["attempts"] += 1
        t["wins"] += bool(row.get("observed") and recomputed)
        t["statuses"][row.get("status")] += 1
    call_ids = Counter(c["call_id"] for c in calls)
    dup = [k for k, v in call_ids.items() if v > 1]
    if dup:
        problems.append(f"duplicate call ids: {dup[:5]}")
    episode_call_ids = set()
    for row in rows.values():
        ids = row.get("call_ids")
        ids = json.loads(ids) if isinstance(ids, str) else (ids or [])
        episode_call_ids.update(ids)
    unmatched = len(episode_call_ids - set(call_ids))
    if unmatched:
        problems.append(f"{unmatched} episode call ids without saved call records")
    cost = dict(
        calls=len(calls),
        prompt_tokens=sum(int(c.get("prompt_tokens") or 0) for c in calls),
        completion_tokens=sum(int(c.get("completion_tokens") or 0) for c in calls),
        failed_calls=sum(1 for c in calls if c.get("failed") or c.get("failure")),
    )
    result = dict(
        schema="teaching-broader-structure-lite-audit-v1",
        checked_epoch=time.time(),
        output=str(output),
        plan_sha256=sha(output / "PLAN.json"),
        condition=plan["conditions"][0],
        world_seed=plan["world_seed"],
        adapter=plan["adapter"]["path"],
        planned_episodes=len(planned),
        recorded_episodes=len(rows),
        observed_episodes=sum(bool(r.get("observed")) for r in rows.values()),
        wins_recomputed=recomputed_wins,
        wins_native=sum(bool(r.get("observed") and r.get("native_score") == 1) for r in rows.values()),
        per_task={k: dict(v, statuses=dict(v["statuses"])) for k, v in per_task.items()},
        physical_cost=cost,
        problems=problems,
        ok=not problems and len(rows) == len(planned),
        scope="Recomputes success from saved final inventories and plan identity; does not replay "
        "native requests or re-decode model outputs (the sealed eight-task audit does that).",
    )
    (output / "LITE-AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+", type=Path)
    args = parser.parse_args()
    for output in args.outputs:
        if not (output / "SUMMARY.json").exists():
            print(json.dumps({"output": str(output), "skipped": "no SUMMARY.json yet"}))
            continue
        r = audit(output)
        print(json.dumps({k: r[k] for k in ("condition", "world_seed", "recorded_episodes", "wins_recomputed", "wins_native", "ok", "problems")}))
