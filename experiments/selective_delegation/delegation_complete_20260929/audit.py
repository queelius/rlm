"""Audit every scheduled root using the unmodified recursive native transition auditor."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import complete as c  # noqa: E402


def analyze(output: Path, *, terminal: bool = True) -> dict:
    import psutil
    from transformers import AutoTokenizer

    plan = c.read(output / "PLAN.json")
    c.check_plan(plan)
    collector, auditor = c.implementation(plan["routing"])
    paths = [output / "PLAN.json"]
    if terminal:
        owners = list(output.glob("OWNER-*.json"))
        if len(owners) != 1:
            raise ValueError("one completed scientific owner required")
        owner = c.read(owners[0])
        end = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
        c.read(end)
        paths += [owners[0], end]
        if str(Path(owner["source"]).resolve()) not in plan["source_sha256"]:
            raise ValueError("unbound scientific owner")
        try:
            process = psutil.Process(owner["pid"])
            if abs(process.create_time() - owner["create_time"]) < 0.01 and (
                process.status() != psutil.STATUS_ZOMBIE
            ):
                raise ValueError("scientific owner still live")
        except psutil.NoSuchProcess:
            pass
    tasks = [
        json.loads(line)
        for line in (Path(plan["prepared"]) / "tasks.jsonl").read_text().splitlines()
    ]
    world = c.flex.routing.native.load_world()
    if c.data.worlds.digest(c.data.worlds.snapshot(world)) != plan["world_sha256"]:
        raise ValueError("native world changed")
    tokenizer = AutoTokenizer.from_pretrained(
        collector.BASE, local_files_only=True, trust_remote_code=False
    )
    calls = {p.stem: c.read(p) for p in (output / "calls").glob("*.json")}
    starts = {p.stem: c.read(p) for p in (output / "starts").glob("*.json")}
    for cid, call in calls.items():
        if (
            cid not in starts
            or starts[cid]["request"] != call["request"]
            or (starts[cid]["request_digest"] != call["request_digest"])
        ):
            raise ValueError("call does not match actual saved request")
    rows, used, native_replays = [], set(), 0
    for job in plan["jobs"]:
        path = output / "episodes" / (job["episode_id"] + ".json")
        if not path.exists():
            rows.append(dict(**job, observed=False, native_score=None, reason="unattempted"))
            continue
        paths.append(path)
        row = c.read(path)
        local = {cid: calls[cid] for cid in row["call_ids"]}
        node_paths = [
            output / "nodes" / f"{job['episode_id']}-{nid}.json" for nid in row["node_ids"]
        ]
        nodes = {c.read(p)["node_id"]: c.read(p) for p in node_paths}
        paths.extend(node_paths)
        checked = auditor.audit_episode(
            tasks[0], job, row, local, nodes, plan, c.sha(output / "PLAN.json"), tokenizer, world
        )
        if row["observed"] and not checked["replayed"]:
            raise ValueError("known root must have full native transition replay")
        if used & local.keys():
            raise ValueError("calls reused across roots")
        used.update(local)
        native_replays += bool(checked["replayed"])
        if (
            sum(len(call["output_token_ids"]) for call in local.values() if call["available"])
            != (row["global_output_tokens"])
        ):
            raise ValueError("global returned tokens are not fully charged")
        rows.append(
            dict(
                **job,
                observed=row["observed"],
                native_score=row["native_score"],
                status=row["status"],
                failure=row["failure"],
                native_audit=checked,
                errors=row["errors"],
                physical_cost=collector.cost(list(local.values())),
                children=[
                    dict(
                        node_id=nid,
                        targets=node["targets"],
                        status=node["status"],
                        native_score=node["native_score"],
                        initial_inventory=node["initial_inventory"],
                        final_inventory=node["final_inventory"],
                    )
                    for nid, node in nodes.items()
                    if node["depth"] > 0
                ],
            )
        )
    for name in ("calls", "starts"):
        paths.extend((output / name).glob("*.json"))
    observed = sum(row["observed"] for row in rows)
    return dict(
        schema="textcraft-complete-goal-native-audit-20260929-v1",
        output=str(output),
        routing=plan["routing"],
        fixture=plan["fixture"],
        planned=2,
        observed=observed,
        unknown=2 - observed,
        successes=sum(row["observed"] and row["native_score"] == 1 for row in rows),
        full_native_replays=native_replays,
        rows=rows,
        physical_cost=collector.cost(list(calls.values())),
        role_cost={
            role: collector.cost([call for call in calls.values() if call["role"] == role])
            for role in ("root", "child")
        },
        errors=dict(sum((Counter(row.get("errors", {})) for row in rows), Counter())),
        unresolved_starts=sorted(starts.keys() - calls.keys()),
        calls_without_episode=sorted(calls.keys() - used),
        receipt_sha256={str(p): c.sha(p) for p in paths},
        scope="BOTH scheduled jobs; observed roots fully replay strict sampled actions, binder "
        "assists, recursive order, shared stock, child entry snapshots, return feedback, root "
        "original scoring snapshot and budgets. Unknown roots retain receipt-only audit.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=c.MODES, required=True)
    parser.add_argument("--study", type=Path, default=c.STUDY)
    args = parser.parse_args()
    output = args.study / args.mode
    report = analyze(output)
    c.persist(output / "NATIVE-AUDIT.json", report)
    print(json.dumps({k: report[k] for k in ("routing", "observed", "successes", "unknown")}))
