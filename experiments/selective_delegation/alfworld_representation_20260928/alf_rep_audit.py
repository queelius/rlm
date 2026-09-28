"""Exact response/token audit plus independent native action replay; no expert at evaluation."""

import argparse
import json
import sys
import time
from pathlib import Path

import alf_rep as r

legacy = r.module(r.LIBRARY / "analyze_alfworld_screen.py", "alf_rep_receipt_auditor")
actor_audit = r.module(r.LIBRARY / "analyze_alfworld_actor.py", "alf_rep_actor_identity_auditor")


def audit(output: Path, require_terminal=True):
    from transformers import AutoTokenizer

    plan = r.read(output / "PLAN.json")
    if require_terminal:
        import psutil

        owners = list(output.glob("OWNER-*.json"))
        if len(owners) != 1:
            raise ValueError("single scientific owner required")
        owner = r.read(owners[0])
        terminal = r.read(owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-")))
        try:
            process = psutil.Process(owner["pid"])
            if abs(process.create_time() - owner["create_time"]) < 0.01 and (
                process.status() != psutil.STATUS_ZOMBIE
            ):
                raise ValueError("scientific owner still live")
        except psutil.NoSuchProcess:
            pass
    else:
        terminal = None
    for name, digest in plan["source_sha256"].items():
        if r.sha(name) != digest:
            raise ValueError("scientific dependency changed")
    calls = {path.stem: r.read(path) for path in (output / "calls").glob("*.json")}
    starts = {path.stem: r.read(path) for path in (output / "starts").glob("*.json")}
    tokenizer = AutoTokenizer.from_pretrained(r.BASE, local_files_only=True)
    client = r.original.BaseClient(None, tokenizer, output, 0)
    projected = {}
    for cid, call in calls.items():
        if cid not in starts or any(
            call[k] != starts[cid][k] for k in ("request", "request_digest", "started")
        ):
            raise ValueError("native request/response binding mismatch")
        if call["trimming"]["dropped_history"] != 0 or (
            client.ids(call["request"]["prompt"]) != call["input_token_ids"]
        ):
            raise ValueError("full-history native tokenization differs")
        if (
            call["available"]
            and tokenizer.decode(
                call["output_token_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            != call["text"]
        ):
            raise ValueError("saved original output token decode differs")
        if plan["actor"] == "trained":
            binding = plan["checkpoint_binding"]
            projected[cid] = actor_audit.adapter_neutral_projection(
                call, binding["path"], binding["adapter_sha256"]
            )
            actor_audit.adapter_neutral_projection(
                starts[cid], binding["path"], binding["adapter_sha256"]
            )
        else:
            projected[cid] = call
    rows, audits = [], {}
    for job in plan["cases"]:
        path = output / "episodes" / (job["episode_id"] + ".json")
        if not path.exists():
            continue
        row = r.read(path)
        previous = sys.modules.get("alfworld_closed_loop")
        sys.modules["alfworld_closed_loop"] = r.controller(plan["representation"])
        try:
            # This local algorithm selector activates shared rejection-history accounting.
            # Scientific PLAN/interface/receipts are neither rewritten nor relabeled.
            checked = legacy.audit_episode(
                job,
                row,
                projected,
                r.read,
                output,
                {**plan, "schema": "alfworld-closed-loop-indexed-v1"},
            )
        finally:
            if previous is None:
                sys.modules.pop("alfworld_closed_loop", None)
            else:
                sys.modules["alfworld_closed_loop"] = previous
        env = r.actor.unseen.CheckedBridge(
            job["game"],
            plan["alfworld_python"],
            plan["data_root"],
            time.time() + 120,
            output / (job["episode_id"] + "-audit.stderr"),
        )
        env.expected_public = job["game"]["public"]
        try:
            actual = env.reset()
            expected = r.read(output / "observations" / (job["episode_id"] + "-000.json"))
            if actual != expected:
                raise ValueError("independent native reset mismatch")
            for index in range(1, row["actions"] + 1):
                expected = r.read(
                    output / "observations" / (job["episode_id"] + f"-{index:03d}.json")
                )
                if env.step(expected["action"]) != expected:
                    raise ValueError("independent native command transition mismatch")
        finally:
            env.close()
        audits[row["episode_id"]] = dict(checked, native_replayed=True)
        rows.append(row)
    covered = [cid for row in rows for cid in row["call_ids"]]
    if len(covered) != len(set(covered)) or set(covered) != set(calls):
        raise ValueError("orphan or reused scientific calls")
    return dict(
        schema="alfworld-representation-native-audit-20260928-v1",
        plan=plan,
        terminal=terminal,
        rows=rows,
        audits=audits,
        complete=len(rows) == len(plan["cases"]) and all(row["observed"] for row in rows),
        unresolved_starts=sorted(set(starts) - set(calls)),
        physical_cost=r.original.evaluation.cost(list(calls.values())),
        receipt_sha256={
            str(path): r.sha(path)
            for folder in ("observations", "calls", "starts", "decisions", "episodes")
            for path in sorted((output / folder).glob("*.json"))
        },
        audit_scope="Authentic adapter identity before host-only neutral projection; exact saved "
        "prompt/token/strict decoder replay plus fresh native TextWorld reset/command transitions. "
        "No hidden expert. Unknown episodes stay unknown.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output)
    r.save(args.output / "NATIVE-AUDIT.json", report)
    print(json.dumps(dict(complete=report["complete"], native_replayed=len(report["audits"]))))
