"""Focused CPU fixture: original rows, pinned files, public-only saved native traces."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import prepare

MANIFEST_SHA = "76e387378c9a89b0d8e26180874db61ad40086721c6cfef9e3b78dd77f915e05"


def check(output: Path, manifest_sha: str = MANIFEST_SHA) -> dict:
    manifest = json.loads((output / "MANIFEST.json").read_text())
    world = prepare.bridge.load_world()
    if prepare.worlds.digest(prepare.worlds.snapshot(world)) != manifest["world_sha256"]:
        raise ValueError("world pin mismatch")
    groups, seen = {}, set()
    for group in prepare.GROUPS:
        tasks = prepare.load_group(output, group, manifest_sha)
        meta = manifest["groups"][group]
        if prepare.sha(output / group / "MANIFEST.json") != meta["manifest_sha256"]:
            raise ValueError("group manifest pin mismatch")
        actions = []
        for task, audit in zip(tasks, meta["audits"], strict=True):
            if prepare.roots(task) & seen:
                raise ValueError("cross-group root duplicate")
            seen.update(prepare.roots(task))
            path = output / group / "native" / (task["id"] + ".json")
            if prepare.sha(path) != audit["trace_sha256"]:
                raise ValueError("native trace pin mismatch")
            saved = json.loads(path.read_text())
            initial = dict(task["misc"]["initial_inventory"])
            targets = dict(task["misc"]["target_items"])
            frame = prepare.bridge.Frame(
                world, dict(initial), targets, prepare.bridge.Budget(512, 131072), 0
            )
            observed = {}
            for step in saved["trace"]:
                expected = prepare.teacher.next_action(
                    targets, initial, dict(frame.inventory), observed
                )
                if step["action"] != expected:
                    raise ValueError("saved action is not derivable from observed public recipes")
                frame.budget.charge(0)
                reply = frame.apply(step["action"])
                if reply != step["feedback"]:
                    raise ValueError("native feedback replay mismatch")
                if step["action"]["action"] == "get_info":
                    for item in reply:
                        observed[item["item"]] = {
                            "is_base": item["is_base"],
                            "recipes": [
                                {
                                    "ingredients": dict(r["ingredients"]),
                                    "result_count": r["result_count"],
                                }
                                for r in item["recipes"]
                            ],
                        }
            score, _ = frame.score()
            if score != 1 or not frame.finished:
                raise ValueError("saved native solution no longer succeeds")
            prompt = prepare.bridge.initial_prompt(task, "flat")
            if hashlib.sha256(prompt.encode()).hexdigest() != audit["initial_prompt_sha256"]:
                raise ValueError("initial prompt dictionary order changed")
            actions.append(frame.budget.calls)
        groups[group] = dict(
            tasks=len(tasks),
            native_scores_all_one=True,
            native_actions=actions,
            original_row_bytes_verified=True,
        )
    try:
        prepare.load_group(output, "train", "0" * 64)
    except ValueError:
        pass
    else:
        raise ValueError("loader accepted incorrect manifest pin")
    return dict(
        checked_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=manifest_sha,
        groups=groups,
        disjoint_root_count=len(seen),
        passed=True,
        gpu_used=False,
        verifier_sha256=prepare.sha(Path(__file__).resolve()),
        checks="Pinned root/group/task/trace files; unchanged original official TRAIN lines; "
        "no protected ID/root overlap; exact saved public-action/native-feedback replay; "
        "score1 and finish on all16; initial prompt identity; incorrect pin rejected.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=prepare.OUTPUT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = check(args.output.resolve())
    if args.receipt:
        prepare.save(args.receipt, result)
    print(json.dumps(result, indent=2))
