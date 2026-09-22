"""CPU-only first063 terminal-RLOO credit census; no reward/gradient changes."""

import argparse
import hashlib
import json
from pathlib import Path

import textcraft_bridge as bridge
import textcraft_trajectory_loss as loss_math


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def error_type(history):
    feedback = history["feedback"]
    if "response" in history:
        return "invalid_schema"
    if isinstance(feedback, str) and feedback.startswith("Rejected action:"):
        return "rejected_action"
    if isinstance(feedback, str) and feedback.startswith("Error:"):
        return "native_action_error"
    if isinstance(feedback, list) and any(isinstance(x, dict) and "error" in x for x in feedback):
        return "query_item_error"
    return "no_explicit_error"


def totals(rows):
    weighted = sum(r["advantage"] * r["tokens"] for r in rows)
    return dict(
        calls=len(rows),
        episodes=len({r["episode_id"] for r in rows}),
        tokens=sum(r["tokens"] for r in rows),
        advantage_weighted_tokens=weighted,
        absolute_advantage_weighted_tokens=sum(abs(r["advantage"]) * r["tokens"] for r in rows),
        signed_score_function_coefficient_sum=weighted / 32,
    )


def analyze(output, native_report):
    native = read(native_report)["native_audit"]
    hashes = native["sha256"]

    def checked(path):
        if hashes.get(str(path)) != sha(path):
            raise ValueError("receipt missing/changed from completed native audit: " + str(path))
        return read(path)

    plan = checked(output / "PLAN.json")
    calls = {path.stem: checked(path) for path in (output / "calls").glob("*.json")}
    episodes = [checked(output / "episodes" / f"{job['episode_id']}.json") for job in plan["jobs"]]
    credits = loss_math.batch_credits(episodes, calls)
    credit = {item.call_id: item.advantage for item in credits}
    rows, episode_rows = [], []
    for episode in episodes:
        if episode["node_ids"] != ["n0"]:
            raise ValueError("first063 flat root inventory required")
        node_path = output / "nodes" / f"{episode['episode_id']}-n0.json"
        node = checked(node_path)
        if node["call_ids"] != episode["call_ids"]:
            raise ValueError("saved native history ordering mismatch")
        local = []
        for cid, history in zip(node["call_ids"], node["public_history"], strict=True):
            call = calls[cid]
            if "action" in history:
                action = bridge.parse_action(call["text"])
                if action != history["action"]:
                    raise ValueError("call action differs from native public history")
                action_name = action["action"]
            else:
                if call["text"] != history["response"]:
                    raise ValueError("invalid response/history mismatch")
                action_name = "unparsed"
            row = dict(
                call_id=cid,
                episode_id=episode["episode_id"],
                task_id=episode["task_id"],
                repeat=episode["repeat"],
                reward=episode["native_score"],
                advantage=credit[cid],
                action=action_name,
                error=error_type(history),
                tokens=len(call["output_token_ids"]),
                sign="positive" if credit[cid] > 0 else "negative" if credit[cid] < 0 else "zero",
                node_path=str(node_path),
                call_path=str(output / "calls" / f"{cid}.json"),
            )
            rows.append(row)
            local.append(row)
        episode_rows.append(
            dict(
                episode_id=episode["episode_id"],
                task_id=episode["task_id"],
                repeat=episode["repeat"],
                reward=episode["native_score"],
                advantage=local[0]["advantage"],
                status=episode["status"],
                **totals(local),
            )
        )
    by_sign = {
        sign: totals([r for r in rows if r["sign"] == sign])
        for sign in ("positive", "negative", "zero")
    }
    categories = sorted({(r["action"], r["error"]) for r in rows})
    by_action_error = {
        f"{action}/{error}": {
            sign: totals(
                [
                    r
                    for r in rows
                    if r["action"] == action and r["error"] == error and r["sign"] == sign
                ]
            )
            for sign in ("positive", "negative", "zero")
        }
        for action, error in categories
    }
    groups = {
        task: [r for r in episode_rows if r["task_id"] == task]
        for task in sorted({r["task_id"] for r in rows})
    }
    examples = [
        r
        for r in rows
        if r["sign"] == "positive"
        and r["action"] == "craft"
        and r["error"] == "native_action_error"
    ]
    return dict(
        schema="textcraft-first063-credit-census-v1",
        planned_tasks=8,
        planned_episodes=32,
        native_successes=sum(e["native_score"] for e in episodes),
        all_calls=totals(rows),
        by_sign=by_sign,
        by_action_error=by_action_error,
        credited_calls=totals([r for r in rows if r["advantage"]]),
        groups=groups,
        positive_failed_craft_examples=sorted(examples, key=lambda r: r["call_id"])[:3],
        call_rows=rows,
        source_sha256={
            str(Path(m.__file__).resolve()): sha(Path(m.__file__)) for m in (bridge, loss_math)
        },
        analyzer_sha256=sha(Path(__file__)),
        native_report=str(native_report),
        native_report_sha256=sha(native_report),
        source_output=str(output),
        checked_native_receipt_count=1 + len(calls) + 2 * len(episodes),
        method="All32 complete observed episodes;8groups x4. Existing native audit receipts "
        "checked; existing batch_credits. Reward1/0 from root native finish. "
        "Weighted tokens=sum(A*emitted_count); score-function coefficient=A/32. "
        "Actual emitted EOS only, no added tokens or filters.",
        caveat="TRAIN objective census, not new optimization or held performance. "
        "No explicit error does not imply a helpful action. Positive trajectory "
        "credit is not causal action credit or guaranteed probability increase; "
        "shared gradients, Adam and clipping matter. Gradient norm/clipping not measured.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    result = analyze(args.output.resolve(), args.native_report.resolve())
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write("# First063 trajectory credit census\n\n" + result["caveat"] + "\n\n")
        stream.write(
            "| Credit | Episodes | Calls | Emitted tokens | Sum A×tokens |\n"
            "|---|---:|---:|---:|---:|\n"
        )
        for name, row in result["by_sign"].items():
            stream.write(
                f"|{name}|{row['episodes']}|{row['calls']}|{row['tokens']}|"
                f"{row['advantage_weighted_tokens']:.3f}|\n"
            )
        failed = result["by_action_error"].get("craft/native_action_error", {})
        stream.write("\nFailed-craft credit: " + json.dumps(failed) + "\n")
