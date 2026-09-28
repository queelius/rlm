"""Project every accepted public teacher action into the compact interface, CPU only."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import compact_bridge as compact

sys.path.insert(0, str(compact.LIBRARY))

import inspect_textcraft_worlds as worlds  # noqa: E402
import prepare_textcraft_sft as encoder  # noqa: E402
import train_textcraft_public as accepted  # noqa: E402

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
ORIGINAL = ROOT / "textcraft-public-discovery-prototype-001"
OUTPUT = ROOT / "textcraft-compact-actions-20260928-001"
sha = accepted.sha
save = encoder.save


def prepare(output: Path) -> dict:
    if output.exists():
        raise FileExistsError("immutable compact dataset already exists")
    contract = accepted.validate_prepared(ORIGINAL)
    old_manifest = accepted.read(ORIGINAL / "MANIFEST.json")
    rows = [json.loads(line) for line in (ORIGINAL / "rows.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (ORIGINAL / "tasks.jsonl").read_text().splitlines()]
    output.mkdir(parents=True)
    with (output / "tasks.jsonl").open("xb") as stream:
        stream.write((ORIGINAL / "tasks.jsonl").read_bytes())
    save(
        output / "DESIGN.json",
        {
            "same_tasks_and_action_order": True,
            "no_success_filtering": True,
            "source_contract": contract,
            "source_rows_sha256": sha(ORIGINAL / "rows.jsonl"),
            "projection": "Remove craft.ingredients from targets and all previous action histories; "
            "replay public native get_info/craft/finish to rebuild instructions, inventory, "
            "remaining-token counters and target-only JSON+EOS masks. Preserve target/output_count.",
            "selection": "All32 accepted TRAIN tasks and366 rows, without new selection or models.",
            "scope": "Same epoch/rows/updates/seed/base; NOT equal token dose, FLOPs or wall time.",
        },
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        encoder.BASE, local_files_only=True, trust_remote_code=False
    )
    world = compact.load_world()
    projected, receipts = [], []
    for task in tasks:
        original = [row for row in rows if row["task_id"] == task["id"]]
        frame = compact.Frame(
            world,
            dict(task["misc"]["initial_inventory"]),
            task["misc"]["target_items"],
            compact.Budget(),
            0,
        )
        history, trace, error = [], [], None
        try:
            for index, row in enumerate(original):
                if row["step"] != index or not row["prompt"].startswith(compact.native.INSTRUCTION):
                    raise ValueError("source step/public prompt identity mismatch")
                state = json.loads(row["prompt"][len(compact.native.INSTRUCTION) :])
                if state["current_inventory"] != frame.inventory:
                    raise ValueError("projection changed source native state")
                full_action = compact.native.parse_action(row["target"])
                action = compact.project_action(full_action)
                prompt = compact.public_prompt(frame, history, goal=task["goal"])
                encoded = encoder.encode_row(prompt, action, tokenizer)
                if encoded["prompt_tokens"] + 256 > 8192 or encoded["target_tokens"] > 256:
                    raise ValueError("compact prompt/action exceeds unchanged context/cap")
                compact.parse_action(encoded["target"])
                frame.budget.charge(encoded["target_tokens"])
                reply = frame.apply(action)
                if reply != row["feedback"]:
                    raise ValueError("compact binding changed exact native teacher feedback")
                if full_action["action"] == "craft" and (
                    frame.execution_assists[-1]["executed_action"] != full_action
                ):
                    raise ValueError(
                        "projected craft differs from original executed teacher action"
                    )
                projected.append(
                    {
                        "task_id": task["id"],
                        "step": index,
                        **encoded,
                        "feedback": reply,
                        "source_target_sha256": hashlib.sha256(row["target"].encode()).hexdigest(),
                    }
                )
                history.append({"action": action, "feedback": reply})
                trace.append(
                    {
                        "action": action,
                        "feedback": reply,
                        "execution_assist": frame.execution_assists[-1],
                    }
                )
            if not frame.finished:
                raise ValueError("projected task omitted finish")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        score, details = frame.score()
        receipt = dict(
            task_id=task["id"],
            rows=len(trace),
            expected_rows=len(original),
            native_score=score,
            native_details=details,
            finish_included=frame.finished,
            error=error,
            eligible=error is None and score == 1 and frame.finished,
            output_tokens=frame.budget.output_tokens,
        )
        path = output / "traces" / (task["id"] + ".json")
        save(path, {"receipt": receipt, "trace": trace})
        receipt["trace_sha256"] = sha(path)
        receipts.append(receipt)
    with (output / "rows.jsonl").open("x") as stream:
        for row in projected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    same_order = [(r["task_id"], r["step"]) for r in projected] == [
        (r["task_id"], r["step"]) for r in rows
    ]
    manifest = dict(
        schema="textcraft-compact-public-discovery-projection-20260928-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        ready=len(projected) == 366 and same_order and all(r["eligible"] for r in receipts),
        selected_task_count=32,
        eligible_task_count=sum(r["eligible"] for r in receipts),
        task_ids=[t["id"] for t in tasks],
        tasks=receipts,
        tasks_sha256=sha(output / "tasks.jsonl"),
        rows=len(projected),
        rows_sha256=sha(output / "rows.jsonl"),
        same_source_row_order=same_order,
        action_counts=dict(Counter(json.loads(r["target"])["action"] for r in projected)),
        prompt_tokens=sum(r["prompt_tokens"] for r in projected),
        supervised_tokens_including_eos=sum(r["target_tokens"] for r in projected),
        max_prompt_tokens=max(r["prompt_tokens"] for r in projected),
        max_target_tokens_including_eos=max(r["target_tokens"] for r in projected),
        original=dict(
            prepared=str(ORIGINAL),
            manifest_sha256=sha(ORIGINAL / "MANIFEST.json"),
            rows_sha256=sha(ORIGINAL / "rows.jsonl"),
            prompt_tokens=old_manifest["prompt_tokens"],
            supervised_tokens_including_eos=old_manifest["supervised_tokens_including_eos"],
        ),
        model=str(encoder.BASE),
        model_manifest_sha256=sha(encoder.BASE / "local-research-manifest.json"),
        world_seed=42,
        world_sha256=worlds.digest(worlds.snapshot(world)),
        native_source=compact.trusted_provenance(),
        source_revision=encoder.original.COMMIT,
        source_url="https://github.com/ApGa/platoon",
        source_license="MIT",
        design_sha256=sha(output / "DESIGN.json"),
        source_sha256={
            str(path.resolve()): sha(path)
            for path in (
                Path(__file__),
                Path(compact.__file__),
                Path(compact.native.__file__),
                Path(encoder.__file__),
                Path(accepted.__file__),
                Path(accepted.recipe.__file__),
                Path(worlds.__file__),
            )
        },
        failure_policy="Every task/failed trace retained; no replacement or success-based selection.",
        dose_caveat="One matched epoch,366rows,23updates, same seed/base/optimizer; "
        "changed instruction/history lengths and output token dose, NOT matched compute.",
        gpu_used=False,
        model_calls=0,
        environment_modified=False,
    )
    save(output / "MANIFEST.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output.resolve())
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "ready",
                    "rows",
                    "eligible_task_count",
                    "prompt_tokens",
                    "supervised_tokens_including_eos",
                    "original",
                )
            },
            indent=2,
        )
    )
