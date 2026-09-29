"""CPU-only preparation of the prospectively declared recovery-teaching comparison."""

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

LIBRARY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIBRARY))
import inspect_textcraft_worlds as worlds  # noqa: E402
import prepare_textcraft_public as teacher  # noqa: E402
import prepare_textcraft_sft as encoder  # noqa: E402
import textcraft_bridge as bridge  # noqa: E402

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "textcraft-recovery-data-20260929-001"
DIAGNOSIS = LIBRARY / "fresh_reward_diagnosis_20260929"
DATASET = ROOT / "textcraft-fresh-train-20260928-001/train"
COLLECTION = ROOT / "textcraft-fresh-rl-20260928-002/raw/collect-0001"
DIAGNOSIS_SHA = "fae1e9b6f69c9b408f528d118d39a26b884755990d966afa641d2385fa620fd2"
EXPERIMENT_SHA = "3cbbbe58652d4508e5668e9ac7a99dda64d84e7ca51f4a446c86f80a6eb488d8"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def action_type(row: dict) -> str:
    return json.loads(row["target"])["action"]


def select_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("four distinct rows required")
    selected = [0]
    for reverse in (False, True):
        choices = [
            i for i, row in enumerate(rows) if i not in selected and action_type(row) == "craft"
        ]
        if not choices:
            raise ValueError("four distinct rows require two unused crafts")
        selected.append(choices[-1] if reverse else choices[0])
    finishes = [
        i for i, row in enumerate(rows) if i not in selected and action_type(row) == "finish"
    ]
    if len(finishes) != 1:
        raise ValueError("four distinct rows require one finish")
    selected.append(finishes[0])
    return [rows[i] for i in sorted(selected)]


def match_rows(recovery: list[dict], clean: list[dict]) -> tuple[list[dict], list[dict]]:
    used, matched, pairs = set(), [], []
    for row in recovery:
        candidates = [
            other
            for other in clean
            if other["row_id"] not in used
            and other["task_id"] == row["task_id"]
            and action_type(other) == action_type(row)
        ]
        if not candidates:
            raise ValueError("no unused same-task/action-type control row")
        chosen = min(
            candidates,
            key=lambda other: (abs(other["target_tokens"] - row["target_tokens"]), other["step"]),
        )
        used.add(chosen["row_id"])
        matched.append(chosen)
        pairs.append(
            dict(
                recovery_row_id=row["row_id"],
                ordinary_row_id=chosen["row_id"],
                task_id=row["task_id"],
                action_type=action_type(row),
                recovery_target_tokens=row["target_tokens"],
                ordinary_target_tokens=chosen["target_tokens"],
                target_token_difference=abs(row["target_tokens"] - chosen["target_tokens"]),
                identical_target=row["target"] == chosen["target"],
            )
        )
    return matched, pairs


def restore_frame(state: dict, world) -> bridge.Frame:
    if state["agent_depth"] != 0 or state["max_agent_depth"] != 0 or state["delegated_context"]:
        raise ValueError("only original flat root contexts supported")
    if (
        not 0 <= state["global_calls_remaining"] <= 96
        or not 0 <= state["global_output_tokens_remaining"] <= 8192
    ):
        raise ValueError("invalid saved remaining budget")
    budget = bridge.Budget(96, 8192)
    budget.calls = 96 - state["global_calls_remaining"]
    budget.output_tokens = 8192 - state["global_output_tokens_remaining"]
    frame = bridge.Frame(
        world, dict(state["current_inventory"]), dict(state["target_items"]), budget, 0
    )
    frame.initial_inventory = dict(state["inventory_at_task_start"])
    return frame


def public_recipes(history: list[dict]) -> dict:
    known = {}
    for entry in history:
        if entry.get("action", {}).get("action") != "get_info" or not isinstance(
            entry.get("feedback"), list
        ):
            continue
        for info in entry["feedback"]:
            known[info["item"]] = {
                "is_base": info["is_base"],
                "recipes": [
                    {"ingredients": dict(r["ingredients"]), "result_count": r["result_count"]}
                    for r in info["recipes"]
                ],
            }
    return known


def generate(task: dict, frame, history: list[dict], tokenizer, arm: str, saved_call=None):
    """Native driver; only explicit public copies cross the teacher decision boundary."""
    rows, first_calls, first_tokens = [], frame.budget.calls, frame.budget.output_tokens
    first_history_length = len(history)
    failure = None
    status = "unfinished"
    try:
        while not frame.finished:
            cap = frame.budget.reserve()
            public_input = dict(
                targets=dict(frame.targets),
                initial_inventory=dict(frame.initial_inventory),
                current_inventory=dict(frame.inventory),
                observed_recipes=public_recipes(history),
            )
            action = teacher.next_action(**public_input)
            prompt = bridge.public_prompt(frame, history, goal=task["goal"])
            row = encoder.encode_row(prompt, action, tokenizer)
            prefix_ids = row["input_ids"][: row["prompt_tokens"]]
            if not rows and saved_call is not None:
                if prompt != saved_call["request"]["prompt"]:
                    raise ValueError("saved actor prompt bytes changed")
                if (
                    prefix_ids != saved_call["input_token_ids"]
                    or prefix_ids != saved_call["request"]["input_token_ids"]
                ):
                    raise ValueError("saved actor prefix token IDs changed")
            if row["prompt_tokens"] + cap > 8192 or row["target_tokens"] > cap:
                raise ValueError("full original context or response cap exceeded")
            bridge.parse_action(row["target"])
            step = len(rows)
            row.update(
                row_id=f"{arm}:{task['id']}:s{step:03d}",
                task_id=task["id"],
                split="train",
                dataset_group="A",
                arm=arm,
                step=step,
                history_entries=len(history),
                global_calls_before=frame.budget.calls,
                global_output_tokens_before=frame.budget.output_tokens,
                per_response_cap=cap,
                prompt_plus_cap=row["prompt_tokens"] + cap,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                input_token_ids_sha256=digest(prefix_ids),
                label_id="sha256:" + hashlib.sha256(row["target"].encode()).hexdigest(),
                target_ids=row["input_ids"][row["prompt_tokens"] :],
                teacher_public_input_sha256=digest(public_input),
            )
            frame.budget.charge(row["target_tokens"])
            feedback = frame.apply(action)
            row["feedback"] = (
                feedback  # audit metadata only, never part of that row's prefix/target
            )
            rows.append(row)
            history.append({"action": action, "feedback": feedback})
            if isinstance(feedback, str) and feedback.startswith("Error:"):
                raise ValueError("native teacher action rejected: " + feedback)
        status = "finished"
    except Exception as exc:
        failure, status = f"{type(exc).__name__}: {exc}", "failed"
    score, details = frame.score()  # qualification only; no score enters next_action
    audit = dict(
        task_id=task["id"],
        arm=arm,
        status=status,
        failure=failure,
        native_score=score,
        native_details=details,
        prefix_calls=first_calls,
        prefix_output_tokens=first_tokens,
        prefix_history_entries=first_history_length,
        teacher_suffix_calls=len(rows),
        total_calls=frame.budget.calls,
        total_output_tokens=frame.budget.output_tokens,
        max_prompt_plus_cap=max((r["prompt_plus_cap"] for r in rows), default=0),
        first_saved_prompt_and_input_ids_equal=saved_call is not None and bool(rows),
        eligible=status == "finished" and score == 1,
    )
    return rows, audit


def prepare(output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError(
            "explicit empty CUDA_VISIBLE_DEVICES required; tokenizer-only CPU preparation"
        )
    if output.resolve() != OUTPUT:
        raise ValueError("only the newly assigned immutable output is allowed")
    output.mkdir(parents=False, exist_ok=False)
    save(
        output / "ATTEMPT.json",
        dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            question="frozen A-only recovery vs ordinary public teaching data",
            GPU_loaded=False,
            training=False,
            source=str(Path(__file__).resolve()),
            source_sha256=sha(Path(__file__)),
            output=str(output),
        ),
    )
    try:
        if (
            sha(DIAGNOSIS / "RESULTS.json") != DIAGNOSIS_SHA
            or sha(DIAGNOSIS / "EXPERIMENT.md") != EXPERIMENT_SHA
        ):
            raise ValueError("approved diagnosis/spec changed")
        diagnosis = json.loads((DIAGNOSIS / "RESULTS.json").read_text())
        for path, expected in diagnosis["source_sha256"].items():
            if sha(Path(path)) != expected:
                raise ValueError("existing source changed: " + path)
        plan = json.loads((COLLECTION / "PLAN.json").read_text())
        native = json.loads((COLLECTION / "NATIVE-AUDIT.json").read_text())
        for name in ["PLAN.json", "NATIVE-AUDIT.json", "SUMMARY.json"]:
            path = COLLECTION / name
            if sha(path) != diagnosis["small_input_sha256"][str(path)]:
                raise ValueError("source collection receipt changed")
        if (
            sha(DATASET / "tasks.jsonl") != diagnosis["dataset"]["tasks_sha256"]
            or sha(DATASET / "MANIFEST.json") != diagnosis["dataset"]["manifest_sha256"]
        ):
            raise ValueError("official A dataset changed")
        tasks = [json.loads(line) for line in (DATASET / "tasks.jsonl").read_text().splitlines()]
        if len(tasks) != 8 or any(not t["id"].startswith("textcraft_synth.train.") for t in tasks):
            raise ValueError("eight official TRAIN goals required")
        with (output / "tasks.jsonl").open("xb") as stream:
            stream.write((DATASET / "tasks.jsonl").read_bytes())
        model_manifest_path = encoder.BASE / "local-research-manifest.json"
        if sha(model_manifest_path) != plan["model_manifest_sha256"]:
            raise ValueError("pinned model/tokenizer manifest changed")
        model_manifest = json.loads(model_manifest_path.read_text())
        tokenizer_pins = {}
        for name, expected in model_manifest["files"].items():
            if name in {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "vocab.json",
                "merges.txt",
                "config.json",
            }:
                path = encoder.BASE / name
                if sha(path) != expected:
                    raise ValueError("pinned tokenizer asset changed")
                tokenizer_pins[str(path)] = expected
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            encoder.BASE, local_files_only=True, trust_remote_code=False
        )
        world = bridge.load_world()
        if worlds.digest(worlds.snapshot(world)) != diagnosis["dataset"]["world_sha256"]:
            raise ValueError("native world42 changed")
        recovery_pool, ordinary_pool, selected, audits, prefix_receipts = [], [], [], [], []
        for index, task in enumerate(tasks):
            eid = f"t{index:02d}-r0-flat"
            node_path = COLLECTION / "nodes" / f"{eid}-n0.json"
            if sha(node_path) != native["receipt_sha256"][str(node_path)]:
                raise ValueError("saved root-node receipt changed")
            node = json.loads(node_path.read_text())
            error_index = next(
                i
                for i, h in enumerate(node["public_history"])
                if isinstance(h.get("feedback"), str) and h["feedback"].startswith("Error:")
            )
            call_path = COLLECTION / "calls" / f"{node['call_ids'][error_index + 1]}.json"
            if sha(call_path) != native["receipt_sha256"][str(call_path)]:
                raise ValueError("saved request receipt changed")
            call = json.loads(call_path.read_text())
            frozen = diagnosis["public_recovery_fixture"]["records"][index]
            if (
                str(call_path) != frozen["saved_next_call"]
                or sha(call_path) != frozen["saved_next_call_sha256"]
            ):
                raise ValueError("prefix differs from declared eight-prefix selection")
            if call["request"]["task_id"] != task["id"] or not call["request"]["prompt"].startswith(
                bridge.INSTRUCTION
            ):
                raise ValueError("actual original actor interface/task differs")
            state = json.loads(call["request"]["prompt"][len(bridge.INSTRUCTION) :])
            if (
                state["history"] != node["public_history"][: error_index + 1]
                or state["inventory_at_task_start"] != task["misc"]["initial_inventory"]
                or state["target_items"] != task["misc"]["target_items"]
                or state["goal"] != task["goal"]
            ):
                raise ValueError("root baseline, goal or full saved history mismatch")
            frame = restore_frame(state, world)
            rows, audit = generate(task, frame, list(state["history"]), tokenizer, "recovery", call)
            jsonl(output / f"recovery-trajectory-{index:02d}.jsonl", rows)
            save(output / f"recovery-audit-{index:02d}.json", audit)
            audits.append(audit)
            if not audit["eligible"]:
                raise ValueError("selected recovery prefix failed qualification; no replacement")
            selected.extend(select_rows(rows))
            recovery_pool.extend(rows)
            prefix_receipts.append(
                dict(
                    task_id=task["id"],
                    episode_id=eid,
                    first_error_index=error_index,
                    source_node=str(node_path),
                    source_node_sha256=sha(node_path),
                    source_request=str(call_path),
                    source_request_sha256=sha(call_path),
                    prompt_sha256=rows[0]["prompt_sha256"],
                    input_token_ids_sha256=rows[0]["input_token_ids_sha256"],
                    exact_prompt_and_token_ids=True,
                    prefix_calls=frame.budget.calls - len(rows),
                    original_initial_inventory=dict(frame.initial_inventory),
                )
            )
            clean_frame = bridge.Frame(
                world,
                dict(task["misc"]["initial_inventory"]),
                task["misc"]["target_items"],
                bridge.Budget(96, 8192),
                0,
            )
            clean, clean_audit = generate(task, clean_frame, [], tokenizer, "ordinary")
            jsonl(output / f"ordinary-trajectory-{index:02d}.jsonl", clean)
            save(output / f"ordinary-audit-{index:02d}.json", clean_audit)
            audits.append(clean_audit)
            if not clean_audit["eligible"]:
                raise ValueError("ordinary A trajectory failed qualification; no replacement")
            ordinary_pool.extend(clean)
        matched, pairs = match_rows(selected, ordinary_pool)
        if len(selected) != 32 or len(matched) != 32:
            raise ValueError("declared32 rows per arm not possible")
        jsonl(output / "recovery-rows.jsonl", selected)
        jsonl(output / "ordinary-rows.jsonl", matched)
        save(output / "MATCHING.json", pairs)
        save(output / "PREFIXES.json", prefix_receipts)
        save(output / "NATIVE-AUDIT.json", audits)
        arm_stats = {}
        for arm, rows in [("recovery", selected), ("ordinary", matched)]:
            arm_stats[arm] = dict(
                rows=len(rows),
                task_ids=[t["id"] for t in tasks],
                row_ids=[r["row_id"] for r in rows],
                label_ids=[r["label_id"] for r in rows],
                action_counts=dict(Counter(action_type(r) for r in rows)),
                target_tokens=sum(r["target_tokens"] for r in rows),
                prompt_tokens=sum(r["prompt_tokens"] for r in rows),
                max_prompt_plus_cap=max(r["prompt_plus_cap"] for r in rows),
                rows_sha256=sha(output / f"{arm}-rows.jsonl"),
            )
        manifest = dict(
            schema="textcraft-recovery-teaching-data-20260929-v1",
            ready=True,
            GPU_ready=False,
            split="official TRAIN",
            dataset_group="fresh A optimization",
            world_seed=42,
            world_sha256=diagnosis["dataset"]["world_sha256"],
            source_dataset=str(DATASET),
            source_tasks_sha256=sha(DATASET / "tasks.jsonl"),
            source_dataset_manifest_sha256=sha(DATASET / "MANIFEST.json"),
            source_collection=str(COLLECTION),
            source_collection_plan_sha256=sha(COLLECTION / "PLAN.json"),
            source_collection_native_audit_sha256=sha(COLLECTION / "NATIVE-AUDIT.json"),
            future_training_start=diagnosis["warm_checkpoint"],
            approved_spec=dict(path=str(DIAGNOSIS / "EXPERIMENT.md"), sha256=EXPERIMENT_SHA),
            diagnosis=dict(path=str(DIAGNOSIS / "RESULTS.json"), sha256=DIAGNOSIS_SHA),
            selection=(
                "Each A repeat0 first native Error, before next saved response; first suffix action, "
                "first unused craft, last unused craft, finish; chronological output per task."
            ),
            matching=(
                "Frozen A order, chronological recovery rows; unused same-task/action-type clean "
                "row minimizing absolute target-token difference; tie original clean step."
            ),
            qualification=dict(
                recovery_native_successes=8,
                ordinary_native_successes=8,
                first_saved_prompt_and_input_id_equal=8,
                no_failed_task_replacement=True,
            ),
            caps=dict(
                global_calls=96,
                global_output_tokens=8192,
                response_tokens=256,
                input_plus_output=8192,
                prefix_cost_charged=True,
                original_root_initial_inventory_retained=True,
            ),
            teacher_boundary=(
                "Only copied targets, original/current inventory and past get_info is_base/recipes "
                "reach next_action. Native world used only for transition/qualification; no gold, "
                "taskID, scores, future replies or hidden native metadata in teacher inputs."
            ),
            training_fields=(
                "Use prompt/target or input_ids/labels/target_ids only; feedback and native audit "
                "are qualification metadata, not extra training input."
            ),
            tokenization=dict(
                path=str(encoder.BASE),
                model_manifest_sha256=sha(model_manifest_path),
                revision=model_manifest["huggingface_revision"],
                source=model_manifest["source"],
                license=model_manifest["license"],
                assets_sha256=tokenizer_pins,
                add_generation_prompt=True,
                enable_thinking=False,
                target_includes_eos=True,
                EOS=tokenizer.eos_token_id,
            ),
            upstream=json.loads((DATASET / "MANIFEST.json").read_text())["official_source"],
            arms=arm_stats,
            matching_pairs=32,
            identical_target_pairs=sum(p["identical_target"] for p in pairs),
            summed_absolute_target_token_difference=sum(
                p["target_token_difference"] for p in pairs
            ),
            unmatched=(
                "Different target identities, exact target token exposure, prefix lengths/FLOPs "
                "and learner-history distributions. Not a same-label-multiset or equal-token study."
            ),
            teacher_trajectory_rows=dict(recovery=len(recovery_pool), ordinary=len(ordinary_pool)),
            source_sha256={
                str(path): sha(path)
                for path in [
                    Path(__file__).resolve(),
                    Path(__file__).with_name("test_prepare.py").resolve(),
                    Path(teacher.__file__),
                    Path(encoder.__file__),
                    Path(bridge.__file__),
                    Path(worlds.__file__),
                ]
            },
            trusted_native_source=bridge.trusted_provenance(),
            environment=dict(
                python=sys.version,
                executable=sys.executable,
                platform=platform.platform(),
                packages={p: version(p) for p in ["transformers", "tokenizers", "torch", "peft"]},
                CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                uv="not invoked; existing environment unchanged",
            ),
            artifacts_sha256={
                path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()
            },
            future_fit=dict(
                updates=1,
                optimizer="fresh AdamW",
                learning_rate=2e-5,
                weight_decay=0,
                clip=1,
                seed=202609290201,
                loss="T1 full teacher target+EOS token-mean NLL; masked prompts",
                base_dtype="float16",
                lora_dtype="float32",
            ),
            B_role=(
                "No B goals, responses, outcomes or checkpoint selection used. "
                "No training/readout/GPU launch descriptors prepared."
            ),
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        save(output / "MANIFEST.json", manifest)
        return manifest
    except Exception as exc:
        save(
            output / "FAILURE.json",
            dict(
                ready=False,
                error=f"{type(exc).__name__}: {exc}",
                preserved_partial_artifacts=True,
                no_replacement=True,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    result = prepare(parser.parse_args().output)
    print(
        json.dumps(
            dict(
                ready=result["ready"],
                GPU_ready=False,
                output=str(OUTPUT),
                arms={
                    k: {
                        f: v[f] for f in ["rows", "target_tokens", "prompt_tokens", "action_counts"]
                    }
                    for k, v in result["arms"].items()
                },
                manifest_sha256=sha(OUTPUT / "MANIFEST.json"),
            )
        )
    )
