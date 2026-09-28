"""Scripted CPU responses through real NativeClient, native episode and saved-call audit.

This is a runtime fixture, not model inference or scientific evaluation evidence.
"""

import argparse
import json
import time
from pathlib import Path

import compact_bridge as compact
import evaluate
import prepare as data

import prepare_textcraft_public as teacher


class ScriptedModel:
    device = "cpu"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.spec = None

    def parameters(self):
        return iter(())

    def generate(self, input_ids, attention_mask, generation_config):
        import torch

        prompt = self.spec["prompt"]
        if not prompt.startswith(compact.INSTRUCTION):
            raise ValueError("fixture did not receive compact instructions")
        state = json.loads(prompt[len(compact.INSTRUCTION) :])
        observed = {}
        for step in state["history"]:
            if step.get("action", {}).get("action") == "get_info" and isinstance(
                step.get("feedback"), list
            ):
                for reply in step["feedback"]:
                    observed[reply["item"]] = {
                        "is_base": reply["is_base"],
                        "recipes": reply["recipes"],
                    }
        root = next(iter(state["target_items"]))
        index = self.spec["global_call_index"]
        if index == 0:
            action = {"action": "finish", "message": "no", "extra": "reject"}
        elif index == 1:
            action = {"action": "craft", "target_item": root, "output_count": 1}
        elif index == 2:
            action = {"action": "get_info", "items": [root]}
        elif index in (3, 4, 5):
            count = observed[root]["recipes"][0]["result_count"]
            action = {"action": "craft", "target_item": root, "output_count": count}
            if index == 3:
                action["ingredients"] = {"old_format_extra": 1}
            elif index == 4:
                action["output_count"] = count + 1 if count > 1 else 0
            # index5 is divisible, but prerequisites are still absent: no stock repair.
        else:
            full = teacher.next_action(
                state["target_items"],
                state["inventory_at_task_start"],
                state["current_inventory"],
                observed,
            )
            action = compact.project_action(full)
        text = json.dumps(action, separators=(",", ":"))
        emitted = self.tokenizer.encode(text, add_special_tokens=False) + [
            self.tokenizer.eos_token_id
        ]
        if len(emitted) > generation_config.max_new_tokens:
            raise ValueError("scripted fixture output exceeds requested cap")
        return torch.tensor([input_ids[0].tolist() + emitted], dtype=torch.long)


def check(output: Path):
    if output.exists():
        raise FileExistsError("immutable runtime fixture exists")
    outcomes = []
    for seed in (42, 50):
        directory = output / f"world{seed}"
        args = argparse.Namespace(world=seed, mode="compact", hours=0.75, output=directory)
        plan, tasks, world, tokenizer, collector, auditor = evaluate.build(args, False)
        plan.update(fixture=True, model_generation=False, adapter=None)
        plan["source_sha256"][str(Path(__file__).resolve())] = data.sha(Path(__file__))
        data.save(directory / "PLAN.json", plan)
        plan_sha = data.sha(directory / "PLAN.json")
        model = ScriptedModel(tokenizer)

        class Client(collector.NativeClient):
            def call(self, spec):
                model.spec = spec
                return super().call(spec)

        client = Client(
            model, tokenizer, directory, time.time() + 60, plan["model_manifest_sha256"], plan_sha
        )
        job = plan["jobs"][0]
        task = next(t for t in tasks if t["id"] == job["task_id"])
        row = collector.episode(task, job, client, world, directory, time.time() + 60)
        calls = {p.stem: data.accepted.read(p) for p in (directory / "calls").glob("*.json")}
        nodes = {
            nid: data.accepted.read(directory / "nodes" / f"{job['episode_id']}-{nid}.json")
            for nid in row["node_ids"]
        }
        audit = auditor.audit_episode(
            task, job, row, calls, nodes, plan, plan_sha, tokenizer, world
        )
        if (
            not row["observed"]
            or row["native_score"] != 1
            or not audit["replayed"]
            or audit["errors"].get("invalid_schema", 0) < 2
            or audit["errors"].get("rejected_action", 0) < 1
            or audit["errors"].get("native_action_error", 0) < 1
        ):
            raise ValueError("actual saved native fixture did not preserve error/finish contracts")
        if audit["calls"] != len(calls) or audit["output_tokens"] != sum(
            len(call["output_token_ids"]) for call in calls.values()
        ):
            raise ValueError("rejected responses were not charged exactly")
        native = nodes["n0"]["public_history"]
        if any("ingredients" in step.get("action", {}) for step in native):
            raise ValueError("compact public action history exposed executed ingredient fields")
        summary = dict(
            world=seed,
            task_id=task["id"],
            passed=True,
            **audit,
            scripted_responses=True,
            scientific_model_calls=0,
            files_sha256={str(p): data.sha(p) for p in directory.rglob("*.json")},
        )
        data.save(directory / "AUDIT.json", summary)
        outcomes.append(summary)
    report = dict(
        schema="compact-runtime-cpu-fixture-20260928-v1",
        passed=True,
        cases=[{k: v for k, v in o.items() if k != "files_sha256"} for o in outcomes],
        gpu_used=False,
        verifier_sha256=data.sha(Path(__file__)),
        scope="Scripted CPU tensors use actual NativeClient request construction, "
        "tokenizer, decode, immutable starts/calls, compact native episode and saved "
        "receipt auditor. Rejected schema/unobserved/nondivisible/stock attempts charged.",
    )
    data.save(output / "VERIFICATION.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=data.OUTPUT / "runtime-fixture-001")
    args = parser.parse_args()
    print(json.dumps(check(args.output.resolve()), indent=2))
