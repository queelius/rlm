"""CPU scripted Phi stop IDs through real request/decode/native-episode/audit paths."""

import json
import time
from pathlib import Path

import acquire
import native_adapter
import qualify

import prepare_textcraft_public as teacher


class ScriptedModel:
    device = "cpu"

    def __init__(self, tokenizer, native):
        self.tokenizer, self.native = tokenizer, native
        self.spec = None
        self.injected = False

    def parameters(self):
        return iter(())

    def generate(self, input_ids, attention_mask, generation_config):
        import torch

        if generation_config.eos_token_id != qualify.STOP_IDS:
            raise ValueError("actual generation did not receive Phi's native stop IDs")
        state = json.loads(self.spec["prompt"][len(self.native.INSTRUCTION) :])
        observed = {}
        for row in state["history"]:
            if row.get("action", {}).get("action") == "get_info" and isinstance(
                row.get("feedback"), list
            ):
                for info in row["feedback"]:
                    observed[info["item"]] = dict(is_base=info["is_base"], recipes=info["recipes"])
        action = teacher.next_action(
            state["target_items"],
            state["inventory_at_task_start"],
            state["current_inventory"],
            observed,
        )
        if action["action"] == "craft" and not self.injected:
            action = {**action, "ingredients": {"unused_field": 1}}
            self.injected = True
        stop = qualify.STOP_IDS[self.spec["global_call_index"] % 2]
        emitted = self.tokenizer.encode(
            json.dumps(action, separators=(",", ":")), add_special_tokens=False
        ) + [stop]
        if len(emitted) > generation_config.max_new_tokens:
            raise ValueError("scripted output cap exceeded")
        return torch.tensor([input_ids[0].tolist() + emitted], dtype=torch.long)


def check():
    from transformers import AutoTokenizer

    output = acquire.OUTPUT / "native-fixture-001"
    if output.exists():
        raise FileExistsError("immutable Phi native fixture exists")
    qualification = acquire.read(acquire.OUTPUT / "QUALIFICATION.json")
    prepared = Path(qualification["fixed_panels"]["42"]["prepared"])
    task = json.loads((prepared / "tasks.jsonl").read_text().splitlines()[0])
    tokenizer = AutoTokenizer.from_pretrained(
        acquire.MODEL, local_files_only=True, trust_remote_code=False
    )
    cases = []
    for assistance in ("raw", "binder"):
        directory = output / assistance
        collector, auditor, binding = native_adapter.implementation(assistance)
        world = collector.bridge.load_world()
        job = dict(
            episode_id="phi-fixture-" + assistance,
            task_id=task["id"],
            policy="flat",
            repeat=0,
            seed=2026092204,
        )
        plan = dict(
            fixture=True,
            model_generation=False,
            jobs=[job],
            profile="original",
            model=str(acquire.MODEL),
            model_manifest_sha256=acquire.sha(acquire.MODEL / "local-research-manifest.json"),
            adapter=None,
            native_stop_binding=binding,
            source_sha256={
                **binding["source_sha256"],
                str(Path(__file__).resolve()): acquire.sha(Path(__file__)),
            },
        )
        acquire.save(directory / "PLAN.json", plan)
        plan_sha = acquire.sha(directory / "PLAN.json")

        class Client(collector.NativeClient):
            def call(self, spec):
                self.model.spec = spec
                return super().call(spec)

        model = ScriptedModel(tokenizer, collector.bridge)
        client = Client(
            model, tokenizer, directory, time.time() + 90, plan["model_manifest_sha256"], plan_sha
        )
        row = collector.episode(task, job, client, world, directory, time.time() + 90)
        calls = {p.stem: acquire.read(p) for p in (directory / "calls").glob("*.json")}
        nodes = {
            nid: acquire.read(directory / "nodes" / f"{job['episode_id']}-{nid}.json")
            for nid in row["node_ids"]
        }
        audit = auditor.audit_episode(
            task, job, row, calls, nodes, plan, plan_sha, tokenizer, world
        )
        if not row["observed"] or row["native_score"] != 1 or not audit["replayed"]:
            raise ValueError("Phi native saved-response fixture did not replay to score1")
        if {call["output_token_ids"][-1] for call in calls.values()} != set(qualify.STOP_IDS) or (
            any(call["finish_reason"] != "eos" for call in calls.values())
        ):
            raise ValueError("both native stop tokens must be recorded as EOS")
        expected_errors = 1 if assistance == "raw" else 0
        if audit["errors"].get("native_action_error", 0) != expected_errors:
            raise ValueError("observed-recipe assistance/native error contract changed")
        if audit["output_tokens"] != sum(len(c["output_token_ids"]) for c in calls.values()):
            raise ValueError("actual native output-token charges differ")
        result = dict(
            assistance=assistance,
            **audit,
            native_stop_ids=qualify.STOP_IDS,
            binding=binding,
            saved_calls_sha256={str(p): acquire.sha(p) for p in directory.rglob("*.json")},
        )
        acquire.save(directory / "AUDIT.json", result)
        cases.append({k: v for k, v in result.items() if k != "saved_calls_sha256"})
    report = dict(
        schema="phi-native-scripted-cpu-fixture-20260928-v1",
        passed=True,
        gpu_used=False,
        scientific_model_calls=0,
        cases=cases,
        verifier_sha256=acquire.sha(Path(__file__)),
        scope="Scripted CPU tensors through actual request/generation config/tokenizer/decode, "
        "native TextCraft episode/score and saved-call audit. Both Phi stop IDs exercised; "
        "raw versus observed-recipe binder craft-error behavior checked. No model-quality claim.",
    )
    acquire.save(output / "VERIFICATION.json", report)
    return report


if __name__ == "__main__":
    result = check()
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "cases": [
                    {
                        k: case[k]
                        for k in ("assistance", "native_score", "calls", "output_tokens", "errors")
                    }
                    for case in result["cases"]
                ],
            },
            indent=2,
        )
    )
