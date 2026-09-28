"""Scripted CPU requests exercise real client/decode, native helper and saved audit."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prepare_data as data  # noqa: E402
import prepare_textcraft_public as teacher  # noqa: E402
import routing  # noqa: E402
import run  # noqa: E402


class ScriptedModel:
    device = "cpu"

    def __init__(self, tokenizer, refusal=False, inject_errors=False):
        self.tokenizer, self.refusal, self.inject_errors = tokenizer, refusal, inject_errors
        self.spec = None
        self.child_schema_error = self.child_stock_error = False

    def parameters(self):
        return iter(())

    def generate(self, input_ids, attention_mask, generation_config):
        import torch

        prompt = self.spec["prompt"]
        state, _ = json.JSONDecoder().raw_decode(prompt[len(routing.native.INSTRUCTION) :])
        control = json.loads(prompt.rsplit(routing.MARKER, 1)[1])
        required = control["required_next_action"]
        observed = {}
        for step in state["history"]:
            if step.get("action", {}).get("action") == "get_info" and isinstance(
                step.get("feedback"), list
            ):
                for info in step["feedback"]:
                    observed[info["item"]] = dict(is_base=info["is_base"], recipes=info["recipes"])
        root = next(iter(state["target_items"]))
        if self.refusal and required is not None and required["action"] == "delegate":
            action = {"action": "view_inventory"}
        elif self.inject_errors and self.spec["global_call_index"] == 0:
            action = {"action": "view_inventory", "extra": "reject"}
        elif self.inject_errors and state["agent_depth"] > 0 and not self.child_schema_error:
            self.child_schema_error = True
            action = {"action": "view_inventory", "extra": "child schema rejection"}
        elif (
            self.inject_errors
            and state["agent_depth"] > 0
            and root in observed
            and (not self.child_stock_error)
        ):
            self.child_stock_error = True
            recipe = observed[root]["recipes"][0]
            action = dict(
                action="craft",
                target_item=root,
                output_count=100 * recipe["result_count"],
                ingredients={k: 100 * v for k, v in recipe["ingredients"].items()},
            )
        elif required is not None:
            action = required
        else:
            action = teacher.next_action(
                state["target_items"],
                state["inventory_at_task_start"],
                state["current_inventory"],
                observed,
            )
        emitted = self.tokenizer.encode(
            json.dumps(action, separators=(",", ":")), add_special_tokens=False
        ) + [self.tokenizer.eos_token_id]
        if len(emitted) > generation_config.max_new_tokens:
            raise ValueError("fixture output exceeds real native response cap")
        return torch.tensor([input_ids[0].tolist() + emitted], dtype=torch.long)


def check(output):
    if output.exists():
        raise FileExistsError("immutable runtime fixture exists")
    cases = []
    for mode, special in (
        ("flat", "ordinary"),
        ("fixed", "errors"),
        ("adaptive", "ordinary"),
        ("fixed", "refusal"),
        ("fixed", "cap"),
    ):
        directory = output / f"{mode}-{special}"
        args = argparse.Namespace(mode=mode, hours=0.25, output=directory)
        plan, tasks, world, tokenizer, collector, _ = run.build(args, fixture=True)
        plan["source_sha256"][str(Path(__file__).resolve())] = data.sha(Path(__file__))
        if special == "cap":
            plan.update(admission_response_cap=3, max_native_calls=3)
        plan["scripted_case"] = special
        data.save(directory / "PLAN.json", plan)

        class Client(collector.NativeClient):
            def call(self, spec):
                self.model.spec = spec
                return super().call(spec)

        model = ScriptedModel(tokenizer, special == "refusal", special == "errors")
        client = Client(
            model,
            tokenizer,
            directory,
            time.time() + 90,
            plan["model_manifest_sha256"],
            data.sha(directory / "PLAN.json"),
        )
        client.response_cap = plan["admission_response_cap"]
        row = collector.episode(
            tasks[0], plan["jobs"][0], client, world, directory, time.time() + 90
        )
        result = run.audit(directory, require_terminal=False)
        if result["failed"] or row["global_calls"] != client.returned:
            raise ValueError("fixture response/attempt accounting mismatch")
        if special in ("cap", "refusal"):
            if (
                row["observed"]
                or row["native_score"] is not None
                or ("AdmissionStop:" not in row["failure"])
            ):
                raise ValueError("planned stop must preserve unknown native root outcome")
            if special == "cap" and result["returned"] != 3:
                raise ValueError("stop must occur before creating a fourth request")
            if special == "refusal" and (
                result["errors"].get("rejected_action") != 2 or result["child_nodes"] != 0
            ):
                raise ValueError("two rejected delegation instructions must stop the arm")
        elif (
            not row["observed"]
            or row["native_score"] != 1
            or (not result["native_audit"]["replayed"])
        ):
            raise ValueError("native completed fixture must fully replay to root score1")
        if special == "errors" and (
            result["child_nodes"] != 1
            or result["errors"].get("invalid_schema") != 2
            or result["errors"].get("native_action_error") != 1
        ):
            raise ValueError("root and child rejected/error responses must be charged unchanged")
        data.save(directory / "AUDIT.json", result)
        cases.append({k: v for k, v in result.items() if k != "artifacts_sha256"})
    report = dict(
        schema="textcraft-decomposition-cpu-fixture-20260928-v1",
        passed=True,
        scientific_model_calls=0,
        gpu_used=False,
        cases=cases,
        verifier_sha256=data.sha(Path(__file__)),
        scope="Actual NativeClient requests, tokenizer, scripted CPU tensors, response decode, "
        "saved starts/calls, unchanged native episode/helper/score and independent audit. "
        "Three policy cases plus two-refusal and pre-request-cap cases; not GPU model evidence.",
    )
    data.save(output / "VERIFICATION.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=data.OUTPUT / "runtime-fixture-001")
    args = parser.parse_args()
    print(json.dumps(check(args.output.resolve()), indent=2))
