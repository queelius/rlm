"""CPU scripted generation through the actual saved-request/metrics/native replay seam."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import transfer_common as t


def run(output: Path) -> dict:
    import torch
    from transformers import AutoTokenizer

    if (output / "FIXTURE.json").exists() or any(output.glob("*/calls")):
        raise ValueError("preserve prior CPU fixture; use a new output")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(t.f.c.BASE, local_files_only=True)
    manifest, tasks = t.f.load_dataset(t.f.DATA / "diagnostic", "readout")
    task = tasks[0]
    trace_path = t.f.DATA / "diagnostic/native" / (task["id"] + ".json")
    trace = t.f.read(trace_path)["trace"]
    report = dict(
        GPU_loaded=False,
        scientific_model_calls=0,
        scope="Scripted qualification trace, real tokenizer, MetricsClient, native "
        "decoder/transitions and independent replay. Uniform synthetic score tensors verify "
        "capture plumbing, not real model probabilities, entropy or task efficacy.",
        fixture_source_sha256=t.f.sha(Path(__file__)),
        trace_sha256=t.f.sha(trace_path),
    )
    for mode in ("raw", "binder"):
        actions = [copy.deepcopy(step["action"]) for step in trace]
        changed = next(i for i, action in enumerate(actions) if action["action"] == "craft")
        if mode == "binder":
            action = actions[changed]
            action["ingredients"] = {key: value + 1 for key, value in action["ingredients"].items()}
        answers = iter(json.dumps(action) for action in actions)

        class ScriptedModel:
            device = torch.device("cpu")
            peft_config = {"textcraft_action": {}}
            active_adapters = ["textcraft_action"]

            def __init__(self, responses):
                self.responses = responses

            def parameters(self):
                return []

            def modules(self):
                return []

            def generate(self, **kwargs):
                tokens = tokenizer.encode(next(self.responses), add_special_tokens=False)
                tokens.append(tokenizer.eos_token_id)
                scores = torch.zeros(1, len(tokenizer))
                return SimpleNamespace(
                    sequences=torch.cat([kwargs["input_ids"], torch.tensor([tokens])], dim=1),
                    scores=tuple(scores for _ in tokens),
                )

        native = t.load_collector()
        args = t.arguments(mode, mode, t.STUDY, prepare_only=True)
        plan, _ = t.build_plan(args)
        plan.update(
            fixture_only=True,
            fixture_scope=report["scope"],
            jobs=plan["jobs"][:1],
            planned_episodes=1,
        )
        directory = output / mode
        t.f.c.save(directory / "PLAN.json", plan)
        client = native.MetricsClient(
            ScriptedModel(answers),
            tokenizer,
            directory,
            time.time() + 120,
            plan["model_manifest_sha256"],
            t.f.sha(directory / "PLAN.json"),
            adapter=plan["adapter"],
        )
        collector, _ = native.p.implementation(mode)
        collector.STOP = False
        world = t.f.c.bridge.load_world()
        job = plan["jobs"][0]
        result = collector.episode(task, job, client, world, directory, time.time() + 120)
        if not result["observed"] or result["native_score"] != 1:
            raise ValueError("scripted real native fixture did not finish successfully")
        audit = native.p.audit_collection(directory, plan, [task], tokenizer, world, mode)
        t.f.c.save(directory / "NATIVE-AUDIT.json", audit)
        for index, action in enumerate(actions):
            call_id = f"{job['episode_id']}-c{index:03d}"
            call = t.f.read(directory / "calls" / (call_id + ".json"))
            metrics = t.f.read(directory / "generation-logps" / (call_id + ".json"))
            _, target = t.f.rl.loss_math.causal_inputs(call)
            if (
                json.loads(tokenizer.decode(target, skip_special_tokens=True)) != action
                or len(metrics["logps"]) != len(target)
                or len(metrics["token_entropies"]) != len(target)
                or metrics["call_sha256"] != t.f.sha(directory / "calls" / (call_id + ".json"))
            ):
                raise ValueError("original sampled target or capture receipt changed")
        first = t.f.read(directory / "calls" / f"{job['episode_id']}-c000.json")
        prompt_matches = (
            hashlib.sha256(first["request"]["prompt"].encode()).hexdigest()
            == manifest["audits"][0]["initial_prompt_sha256"]
        )
        if not prompt_matches:
            raise ValueError("initial student-visible prompt differs from frozen B")
        report[mode] = dict(
            native_success=audit["successes"],
            calls=audit["physical_cost"]["calls"],
            output_tokens=audit["physical_cost"]["completion_tokens"],
            first_prompt_matches_qualified_b=prompt_matches,
            bad_requested_ingredients_preserved=mode == "binder",
            original_token_capture_complete=True,
            native_audit_sha256=t.f.sha(directory / "NATIVE-AUDIT.json"),
            plan_sha256=t.f.sha(directory / "PLAN.json"),
        )
    t.f.persist(output / "FIXTURE.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
