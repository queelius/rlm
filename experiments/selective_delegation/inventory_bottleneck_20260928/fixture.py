"""Replay a real saved deep episode through NativeClient and both table renderings on CPU."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import demand_table as table
import readout


class SavedResponseModel:
    device = "cpu"

    def __init__(self, calls):
        self.calls, self.spec = calls, None
        self.token_lengths = []

    def parameters(self):
        return iter(())

    def generate(self, input_ids, attention_mask, generation_config):
        import torch

        collector = readout.runtime()[0]
        prompt = self.spec["prompt"]
        payload = json.loads(prompt[len(collector.bridge.INSTRUCTION) :])
        original = {
            k: v
            for k, v in payload.items()
            if k
            not in ("quantity_table_description", "public_quantity_table", "matched_length_padding")
        }
        _, lengths = table.render_pair(collector.bridge.INSTRUCTION, original, readout.tokenizer())
        readout.require(lengths["demand"] == lengths["masked"], "actual tokenizer lengths differ")
        self.token_lengths.append(lengths)
        previous = self.calls[self.spec["global_call_index"]]
        previous_payload = json.loads(
            previous["request"]["prompt"][len(collector.bridge.INSTRUCTION) :]
        )
        readout.require(original == previous_payload, "saved public input/history changed")
        emitted = previous["output_token_ids"]
        readout.require(
            len(emitted) <= generation_config.max_new_tokens, "saved output cap differs"
        )
        return torch.tensor([input_ids[0].tolist() + emitted], dtype=torch.long)


def check(output):
    readout.require(not output.exists(), "immutable CPU fixture already exists")
    original_eid = "t05-r1-flat-original"
    old_audit = readout.read(readout.BASELINE / "NATIVE-AUDIT.json")
    old_node_path = readout.BASELINE / "nodes" / f"{original_eid}-n0.json"
    readout.require(
        readout.sha(old_node_path) == old_audit["sha256"][str(old_node_path)], "node changed"
    )
    old_node = readout.read(old_node_path)
    saved = []
    for cid in old_node["call_ids"]:
        path = readout.BASELINE / "calls" / f"{cid}.json"
        readout.require(
            readout.sha(path) == old_audit["sha256"][str(path)], "saved response changed"
        )
        saved.append(readout.read(path))
    results = []
    for mode in table.MODES:
        plan, tasks, _, world, _ = readout.build(mode, persist=False)
        plan = copy.deepcopy(plan)
        plan.update(fixture=True, scientific_model_generation=False, adapter=None)
        plan["source_sha256"][str(Path(__file__).resolve())] = readout.sha(Path(__file__))
        directory = output / mode
        readout.save(directory / "PLAN.json", plan)
        collector, auditor, _, _ = readout.runtime()
        model = SavedResponseModel(saved)

        class Client(collector.NativeClient):
            def call(self, spec):
                self.model.spec = spec
                return super().call(spec)

        client = Client(
            model,
            readout.tokenizer(),
            directory,
            time.time() + 180,
            plan["model_manifest_sha256"],
            readout.sha(directory / "PLAN.json"),
        )
        job = next(j for j in plan["jobs"] if j["episode_id"] == original_eid)
        task = next(t for t in tasks if t["id"] == job["task_id"])
        with (
            readout.correct_summary_counts(collector),
            table.installed(collector.bridge, mode, readout.tokenizer()),
        ):
            row = collector.episode(task, job, client, world, directory, time.time() + 180)
            calls = {p.stem: readout.read(p) for p in (directory / "calls").glob("*.json")}
            nodes = {
                nid: readout.read(directory / "nodes" / f"{job['episode_id']}-{nid}.json")
                for nid in row["node_ids"]
            }
            audit = auditor.audit_episode(
                task,
                job,
                row,
                calls,
                nodes,
                plan,
                readout.sha(directory / "PLAN.json"),
                readout.tokenizer(),
                world,
            )
            summary = collector.summarize(directory, plan)
        readout.require(
            row["observed"] and audit["replayed"] and row["global_calls"] == len(saved) == 34,
            "saved episode was not fully replayed",
        )
        readout.require(
            nodes["n0"]["public_history"] == old_node["public_history"]
            and nodes["n0"]["final_inventory"] == old_node["final_inventory"]
            and row["native_score"] == old_node["native_score"] == 0,
            "display changed native semantics or score",
        )
        readout.require(
            all(
                g["planned"] == 8 and g["missing_or_unknown"] == 7
                for g in summary["groups"].values()
            ),
            "historical collector mislabeled eight-slot pilot",
        )
        result = dict(
            mode=mode,
            task_id=task["id"],
            passed=True,
            saved_response_source=str(old_node_path),
            scientific_model_calls=0,
            cpu_only=True,
            native_audit=audit,
            matched_input_lengths=model.token_lengths,
            sha256={str(p): readout.sha(p) for p in directory.rglob("*.json")},
        )
        readout.save(directory / "AUDIT.json", result)
        results.append({k: v for k, v in result.items() if k != "sha256"})
    report = dict(
        schema="public-demand-saved-request-fixture-20260928-v1",
        passed=True,
        cases=results,
        scope="Actual native saved-response tensors through NativeClient request construction, "
        "tokenizer, decode, immutable starts/calls and independent native replay. Model inference "
        "is replaced by responses from one previously audited episode; "
        "not new scientific evidence.",
        source_sha256=readout.sha(Path(__file__)),
    )
    readout.save(output / "VERIFICATION.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=readout.OUTPUT / "runtime-fixture-001")
    args = parser.parse_args()
    result = check(args.output)
    print(json.dumps(dict(passed=result["passed"], cases=len(result["cases"]))))
