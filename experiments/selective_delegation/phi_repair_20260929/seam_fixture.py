"""Actual native request/response and replay under the new private Phi repair PLAN."""

import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import patch

import common as c


def check():
    evaluate, _, _, qualify = c.phi()
    path = c.E / "phi_transfer_20260928/fixture.py"
    spec = importlib.util.spec_from_file_location("phi_repair_original_scripted_fixture", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    plan, tasks, world, tokenizer, collector, auditor = c.build_readout(42)
    output = c.ROOT / "seam-fixture-001"
    c.require(not output.exists(), "immutable fixture already exists")
    job, task = plan["jobs"][0], tasks[0]
    plan.update(
        fixture=True,
        model_generation=False,
        jobs=[job],
        planned_episodes=1,
        explanation="Scripted CPU model only; no repaired Phi endpoint exists yet.",
    )
    c.save(output / "PLAN.json", plan)
    plan_sha = c.sha(output / "PLAN.json")

    class Client(collector.NativeClient):
        def call(self, request):
            self.model.spec = request
            return super().call(request)

    model = old.ScriptedModel(tokenizer, collector.bridge)
    client = Client(
        model, tokenizer, output, time.time() + 90, plan["model_manifest_sha256"], plan_sha
    )
    row = collector.episode(task, job, client, world, output, time.time() + 90)
    calls = {p.stem: c.read(p) for p in (output / "calls").glob("*.json")}
    nodes = {
        nid: c.read(output / "nodes" / f"{job['episode_id']}-{nid}.json") for nid in row["node_ids"]
    }
    checked = auditor.audit_episode(task, job, row, calls, nodes, plan, plan_sha, tokenizer, world)
    c.require(checked["replayed"] and row["native_score"] == 1, "native fixture failed")
    c.require(
        {call["output_token_ids"][-1] for call in calls.values()} == set(qualify.STOP_IDS),
        "both native stop IDs must be exercised",
    )
    c.require(
        checked["errors"].get("native_action_error") == 1,
        "raw invalid craft was incorrectly assisted",
    )
    # Exercise the actual production dispatch function, with only owner launch replaced.
    dispatch_plan = c.read(c.readout_output(42) / "PENDING-PLAN.json")
    dispatch_plan.update(adapter={"fixture_only": True}, endpoint_pending=False)
    seen = []

    def runtime(args, **kwargs):
        c.require(
            args.prepare_only is False and args.teacher == "stable_visible",
            "actual collector namespace incomplete",
        )
        c.require(
            kwargs["prepared_run"][0]["adapter"] == {"fixture_only": True},
            "dispatch lost actual endpoint binding",
        )
        seen.append(args.world)

    def save_fixture_only(path, value):
        c.require(Path(path) == c.readout_output(42) / "PLAN.json", "wrong dispatch output")
        c.require(value == dispatch_plan, "dispatch plan differs")

    with (
        patch.object(
            c,
            "build_readout",
            return_value=(dispatch_plan, tasks, world, tokenizer, collector, auditor),
        ),
        patch.object(c, "save", save_fixture_only),
        patch.object(collector, "run", runtime),
    ):
        c.run_readout(42)
    c.require(seen == [42], "production dispatch not exercised")
    # Independently replay one real, already completed Phi episode, without changing its PLAN.
    saved = c.R / "textcraft-phi-discovery-raw-w42-20260928-001"
    real_plan = c.read(saved / "PLAN.json")
    real_job = real_plan["jobs"][0]
    real_row = c.read(saved / "episodes" / f"{real_job['episode_id']}.json")
    real_calls = {cid: c.read(saved / "calls" / f"{cid}.json") for cid in real_row["call_ids"]}
    real_nodes = {
        nid: c.read(saved / "nodes" / f"{real_job['episode_id']}-{nid}.json")
        for nid in real_row["node_ids"]
    }
    real_checked = auditor.audit_episode(
        task,
        real_job,
        real_row,
        real_calls,
        real_nodes,
        real_plan,
        c.sha(saved / "PLAN.json"),
        tokenizer,
        world,
    )
    c.require(
        real_checked["replayed"] and real_checked["native_score"] == real_row["native_score"],
        "actual saved model-response replay differs",
    )
    result = dict(
        schema="phi-repair-native-seam-fixture-20260929-v1",
        passed=True,
        gpu_used=False,
        scientific_model_calls=0,
        scripted_native=checked,
        real_saved_episode=dict(
            source=str(saved), episode_id=real_job["episode_id"], audit=real_checked
        ),
        production_dispatch_prepare_only_false=True,
        pending_plan_roundtrip=(c.canonical(plan) == c.read(output / "PLAN.json")),
        stop_ids=qualify.STOP_IDS,
        source_sha256={
            str(p): c.sha(p)
            for p in (Path(__file__), c.HERE / "common.py", c.HERE / "experiment.py", path)
        },
        caveat="Native CPU seam evidence only; fixture dispatch substitutes owner launch and "
        "never writes an actual readout PLAN or claims a repaired Phi model response.",
    )
    c.save(output / "VERIFICATION.json", result)
    return result


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, indent=2))
