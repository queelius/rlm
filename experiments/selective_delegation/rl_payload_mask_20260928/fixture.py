"""Scripted CPU native binder requests, full replay and the actual masked loss."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import payload as m


def check(output: Path) -> dict:
    import torch
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError("preserve previous fixture attempt")
    torch.set_num_threads(4)
    world = m.p.c.bridge.load_world()
    recipe = next(
        rs[0]
        for rs in world.recipes.values()
        if rs[0].result_count > 1 and all(world.is_base_item(k) for k in rs[0].ingredients)
    )
    craft = dict(
        action="craft",
        ingredients={k: v + 1 for k, v in recipe.ingredients.items()},
        target_item=recipe.result_item,
        output_count=recipe.result_count,
    )
    emitted = [
        "not json",
        json.dumps(craft),
        json.dumps(dict(action="get_info", items=[recipe.result_item])),
        json.dumps(dict(craft, output_count=recipe.result_count + 1)),
        json.dumps(craft),
        '{"action":"finish","message":"done"}',
    ]
    answers = iter(emitted)
    tokenizer = AutoTokenizer.from_pretrained(
        m.p.c.BASE, local_files_only=True, trust_remote_code=False
    )
    scores = torch.zeros((1, len(tokenizer)))

    class Model:
        device = torch.device("cpu")

        def parameters(self):
            return iter(())

        def generate(self, **kwargs):
            tokens = tokenizer.encode(next(answers), add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            return SimpleNamespace(
                sequences=torch.cat([kwargs["input_ids"], torch.tensor([tokens])], dim=1),
                scores=[scores] * len(tokens),
            )

        def __call__(self, input_ids, use_cache, logits_to_keep):
            if use_cache:
                raise ValueError("replay cache must remain disabled")
            return SimpleNamespace(logits=scores.unsqueeze(1).expand(1, logits_to_keep, -1))

    task = dict(
        id="textcraft_synth.train.payload_fixture",
        goal="Craft the target",
        misc=dict(
            target_items={recipe.result_item: recipe.result_count},
            initial_inventory=dict(recipe.ingredients),
        ),
    )
    job = m.p.jobs(
        [task] + [dict(task, id=f"textcraft_synth.train.fixture{i}") for i in range(7)],
        "binder",
        "collect",
        1,
    )[0]
    plan = dict(
        model=str(m.p.c.BASE),
        model_manifest_sha256="scripted-cpu-fixture",
        adapter=None,
        profile="original",
        jobs=[job],
        planned_episodes=1,
        scientific_model_calls=0,
    )
    m.p.c.save(output / "PLAN.json", plan)
    model, native = Model(), m.private(m.LEGACY / "collect.py")
    client = native.MetricsClient(
        model,
        tokenizer,
        output,
        time.time() + 120,
        plan["model_manifest_sha256"],
        m.p.sha(output / "PLAN.json"),
    )
    collector, _ = m.p.implementation("binder")
    episode = collector.episode(task, job, client, world, output, time.time() + 120)
    audit = m.p.audit_collection(output, plan, [task], tokenizer, world, "binder")
    m.p.c.save(output / "NATIVE-AUDIT.json", audit)
    node = m.p.read(output / "nodes" / (job["episode_id"] + "-n0.json"))
    assists = {a["call_id"]: a for a in node["execution_assists"]}
    history, records, before, gaps = [], {}, {}, []
    calls = {}
    for cid, row in zip(node["call_ids"], node["public_history"], strict=True):
        call = m.p.read(output / "calls" / (cid + ".json"))
        calls[cid] = call
        records[cid] = m.call_mask(call, history, assists.get(cid), tokenizer)
        history.append(row)
        captured = m.p.read(output / "generation-logps" / (cid + ".json"))
        if captured["call_sha256"] != m.p.sha(output / "calls" / (cid + ".json")):
            raise ValueError("capture not bound to actual sampled IDs")
        replay = m.p.rl.action_logps(model, call)
        before[cid] = replay.flatten().tolist()
        gaps.extend(abs(a - b) for a, b in zip(before[cid], captured["logps"], strict=True))
        if cid == node["call_ids"][0]:
            m.first_replay(output, call, replay, captured["logps"])
    bits = {cid: r["mask"] for cid, r in records.items()}
    trainer = m.bound_trainer(output, bits)
    for cid, call in calls.items():
        ids, targets = m.p.rl.loss_math.causal_inputs(call)
        if ids != call["input_token_ids"] + call["output_token_ids"][:-1]:
            raise ValueError("masked payload was removed from autoregressive context")
        if targets != call["output_token_ids"]:
            raise ValueError("original sampled targets changed")
        for advantage in (2 / 3, -2 / 3):
            logps = torch.tensor([before[cid]], requires_grad=True)
            objective = trainer.masked_objective(logps, advantage, bits[cid])
            objective.backward()
            expected = [-advantage * flag / 32 for flag in bits[cid]]
            if any(
                abs(a - b) > 1e-8
                for a, b in zip(logps.grad.flatten().tolist(), expected, strict=True)
            ):
                raise ValueError("masked loss derivative or denominator changed")
    ordered = [records[cid] for cid in node["call_ids"]]
    if [r["eligible"] for r in ordered] != [False, False, False, False, True, False]:
        raise ValueError("public-history eligibility differs from declared fixture")
    if not ordered[4]["masked_tokens"] or ordered[4]["mask"][-1] != 1:
        raise ValueError("payload not masked or actual EOS loss incorrectly removed")
    if audit["successes"] != 1 or max(gaps) > 1e-5:
        raise ValueError("native score or full-token replay failed")
    result = dict(
        passed=True,
        scientific_model_calls=0,
        GPU_used=False,
        native_audit=audit["audits"],
        native_score=episode["native_score"],
        errors=episode["errors"],
        calls=len(calls),
        generation_replay_max_gap=max(gaps),
        mask_records=records,
        original_prefixes_and_all_targets_preserved=True,
        signed_loss_derivatives="Retained +/-1/48, masked0 for analytical advantages +/-2/3",
        private_trainer_binding=m.binding_receipt(),
        source_sha256=m.source_pins(),
        qualification="Offline scripted fixture chosen from a base recipe, not learned task "
        "selection or hidden information supplied to the scientific actor. Native receipts "
        "are replayed; loss signs use an explicitly analytical [1,1,0,0] reward example.",
    )
    m.p.c.save(output / "VERIFICATION.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = check(parser.parse_args().output)
    print(
        json.dumps({k: result[k] for k in ("passed", "native_score", "calls", "errors")}, indent=2)
    )
