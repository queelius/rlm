"""Scripted CPU compact actions through real generation capture, receipts and native replay."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import compact_common as c


def check(output: Path):
    import torch
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("preserve prior fixture attempts")
    torch.set_num_threads(4)
    _, tasks = c.f.load_dataset(c.f.DATA / "train", "collect")
    task = tasks[0]
    trace = c.read(c.f.DATA / "train/native" / (task["id"] + ".json"))["trace"]
    actions = [c.interface.compact.project_action(step["action"]) for step in trace]
    root = next(iter(task["misc"]["target_items"]))
    root_reply = next(info for info in trace[0]["feedback"] if info["item"] == root)
    count = root_reply["recipes"][0]["result_count"]
    craft = dict(action="craft", target_item=root, output_count=count)
    emitted_actions = [
        dict(craft, ingredients={"old_format": 1}),
        craft,
        actions[0],
        craft,
        *actions[1:],
    ]
    answers = iter(json.dumps(action) for action in emitted_actions)
    tokenizer = AutoTokenizer.from_pretrained(c.c.BASE, local_files_only=True)
    scores = torch.zeros((1, len(tokenizer)))

    class Model:
        device = torch.device("cpu")

        def parameters(self):
            return iter(())

        def generate(self, **kwargs):
            assert kwargs["return_dict_in_generate"] and kwargs["output_scores"]
            tokens = tokenizer.encode(next(answers), add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id)
            sequence = torch.cat([kwargs["input_ids"], torch.tensor([tokens])], dim=1)
            return SimpleNamespace(sequences=sequence, scores=[scores] * len(tokens))

        def __call__(self, input_ids, use_cache, logits_to_keep):
            assert not use_cache
            return SimpleNamespace(logits=scores.unsqueeze(1).expand(1, logits_to_keep, -1))

    model = Model()
    native = c.load_legacy("collect.py")
    job = c.jobs(tasks, "collect")[0]
    plan = dict(
        model=str(c.c.BASE),
        model_manifest_sha256="scripted-cpu-fixture",
        adapter=None,
        profile="original",
        execution_mode=c.MODE,
        jobs=[job],
        planned_episodes=1,
        fixture=True,
        scientific_model_calls=0,
    )
    c.c.save(output / "PLAN.json", plan)
    client = native.MetricsClient(
        model,
        tokenizer,
        output,
        time.time() + 120,
        plan["model_manifest_sha256"],
        c.sha(output / "PLAN.json"),
    )
    world = c.c.bridge.load_world()
    row = c.implementation(c.MODE)[0].episode(task, job, client, world, output, time.time() + 120)
    if not row["observed"]:
        raise ValueError(row["failure"])
    audit = c.audit_collection(output, plan, [task], tokenizer, world, c.MODE)
    call_paths = sorted((output / "calls").glob("*.json"))
    calls = [c.read(path) for path in call_paths]
    gaps, compact_targets, positive_grads, negative_grads = [], [], [], []
    trainer = c.load_legacy("train.py")
    for call in calls:
        captured = c.read(output / "generation-logps" / (call["call_id"] + ".json"))
        if captured["call_sha256"] != c.sha(output / "calls" / (call["call_id"] + ".json")):
            raise ValueError("generation metrics did not bind actual emitted tokens")
        replay = c.rl.action_logps(model, call).flatten().tolist()
        gaps.extend(abs(x - y) for x, y in zip(captured["logps"], replay, strict=True))
        _, targets = c.rl.loss_math.causal_inputs(call)
        action = json.loads(tokenizer.decode(targets, skip_special_tokens=True))
        if action["action"] == "craft" and "ingredients" not in action:
            compact_targets.append(action == json.loads(call["text"]))
        # Analytical group [1,1,0,0] has advantages [+2/3,+2/3,-2/3,-2/3].
        # This sign/normalization check uses saved emitted tokens; no receipt reward is altered.
        for advantage, derivatives in ((2 / 3, positive_grads), (-2 / 3, negative_grads)):
            logps = torch.tensor(captured["logps"], requires_grad=True)
            trainer.signed_objective(logps, advantage).backward()
            derivatives.extend(logps.grad.tolist())
    node = c.read(output / "nodes" / (job["episode_id"] + "-n0.json"))
    if any("ingredients" in h.get("action", {}) for h in node["public_history"]):
        raise ValueError("expanded ingredients leaked into compact requested-action history")
    result = dict(
        **audit["audits"][job["episode_id"]],
        metric_records=len(list((output / "generation-logps").glob("*.json"))),
        compact_success_targets_preserved=bool(compact_targets) and all(compact_targets),
        invalid_old_format_tokens_preserved=json.loads(calls[0]["text"]) == emitted_actions[0],
        generation_replay_gap=max(gaps),
        scientific_model_calls=0,
        GPU_used=False,
        all_original_tokens_receive_signed_credit=(
            len(positive_grads) == sum(len(call["output_token_ids"]) for call in calls)
            and all(abs(grad + 1 / 48) < 1e-8 for grad in positive_grads)
            and all(abs(grad - 1 / 48) < 1e-8 for grad in negative_grads)
        ),
        positive_token_derivative=positive_grads[0],
        negative_token_derivative=negative_grads[0],
        analytical_credit_fixture="Saved original compact/error/EOS tokens, hypothetical reward "
        "group[1,1,0,0], +/-2/3 RLOO advantages, +/-1/48 per-token derivatives. "
        "This tests the actual trainer loss; native receipt rewards remain unchanged.",
        scope="Scripted CPU output tensors; not model competence. New private compact collector, "
        "full original emitted-token capture/replay and independent saved native audit.",
    )
    c.c.save(output / "VERIFICATION.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(check(parser.parse_args().output), indent=2))
