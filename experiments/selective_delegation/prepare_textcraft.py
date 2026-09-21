"""Freeze the official eight-task pilot before trusted native replay; never replace failures."""

import argparse
import hashlib
import json
import time
from pathlib import Path

CACHE = Path(
    "/project/alex_phd/research-cache/repos/platoon-rao-d9c5857d3a0a056ebc9b047241a2a0c9515aafbe"
)
COMMIT = "d9c5857d3a0a056ebc9b047241a2a0c9515aafbe"
SEED = 2026092203
TASKS = CACHE / "plugins/textcraft/platoon/textcraft/textcraft_synth_val.jsonl"
TASK_SHA = "84a123ee46e29e65f4d7b95f4943aa56907fc3fd5992268faa749602579dfc0c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def select(rows):
    def key(row):
        return hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()

    shallow = sorted([r for r in rows if r["misc"]["max_depth"] in (2, 3)], key=key)[:4]
    longer = sorted([r for r in rows if r["misc"]["max_depth"] == 4], key=key)[:4]
    if len(shallow) != 4 or len(longer) != 4:
        raise ValueError("insufficient fixed strata; no replacement")
    return shallow + longer


def freeze(output):
    if output.exists():
        raise FileExistsError("immutable selection exists")
    if sha(TASKS) != TASK_SHA:
        raise ValueError("official task inventory changed")
    rows = select([json.loads(line) for line in TASKS.read_text().splitlines()])
    output.mkdir(parents=True)
    with (output / "tasks.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(
        output / "SELECTION.json",
        {
            "schema": "textcraft-eight-frozen-selection-v1",
            "selected_before_replay": True,
            "selected_at": time.time(),
            "selection_seed": SEED,
            "selection": "four depth2-3 plus four depth4; SHA256(seed:id), "
            "no gold/performance filter",
            "task_ids": [r["id"] for r in rows],
            "task_count": 8,
            "upstream_commit": COMMIT,
            "upstream_tasks": str(TASKS),
            "upstream_tasks_sha256": TASK_SHA,
            "tasks_sha256": sha(output / "tasks.jsonl"),
            "preparer_sha256": sha(Path(__file__)),
            "max_global_calls": 96,
            "max_global_output_tokens": 8192,
            "max_new_tokens": 256,
            "max_agent_depth": 2,
            "root_depth": 0,
            "policies": ["flat", "recursive"],
            "repeats": 2,
            "planned_episodes": 32,
            "max_native_calls": 3072,
            "model_seconds_cap": 3600,
            "recipe_seed": 42,
            "items_per_domain_tier": 25,
            "provenance_caveat": "Official VAL root-target disjoint from TRAIN, "
            "shared recipe world; "
            "not unseen recipes/pretraining and not a RAO reproduction.",
        },
    )
    return rows


def replay(output):
    import textcraft_bridge as bridge

    selection = json.loads((output / "SELECTION.json").read_text())
    if sha(output / "tasks.jsonl") != selection["tasks_sha256"]:
        raise ValueError("frozen task bytes changed")
    if (output / "REPLAY.json").exists():
        raise FileExistsError("replay receipt already exists")
    rows = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    world = bridge.load_world()
    started = time.time()
    results = [bridge.replay_gold(row, world) for row in rows]
    receipt = {
        "selection_sha256": sha(output / "SELECTION.json"),
        "replay_started": started,
        "replay_ended": time.time(),
        "tasks": results,
        "all_eight_native_success": all(r["native_score"] == 1 for r in results),
        "no_replacement": True,
        "trusted_source": bridge.trusted_provenance(),
        "bridge_sha256": sha(Path(bridge.__file__)),
        "lifecycle": "Exact hash-pinned upstream action/evaluate method bodies; "
        "lightweight explicit state, not full Platoon/IPython lifecycle.",
    }
    save(output / "REPLAY.json", receipt)
    return receipt


def token_audit(output):
    import textcraft_bridge as bridge
    from transformers import AutoTokenizer

    selection = json.loads((output / "SELECTION.json").read_text())
    if sha(output / "tasks.jsonl") != selection["tasks_sha256"]:
        raise ValueError("frozen tasks changed")
    if (output / "TOKEN-AUDIT.json").exists():
        raise FileExistsError("immutable token audit exists")
    base = Path(
        "/project/alex_phd/research-cache/models/"
        "Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)

    def prompt_length(text):
        return len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    tasks = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    world, results = bridge.load_world(), []
    for task in tasks:
        lengths = {p: prompt_length(bridge.initial_prompt(task, p)) for p in ("flat", "recursive")}
        frame = bridge.Frame(
            world,
            dict(task["misc"]["initial_inventory"]),
            task["misc"]["target_items"],
            bridge.Budget(),
            max_depth=0,
        )
        history, action_tokens, prompt_tokens = [], [], []
        actions = []
        for step in task["misc"]["gold_trajectory"]:
            actions.extend(
                [
                    {"action": "get_info", "items": [step["target"][0]]},
                    {
                        "action": "craft",
                        "ingredients": step["ingredients"],
                        "target_item": step["target"][0],
                        "output_count": step["result_count"],
                    },
                ]
            )
        actions.append({"action": "finish", "message": "done"})
        for action in actions:
            prompt_tokens.append(
                prompt_length(bridge.public_prompt(frame, history, goal=task["goal"]))
            )
            raw = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            count = (
                len(tokenizer.encode(raw, add_special_tokens=False)) + 1
            )  # emitted EOS allowance
            action_tokens.append(count)
            frame.budget.charge(count)
            feedback = frame.apply(action)
            history.append({"action": action, "feedback": feedback})
        results.append(
            {
                "task_id": task["id"],
                "initial_prompt_tokens": lengths,
                "gold_public_history_max_prompt_tokens": max(prompt_tokens),
                "gold_max_action_tokens_with_eos": max(action_tokens),
                "gold_global_calls_including_info_and_finish": frame.budget.calls,
                "gold_global_output_tokens_with_eos": frame.budget.output_tokens,
                "native_score_after_info_craft_finish": frame.score()[0],
            }
        )
    world_json = {
        item: [
            {"ingredients": r.ingredients, "result_count": r.result_count, "depth": r.depth}
            for r in recipes
        ]
        for item, recipes in sorted(world.recipes.items())
    }
    receipt = {
        "selection_sha256": sha(output / "SELECTION.json"),
        "replay_sha256": sha(output / "REPLAY.json"),
        "tasks": results,
        "model": str(base),
        "model_manifest_sha256": sha(base / "local-research-manifest.json"),
        "bridge_sha256": sha(Path(bridge.__file__)),
        "preparer_sha256": sha(Path(__file__)),
        "trusted_source": bridge.trusted_provenance(),
        "world_recipe_count": len(world.recipes),
        "world_sha256": hashlib.sha256(json.dumps(world_json, sort_keys=True).encode()).hexdigest(),
        "all_frozen_tasks_native_success": all(
            r["native_score_after_info_craft_finish"] == 1 for r in results
        ),
        "max_new_tokens": 256,
        "input_plus_output_limit": 8192,
        "no_truncation": True,
        "ready": all(
            r["gold_public_history_max_prompt_tokens"] + 256 <= 8192
            and r["gold_max_action_tokens_with_eos"] <= 256
            for r in results
        ),
        "interpretation": "CPU constructive trace audit only, not model outcomes or minimum cost. "
        "Arbitrary policy contexts may still reach the same enforced limits; "
        "retain those outcomes.",
    }
    save(output / "TOKEN-AUDIT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("freeze", "replay", "token-audit"), required=True)
    args = parser.parse_args()
    result = {"freeze": freeze, "replay": replay, "token-audit": token_audit}[args.stage](
        args.output.resolve()
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "output": str(args.output),
                "count": len(result) if isinstance(result, list) else len(result["tasks"]),
            }
        )
    )
