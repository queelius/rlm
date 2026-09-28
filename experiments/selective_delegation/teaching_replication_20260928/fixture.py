"""CPU saved-request/native-replay fixture for the replication's unchanged raw interface."""

import argparse
import hashlib
import json
from pathlib import Path

import replication as study


def run(root: Path) -> dict:
    old = study.original_module()
    collector, _, worlds = old.load_runtime("raw")
    import analyze_textcraft_profiles as profiles
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        collector.BASE, local_files_only=True, trust_remote_code=False
    )
    cases, hashes = [], {}

    def read(path):
        hashes[str(path)] = study.sha(path)
        return study.read(path)

    for mode in study.MODES:
        directory = study.ORIGINAL / f"eval-{mode}-s2026092208-p00-w42-raw"
        plan = read(directory / "PLAN.json")
        job = plan["jobs"][0]
        episode = read(directory / "episodes" / f"{job['episode_id']}.json")
        tasks = [
            json.loads(line)
            for line in (Path(plan["prepared"]) / "tasks.jsonl").read_text().splitlines()
        ]
        task = next(t for t in tasks if t["id"] == job["task_id"])
        calls = {cid: read(directory / "calls" / f"{cid}.json") for cid in episode["call_ids"]}
        nodes = {
            nid: read(directory / "nodes" / f"{job['episode_id']}-{nid}.json")
            for nid in episode["node_ids"]
        }
        audit = profiles.audit.audit_episode(
            task,
            job,
            episode,
            calls,
            nodes,
            plan,
            study.sha(directory / "PLAN.json"),
            tokenizer,
            worlds.world(42),
        )
        saved = read(directory / "NATIVE-AUDIT.json")["audits"][job["episode_id"]]
        study.require(audit == saved and audit["replayed"], "actual native replay differs")
        first = calls[episode["call_ids"][0]]
        cases.append(
            dict(
                mode=mode,
                episode_id=job["episode_id"],
                actual_saved_calls=len(calls),
                native_score=audit["native_score"],
                output_tokens=audit["output_tokens"],
                request_digest=first["request_digest"],
                first_response_sha256=hashlib.sha256(first["text"].encode()).hexdigest(),
            )
        )
    prompt_cases = []
    for cell in study.schedule():
        if cell["seed"] != 2026092208:
            continue
        _, world, output, plan, tasks, _ = study.build(cell, root)
        reference = study.baseline(cell["seed"], cell["panel"], cell["world"], "discovery")
        job = plan["jobs"][0]
        task = next(t for t in tasks if t["id"] == job["task_id"])
        frame = collector.bridge.Frame(
            world,
            dict(task["misc"]["initial_inventory"]),
            task["misc"]["target_items"],
            collector.bridge.Budget(96, 8192),
            max_depth=0,
        )
        prompt = collector.render_prompt(frame, [], goal=task["goal"], profile="original")
        call = read(reference / "calls" / f"{job['episode_id']}-c000.json")
        study.require(prompt == call["request"]["prompt"], "new readout changes public prompt")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        study.require(ids == call["input_token_ids"], "new readout changes request tokenization")
        prompt_cases.append(dict(**cell, output=str(output), exact_saved_prompt_and_ids=True))
    blocked = []
    for mode in study.MODES:
        try:
            study.endpoint(mode, 2026092291, root)
        except FileNotFoundError:
            blocked.append(mode)
        else:
            raise ValueError("future endpoint unexpectedly present during CPU preparation")
    result = dict(
        passed=True,
        saved_native_replays=cases,
        new_panel_prompt_cases=prompt_cases,
        pending_second_seed_endpoints_rejected=blocked,
        hashes=hashes,
        source_sha256={str(p): study.sha(p) for p in (Path(__file__), Path(study.__file__))},
        scope="Two actual saved raw episodes replay request binding, response decoding and native "
        "scoring; four new panel01 initial requests exactly match saved baseline prompt/token IDs. "
        "No model response was generated; future trained actors remain unqualified until real "
        "checkpoint and owner receipts exist.",
    )
    study.save(root / "CPU-FIXTURE.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=study.OUTPUT)
    result = run(parser.parse_args().root)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("hashes", "source_sha256")}, indent=2
        )
    )
