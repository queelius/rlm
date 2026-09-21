"""CPU-only, outcome-blind balanced unseen-game selection and public reset receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

DATA = Path("/project/alex_phd/research-cache/datasets/alfworld-text-0.4.2-20260921")
MANIFEST = DATA / "readiness/MANIFEST.json"
MANIFEST_SHA = "7c0a86472377cd73af66585ec04dff4010c75de3a2346e9f2988a86840eb76c5"
ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
BRIDGE = ROOT / "source-026-alfworld-closed-loop/alfworld_bridge.py"
FAMILIES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)
SEED = 2026092194


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(inventory):
    def family(path):
        return Path(path).parent.parent.name.split("-")[0]

    def key(path):
        relative = str(Path(path).relative_to(DATA))
        return hashlib.sha256(f"{SEED}:{relative}".encode()).hexdigest()

    chosen = []
    for task in FAMILIES:
        eligible = [p for p in inventory if family(p) == task]
        if len(eligible) < 2:
            raise ValueError("fewer than two official unseen games in family: " + task)
        chosen.extend(sorted(eligible, key=key)[:2])
    return chosen


def prepare(output):
    if output.exists():
        raise FileExistsError("immutable panel output exists")
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("official inspected readiness inventory changed")
    source = json.loads(MANIFEST.read_text())
    inventory = source["official_filtered_game_inventories"]["eval_out_of_distribution"]
    selected = select(inventory)
    previous = {g["game"] for g in source["proposed_screen"]["games"]}
    previous.update(t["game"] for t in source["manual_traces"])
    if set(selected) & previous or any("/valid_unseen/" not in p for p in selected):
        raise ValueError("selected game is not a new official unseen slot")
    env = Path(source["environment"]["path"])
    python = env / ".venv/bin/python"
    output.mkdir(parents=True)
    rows = []
    for index, path in enumerate(selected):
        game = Path(path)
        digest = sha(game)
        result = subprocess.run(
            [str(python), str(BRIDGE), "--game", path, "--sha256", digest],
            input='{"op":"close"}\n',
            text=True,
            capture_output=True,
            timeout=30,
            env={**os.environ, "ALFWORLD_DATA": str(DATA)},
        )
        # Reset failures stay in the frozen inventory; no replacement or success filtering.
        receipt = {
            "game_index": index,
            "game": path,
            "game_sha256": digest,
            "family": game.parent.parent.name.split("-")[0],
            "scene": game.parent.parent.name.rsplit("-", 1)[1],
            "split": "valid_unseen",
            "selection_hash": hashlib.sha256(
                f"{SEED}:{game.relative_to(DATA)}".encode()
            ).hexdigest(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode == 0:
            events = [json.loads(line) for line in result.stdout.splitlines()]
            if len(events) != 1 or set(events[0]["public"]) != {"feedback", "admissible_commands"}:
                raise ValueError("unexpected public reset schema")
            receipt.update(public=events[0]["public"], host=events[0]["host"])
        with (output / f"reset-{index:02d}.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
        rows.append(receipt)
    report = {
        "schema": "alfworld-unseen-balanced-proposal-v1",
        "status": "proposed_not_gpu_accepted",
        "source_manifest": str(MANIFEST),
        "source_manifest_sha256": MANIFEST_SHA,
        "selection_seed": SEED,
        "selection": "First two per native family by SHA256(seed:relative_game_path); "
        "official valid_unseen inventory; no scene deduplication, outcome filtering, or replacements.",
        "split": "valid_unseen",
        "official_pool_count": len(inventory),
        "family_pool_counts": dict(
            Counter(Path(p).parent.parent.name.split("-")[0] for p in inventory)
        ),
        "games": rows,
        "game_count": 12,
        "proposed_sampling_seeds": [2026092195, 2026092196],
        "policies": ["flat", "manager_worker", "local_reason"],
        "planned_episodes": 72,
        "maximum_seconds": 7200,
        "action_limit": 50,
        "token_limit": 2048,
        "request_cap": 128,
        "context_limit": 8192,
        "new_runtime_implemented": False,
        "model_calls": 0,
        "manual_actions": 0,
        "reset_failures_retained": sum(r["returncode"] != 0 for r in rows),
        "selected_scene_ids": sorted({r["scene"] for r in rows}),
        "source_environment_sha256": {
            str(p): sha(p)
            for p in (Path(__file__).resolve(), BRIDGE, env / "pyproject.toml", env / "uv.lock")
        },
        "reset_receipt_sha256": {p.name: sha(p) for p in sorted(output.glob("reset-*.json"))},
        "caveat": "Balanced twelve-game unseen exploratory panel, not the natural whole-benchmark mix; "
        "native admissible actions provide affordance assistance. No gold policy/state in prompts.",
    }
    with (output / "MANIFEST.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                "manifest": str(output / "MANIFEST.json"),
                "sha256": sha(output / "MANIFEST.json"),
                "reset_failures": report["reset_failures_retained"],
                "model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
