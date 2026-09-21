"""Collect a tiny, host-target-only ALFWorld TRAIN expert-trace qualification set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SEED = 2026092197
SUPPORTED_FAMILIES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(game: Path) -> str:
    return game.parent.parent.name.split("-", 1)[0]


def select_games(games: list[Path]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for task_family in SUPPORTED_FAMILIES:
        candidates = [game for game in games if family(game) == task_family]
        if not candidates:
            raise ValueError(f"no TRAIN game for supported family {task_family}")
        selected[task_family] = min(
            candidates,
            key=lambda path: hashlib.sha256(f"{SEED}:{path}".encode()).hexdigest(),
        )
    return selected


def public_row(feedback: str, admissible_commands: list[str]) -> dict:
    return {"feedback": feedback, "admissible_commands": list(admissible_commands)}


def completed_won(rows: list[dict]) -> bool:
    return bool(rows and rows[-1].get("host_result", {}).get("won"))


def collect_game(game: Path) -> dict:
    import textworld
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
    )

    env = textworld.start(
        str(game),
        textworld.EnvInfos(won=True, admissible_commands=True),
        wrappers=[
            AlfredDemangler(shuffle=False),
            AlfredExpert(expert_type=AlfredExpertType.HANDCODED),
        ],
    )
    rows = []
    failure = None
    try:
        state = env.reset()
        done = False
        while not done:
            commands = list(state.admissible_commands)
            target = state.get("extra.expert_plan", [])
            if len(target) != 1 or target[0] not in commands:
                failure = "expert_missing_or_not_admissible"
                break
            command = target[0]
            rows.append(
                {
                    "public": public_row(state.feedback, commands),
                    "host_target": {
                        "expert_command": command,
                        "action_index": commands.index(command),
                    },
                }
            )
            state, reward, done = env.step(command)
            rows[-1]["host_result"] = {"reward": reward, "won": bool(state.won), "done": bool(done)}
    except Exception as exc:  # Preserve a bounded qualification failure as data.
        failure = f"{type(exc).__name__}: {exc}"
        if rows and "host_result" not in rows[-1]:
            rows[-1]["host_result"] = {"status": "unavailable", "error": failure}
    finally:
        env.close()
    return {
        "family": family(game),
        "game": str(game),
        "game_sha256": sha256(game),
        "steps": rows,
        "won": completed_won(rows),
        "failure": failure,
    }


def main(args: argparse.Namespace) -> int:
    data_root, output = args.data_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("immutable output already exists")
    os.environ["ALFWORLD_DATA"] = str(data_root)
    games = sorted((data_root / "json_2.1.1" / "train").glob("*/*/game.tw-pddl"))
    selected = select_games(games)
    output.mkdir(parents=True)
    records = [collect_game(selected[task_family]) for task_family in SUPPORTED_FAMILIES]
    with (output / "TRACES.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "schema": "alfworld-train-expert-trace-feasibility-v1",
        "selection_seed": SEED,
        "selection": "one SHA256(seed:path) TRAIN game per supported official text family",
        "supported_families": SUPPORTED_FAMILIES,
        "source_data_root": str(data_root),
        "records": len(records),
        "completed_wins": sum(record["won"] for record in records),
        "failures": sum(record["failure"] is not None for record in records),
        "traces_sha256": sha256(output / "TRACES.jsonl"),
        "public_contract": (
            "Only feedback and admissible_commands are policy inputs; "
            "expert command is host target."
        ),
    }
    with (output / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
