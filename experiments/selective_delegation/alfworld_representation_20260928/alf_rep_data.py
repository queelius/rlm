"""Same524 native TRAIN targets and new outcome-blind12-game native reset inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import alf_rep as r

selector = r.module(r.LIBRARY / "prepare_alfworld_unseen_panel.py", "alf_rep_old_panel")
DATA, FAMILIES = selector.DATA, selector.FAMILIES
PANEL, COMMAND_DATA = r.STUDY / "panel", r.STUDY / "command-data"
SELECTION_SEED = 2026092812


def original_rows():
    path = r.ORIGINAL_DATA / "examples.jsonl"
    if r.sha(path) != r.ORIGINAL_DATA_SHA:
        raise ValueError("original524 TRAIN examples changed")
    return [json.loads(line) for line in path.read_text().splitlines()]


def project_rows():
    rows = []
    for original in original_rows():
        if original["dropped_history"] != 0:
            raise ValueError("original demonstration lacks full history")
        context = original["prompt"].split("\n", 1)[1]
        public = json.loads(context)["public_context"]
        index = json.loads(original["target_json"])["action_index"]
        options = public["admissible_commands"]
        if [row["action_index"] for row in options] != list(range(len(options))):
            raise ValueError("original native numbered list changed")
        command = options[index]["command"]
        target = json.dumps({"command": command}, separators=(",", ":"))
        rows.append(
            dict(
                original,
                prompt=r.COMMAND_INSTRUCTION + "\n" + context,
                target_json=target,
                original_action_index=index,
                executed_command=command,
            )
        )
    return rows


def select_panel():
    if r.sha(selector.MANIFEST) != selector.MANIFEST_SHA:
        raise ValueError("native readiness inventory changed")
    readiness = r.read(selector.MANIFEST)
    old_panel = r.read(r.ROOT / "alfworld-unseen-inputs-001/MANIFEST.json")
    excluded = {row["game"] for row in readiness["proposed_screen"]["games"]}
    excluded.update(row["game"] for row in readiness["manual_traces"])
    excluded.update(row["game"] for row in old_panel["games"])
    pool = readiness["official_filtered_game_inventories"]["eval_out_of_distribution"]
    selected = []
    for family in FAMILIES:
        candidates = [
            path
            for path in pool
            if path not in excluded and (Path(path).parent.parent.name.split("-", 1)[0] == family)
        ]
        if len(candidates) < 2:
            raise ValueError("insufficient untouched official family inventory")
        selected.extend(
            sorted(
                candidates,
                key=lambda path: hashlib.sha256(
                    f"{SELECTION_SEED}:{Path(path).relative_to(DATA)}".encode()
                ).hexdigest(),
            )[:2]
        )
    return dict(
        games=selected,
        excluded_games=sorted(excluded),
        selection_seed=SELECTION_SEED,
        family_counts=dict(Counter(Path(p).parent.parent.name.split("-", 1)[0] for p in selected)),
        selection="First two per family by SHA256(seed:relative_path), "
        "after prior-panel exclusion; "
        "no outcome/feasibility replacement or scene deduplication.",
    )


def prepare():
    from transformers import AutoTokenizer

    if PANEL.exists() or COMMAND_DATA.exists():
        raise ValueError("preserve immutable preparation attempts")
    tokenizer = AutoTokenizer.from_pretrained(r.BASE, local_files_only=True)
    projected, old = project_rows(), original_rows()
    for row in projected:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = tokenizer.encode(row["target_json"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if len(prefix) + 128 > 8192 or len(target) > 128:
            raise ValueError("projected demonstration does not fit original limits")
        row.update(
            prompt_tokens=len(prefix),
            target_tokens=len(target),
            loss_mask_tokens=len(target),
            target_with_eos=row["target_json"] + tokenizer.eos_token,
        )
    COMMAND_DATA.mkdir(parents=True)
    with (COMMAND_DATA / "examples.jsonl").open("x") as stream:
        for row in projected:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    r.save(
        COMMAND_DATA / "MANIFEST.json",
        dict(
            schema="alfworld-exact-command-projection-20260928-v1",
            examples=524,
            successful_games_only=True,
            examples_sha256=r.sha(COMMAND_DATA / "examples.jsonl"),
            original_examples_sha256=r.ORIGINAL_DATA_SHA,
            same_row_order=True,
            same_executed_native_commands=True,
            all_histories_complete=True,
            prompt_tokens=sum(row["prompt_tokens"] for row in projected),
            supervised_tokens=sum(row["target_tokens"] for row in projected),
            original_prompt_tokens=sum(row["prompt_tokens"] for row in old),
            original_supervised_tokens=sum(row["target_tokens"] for row in old),
            max_prompt_tokens=max(row["prompt_tokens"] for row in projected),
            max_target_tokens=max(row["target_tokens"] for row in projected),
            source_sha256={
                str(path): r.sha(path) for path in (Path(__file__).resolve(), r.HERE / "alf_rep.py")
            },
            caveat="Same actions/rows/updates, not equal token exposure "
            "or per-example token-mean weight.",
        ),
    )
    selection = select_panel()
    readiness = r.read(selector.MANIFEST)
    python = str(Path(readiness["environment"]["path"]) / ".venv/bin/python")
    games = []
    for index, filename in enumerate(selection["games"]):
        game = Path(filename)
        native = subprocess.run(
            [python, str(selector.BRIDGE), "--game", filename, "--sha256", r.sha(game)],
            input='{"op":"close"}\n',
            text=True,
            capture_output=True,
            timeout=30,
            env={**os.environ, "ALFWORLD_DATA": str(DATA)},
        )
        row = dict(
            game_index=index,
            game=filename,
            game_sha256=r.sha(game),
            family=game.parent.parent.name.split("-", 1)[0],
            scene=game.parent.parent.name.rsplit("-", 1)[1],
            split="valid_unseen",
            returncode=native.returncode,
            stdout=native.stdout,
            stderr=native.stderr,
        )
        if native.returncode == 0:
            events = [json.loads(line) for line in native.stdout.splitlines()]
            if len(events) != 1 or set(events[0]["public"]) != {"feedback", "admissible_commands"}:
                raise ValueError("native public reset schema changed")
            row.update(public=events[0]["public"], host=events[0]["host"])
        r.save(PANEL / f"reset-{index:02d}.json", row)
        games.append(row)
    r.save(
        PANEL / "MANIFEST.json",
        dict(
            schema="alfworld-representation-fresh-panel-20260928-v1",
            split="valid_unseen",
            selection=selection,
            games=games,
            game_count=12,
            seeds=list(r.SEEDS),
            alfworld_python=python,
            data_root=str(DATA),
            readiness_manifest_sha256=selector.MANIFEST_SHA,
            previous_panel_sha256=r.sha(r.ROOT / "alfworld-unseen-inputs-001/MANIFEST.json"),
            ready=all(row["returncode"] == 0 and not row["host"]["won"] for row in games),
            reset_failures_retained=sum(row["returncode"] != 0 for row in games),
            scenes=sorted({row["scene"] for row in games}),
            model_calls=0,
            expert_evaluation_actions=0,
            source_sha256={
                str(path): r.sha(path) for path in (Path(__file__).resolve(), selector.BRIDGE)
            },
        ),
    )
    return dict(
        training=r.read(COMMAND_DATA / "MANIFEST.json"), panel=r.read(PANEL / "MANIFEST.json")
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = prepare()
    print(
        json.dumps(
            dict(
                examples=result["training"]["examples"],
                target_tokens=result["training"]["supervised_tokens"],
                original_target_tokens=result["training"]["original_supervised_tokens"],
                panel_ready=result["panel"]["ready"],
                scenes=result["panel"]["scenes"],
                GPU_used=False,
            ),
            indent=2,
        )
    )
