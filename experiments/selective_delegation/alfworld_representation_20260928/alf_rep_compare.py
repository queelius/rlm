"""Native-success training-by-representation interaction; missing cells remain unknown."""

import argparse
import json
from collections import Counter, defaultdict
from statistics import mean

import alf_rep as r
import alf_rep_audit as audit
import alf_rep_data as data

old = r.module(r.LIBRARY / "analyze_alfworld_unseen.py", "alf_rep_paired_statistics")
CELLS = ("index-base", "index-trained", "command-base", "command-trained")


def interaction(cells):
    if any(cells.get(name) is None for name in CELLS):
        return None
    return (
        cells["command-trained"]
        - cells["command-base"]
        - (cells["index-trained"] - cells["index-base"])
    )


def analyze():
    panel = r.read(data.PANEL / "MANIFEST.json")
    games = {game["game_index"]: game for game in panel["games"]}
    rows, cells = [], {}
    for name in CELLS:
        output = r.STUDY / name
        if not (output / "PLAN.json").exists() or not list(output.glob("TERMINAL-*.json")):
            cells[name] = dict(available=False, reason="no terminal scientific cell")
            continue
        path = output / "NATIVE-AUDIT.json"
        if path.exists():
            report = r.read(path)
            for filename, digest in report["receipt_sha256"].items():
                if r.sha(filename) != digest:
                    raise ValueError("previously native-audited receipt changed")
        else:
            report = audit.audit(output)
            r.save(path, report)
        plan = report["plan"]
        if f"{plan['representation']}-{plan['actor']}" != name or (
            plan["panel_sha256"] != r.sha(data.PANEL / "MANIFEST.json")
        ):
            raise ValueError("comparison interface/panel identity changed")
        episodes = report["rows"]
        rows.extend(dict(row, policy=name) for row in episodes)
        repetitions = []
        for row in episodes:
            actions = [
                r.read(output / "observations" / (row["episode_id"] + f"-{i:03d}.json"))["action"]
                for i in range(1, row["actions"] + 1)
            ]
            repetitions.append(sum(a == b for a, b in zip(actions, actions[1:], strict=False)))
        calls = [r.read(path) for path in (output / "calls").glob("*.json")]
        cells[name] = dict(
            available=report["complete"],
            planned=24,
            recorded=len(episodes),
            observed=sum(row["observed"] for row in episodes),
            won=sum(row["observed"] and row["won"] for row in episodes),
            invalid_outputs=sum(row["invalid_outputs"] for row in episodes),
            terminations=dict(Counter(row["termination"] for row in episodes)),
            immediate_repeated_native_commands=sum(repetitions),
            physical_cost=report["physical_cost"],
            generated_tokens=sum(row["generated_tokens"] for row in episodes),
            max_prompt_tokens=max((len(call["input_token_ids"]) for call in calls), default=0),
            history_trimmed_calls=sum(call["trimming"]["dropped_history"] > 0 for call in calls),
            audit_sha256=r.sha(path),
            plan_sha256=r.sha(output / "PLAN.json"),
            checkpoint_binding=plan["checkpoint_binding"],
        )
    lookup = {(row["game_index"], row["seed"], row["policy"]): row for row in rows}
    interactions, game_values, scenes = [], {}, defaultdict(list)
    for game, meta in games.items():
        deltas = []
        for seed in r.SEEDS:
            outcomes = {}
            for cell in CELLS:
                row = lookup.get((game, seed, cell))
                outcomes[cell] = int(row["won"]) if row and row["observed"] else None
            delta = interaction(outcomes)
            deltas.append(delta)
            interactions.append(dict(game_index=game, seed=seed, cells=outcomes, delta=delta))
        if all(value is not None for value in deltas):
            game_values[str(game)] = mean(deltas)
        scenes[meta["scene"]].append(str(game))
    complete = len(game_values) == 12
    return dict(
        schema="alfworld-representation-learning-gain-20260928-v1",
        cells=cells,
        within_interface_gain={
            mode: old.paired(rows, games, r.SEEDS, f"{mode}-trained", f"{mode}-base")
            for mode in ("index", "command")
        },
        interaction=dict(
            definition="(command-trained-command-base)-(index-trained-index-base)",
            pairs=interactions,
            unknown=sum(pair["delta"] is None for pair in interactions),
            game_bootstrap=old.analyze_helper.clustered_interval(
                game_values, [[str(game)] for game in games], 20000, 2026092812
            )
            if complete
            else None,
            scene_bootstrap=old.analyze_helper.clustered_interval(
                game_values, list(scenes.values()), 20000, 2026092812
            )
            if complete
            else None,
        ),
        original_index_checkpoint=r.checkpoint_binding("index"),
        command_training_manifest=r.read(data.COMMAND_DATA / "MANIFEST.json"),
        interpretation="Native success primary. Same rows/actions/33 updates, but unequal target "
        "tokens, token-mean example weights and output latency. Instruction and decoder differ; "
        "a gain interaction is the training/representation package, not proof of symbol binding. "
        "12 prospectively new games, shared scenes, two correlated seeds; exploratory only. "
        "No manager hierarchy, action permutation or automatic scaling.",
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = analyze()
    r.save(r.STUDY / "COMPARISON.json", report)
    print(json.dumps(dict(cells=report["cells"], interaction=report["interaction"]), indent=2))
