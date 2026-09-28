"""Saved scripted CPU outputs through the real client, exact decoders and native environment."""

import argparse
import json
import time
from pathlib import Path

import alf_rep as r
import alf_rep_audit as audit
import alf_rep_data as data


def check(output):
    import torch
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("preserve prior fixture attempts")
    records = r.ROOT / "alfworld-train-expert-raw-001/records"
    record = next(
        r.read(path) for path in sorted(records.glob("*.json")) if r.read(path)["won_within_50"]
    )
    readiness = r.read(data.selector.MANIFEST)
    python = str(Path(readiness["environment"]["path"]) / ".venv/bin/python")
    tokenizer = AutoTokenizer.from_pretrained(r.BASE, local_files_only=True)
    commands = [step["executed_action"] for step in record["steps"]]
    game = dict(
        game=record["game"],
        game_sha256=record["game_sha256"],
        game_index=0,
        public=record["steps"][0]["public"],
        family=record["family"],
        scene="TRAIN-fixture",
    )
    cases, sequences = [], []
    for mode in ("index", "command"):
        directory = output / mode
        job = dict(
            episode_id="train-fixture-flat", game_index=0, game=game, seed=r.SEEDS[0], policy="flat"
        )
        plan = dict(
            schema="alfworld-index-command-readout-20260928-v1",
            representation=mode,
            actor="base",
            fixture=True,
            cases=[job],
            planned_episodes=1,
            model=str(r.BASE),
            action_limit=50,
            token_limit=2048,
            request_cap=128,
            context_limit=8192,
            alfworld_python=python,
            data_root=str(data.DATA),
            source_sha256=r.source_pins(),
        )
        r.save(directory / "PLAN.json", plan)
        actions = iter(commands)

        class Model:
            device = "cpu"
            invalid_sent = False

            def __init__(self, interface, command_iterator):
                self.interface, self.commands = interface, command_iterator

            def generate(self, input_ids, **kwargs):
                prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                start = prompt.index('{"public_context":')
                context = json.JSONDecoder().raw_decode(prompt[start:])[0]["public_context"]
                if self.interface == "command" and not self.invalid_sent:
                    self.invalid_sent = True
                    text = '{"command":"LOOK"}'
                else:
                    command = next(self.commands)
                    options = [row["command"] for row in context["admissible_commands"]]
                    if command not in options:
                        raise ValueError("fixed TRAIN action differs from actual native state")
                    target = (
                        {"command": command}
                        if self.interface == "command"
                        else {"action_index": options.index(command)}
                    )
                    text = json.dumps(target, separators=(",", ":"))
                emitted = tokenizer.encode(text, add_special_tokens=False) + [
                    tokenizer.eos_token_id
                ]
                return torch.tensor([input_ids[0].tolist() + emitted])

        native = r.controller(mode)
        client = native.BaseClient(Model(mode, actions), tokenizer, directory, time.time() + 120)
        env = r.actor.unseen.CheckedBridge(
            game, python, str(data.DATA), time.time() + 120, directory / "native.stderr"
        )
        env.expected_public = game["public"]
        try:
            row = native.play_episode(client, env, job, directory)
        finally:
            env.close()
        checked = audit.audit(directory, require_terminal=False)
        r.save(directory / "NATIVE-AUDIT.json", checked)
        sequence = [
            r.read(directory / "observations" / f"train-fixture-flat-{i:03d}.json")["action"]
            for i in range(1, row["actions"] + 1)
        ]
        sequences.append(sequence)
        cases.append(
            dict(
                mode=mode,
                won=row["observed"] and row["won"],
                actions=row["actions"],
                calls=len(row["call_ids"]),
                invalid_outputs=row["invalid_outputs"],
                native_replayed=checked["audits"][job["episode_id"]]["native_replayed"],
            )
        )
    result = dict(
        passed=all(row["won"] and row["native_replayed"] for row in cases),
        cases=cases,
        same_native_action_sequence=sequences[0] == sequences[1] == commands,
        scientific_model_calls=0,
        GPU_used=False,
        scope="Fixed TRAIN demonstration-scripted tensors, real tokenizer/decode/client receipts, "
        "native bridge and independently restarted native replay. Not model competence.",
    )
    r.save(output / "VERIFICATION.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(check(parser.parse_args().output), indent=2))
