"""Isolated official text engine; public observation and host metrics never merged."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def project(state, *, action=None, reward=0, done=False):
    return {
        "event": "reset" if action is None else "step",
        "action": action,
        "public": {
            "feedback": state.feedback,
            "admissible_commands": list(state.admissible_commands),
        },
        "host": {"won": bool(state.won), "done": bool(done), "reward": reward},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.game.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("frozen game changed")
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    env = textworld.start(
        str(args.game),
        textworld.EnvInfos(won=True, admissible_commands=True),
        wrappers=[AlfredDemangler(shuffle=False)],
    )
    try:
        state = env.reset()
        row = project(state)
        row["host"]["versions"] = {
            p: importlib.metadata.version(p) for p in ("alfworld", "textworld")
        }
        print(json.dumps(row), flush=True)
        done = False
        for line in sys.stdin:
            request = json.loads(line)
            if request == {"op": "close"}:
                break
            if (
                set(request) != {"action"}
                or not isinstance(request["action"], str)
                or request["action"] not in state.admissible_commands
                or done
            ):
                raise ValueError("invalid bridge action or terminal environment")
            state, reward, done = env.step(request["action"])
            print(
                json.dumps(project(state, action=request["action"], reward=reward, done=done)),
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
