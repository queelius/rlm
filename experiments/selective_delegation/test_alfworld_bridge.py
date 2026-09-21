import hashlib
import json
import os
import subprocess
from pathlib import Path


def test_real_isolated_bridge_reset_step_and_public_projection():
    script = Path(__file__).with_name("alfworld_bridge.py")
    assert script.exists(), "bridge implementation missing"
    root = Path("/project/alex_phd/research-cache/datasets/alfworld-text-0.4.2-20260921")
    manifest = json.loads((root / "readiness/MANIFEST.json").read_text())
    game = manifest["proposed_screen"]["games"][0]
    python = Path(manifest["environment"]["path"]) / ".venv/bin/python"
    # Real text engine, no model or GPU. 'look' is native and outcome independent.
    result = subprocess.run(
        [str(python), str(script), "--game", game["game"], "--sha256", game["game_sha256"]],
        input='{"action":"look"}\n{"op":"close"}\n',
        text=True,
        capture_output=True,
        timeout=30,
        env={**os.environ, "ALFWORLD_DATA": str(root)},
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 2
    assert rows[0]["event"] == "reset" and rows[1]["event"] == "step"
    assert rows[1]["action"] == "look" and not rows[0]["host"]["won"]
    assert set(rows[0]["public"]) == {"feedback", "admissible_commands"}
    assert game["game"] not in json.dumps(rows[0]["public"])
    assert hashlib.sha256(Path(game["game"]).read_bytes()).hexdigest() == game["game_sha256"]
