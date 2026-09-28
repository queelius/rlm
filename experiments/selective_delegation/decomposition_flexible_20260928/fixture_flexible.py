"""Reuse the real request/native replay fixture with the new routing adapter."""

import argparse
import json
import sys
from pathlib import Path

import routing_flexible
import run_flexible

data = run_flexible.data
# Upstream environment imports add unrelated repositories to sys.path. Bind the
# fixture's intentionally generic import only while loading that frozen module.
previous = sys.modules.get("run")
sys.modules["run"] = run_flexible
try:
    fixture = data.load(run_flexible.OLD / "fixture.py", "flexible_native_fixture")
finally:
    if previous is None:
        sys.modules.pop("run", None)
    else:
        sys.modules["run"] = previous
fixture.run = run_flexible
fixture.routing = routing_flexible

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fixture.check(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))
