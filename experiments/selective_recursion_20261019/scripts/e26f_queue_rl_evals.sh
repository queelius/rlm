#!/bin/bash
# Waits for the E26c RL run's FINAL marker and queues its evaluations with the final adapter (served on 8001).
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
R=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007
PY=$R/.venv/bin/python
SP=$S/queue/specs-e26f; mkdir -p $SP
until [ -f $H/e26c-proposer-rl/FINAL ]; do sleep 60; done
name=$(head -n 1 $H/e26c-proposer-rl/FINAL)
$PY - "$name" "$S" "$PY" "$SP" <<'PYEOF'
import json, sys
name, S, PY, SP = sys.argv[1:5]
jobs = []
for sd in (0, 1, 2):
    jobs.append(dict(name=f"e21-boot-4brl-s{sd}", lane="gpu1", priority=9, cmd=[PY, "bootstrap_library.py", "--family", "mrcr", "--out", f"{S}/e21-boot-4brl-s{sd}", "--model", name, "--base-url", "http://127.0.0.1:8001/v1", "--helper-base-url", "http://127.0.0.1:8001/v1", "--max-output-tokens", "4096", "--candidates", "8", "--repair-rounds", "2", "--timeout", "1200", "--seed", str(sd)]))
for bs, suffix in ((0, ""), (1, "-r2")):
    jobs.append(dict(name=f"e26-disc-4brl-heldout{suffix}", lane="gpu1", priority=9, cmd=[PY, "discovery_bench.py", "--out", f"{S}/e26-disc-4brl-heldout{suffix}", "--candidates", "4", "--repair-rounds", "1", "--concurrency", "4", "--model", name, "--base-url", "http://127.0.0.1:8001/v1", "--helper-base-url", "http://127.0.0.1:8001/v1", "--split", "heldout", "--max-output-tokens", "4096", "--boot-seed", str(bs)]))
for j in jobs:
    open(f"{SP}/{j['name']}.json", "w").write(json.dumps(j, indent=1))
print(name, len(jobs))
PYEOF
for f in $SP/*.json; do $PY $R/experiments/ops/queue.py add $f; done
echo "[$(date -u +%FT%TZ)] queued RL evaluations for $name"
