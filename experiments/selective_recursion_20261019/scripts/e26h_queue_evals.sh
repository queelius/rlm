#!/bin/bash
# usage: e26h_queue_evals.sh RUN_DIR TAG -- waits for RUN_DIR/FINAL, then queues held-out discovery (2 seeds) and one
# MRCR bootstrap seed with that adapter (served on 8001).
RUN=$1; TAG=$2
S=/project/alex_phd/runs/selective-recursion-20261019
R=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007
PY=$R/.venv/bin/python
SP=$S/queue/specs-$TAG; mkdir -p $SP
until [ -f $RUN/FINAL ]; do sleep 60; done
name=$(head -n 1 $RUN/FINAL)
$PY - "$name" "$S" "$PY" "$SP" "$TAG" <<'PYEOF'
import json, sys
name, S, PY, SP, TAG = sys.argv[1:6]
jobs = [dict(name=f"e21-boot-{TAG}-s0", lane="gpu1", priority=9, cmd=[PY, "bootstrap_library.py", "--family", "mrcr", "--out", f"{S}/e21-boot-{TAG}-s0", "--model", name, "--base-url", "http://127.0.0.1:8001/v1", "--helper-base-url", "http://127.0.0.1:8001/v1", "--max-output-tokens", "4096", "--candidates", "8", "--repair-rounds", "2", "--timeout", "1200", "--seed", "0"])]
for bs, suffix in ((0, ""), (1, "-r2")):
    jobs.append(dict(name=f"e26-disc-{TAG}-heldout{suffix}", lane="gpu1", priority=9, cmd=[PY, "discovery_bench.py", "--out", f"{S}/e26-disc-{TAG}-heldout{suffix}", "--candidates", "4", "--repair-rounds", "1", "--concurrency", "4", "--model", name, "--base-url", "http://127.0.0.1:8001/v1", "--helper-base-url", "http://127.0.0.1:8001/v1", "--split", "heldout", "--max-output-tokens", "4096", "--boot-seed", str(bs)]))
for j in jobs:
    open(f"{SP}/{j['name']}.json", "w").write(json.dumps(j, indent=1))
PYEOF
for f in $SP/*.json; do $PY $R/experiments/ops/queue.py add $f; done
echo "[$(date -u +%FT%TZ)] queued $TAG evaluations for $name"
