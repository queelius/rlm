#!/bin/bash
# E09 (10:40 UTC): the 293-row held-out OOLONG run of the untrained controller with the prebound OOLONG
# library (the paper number for L1-d), queued behind the growth chains so the server is not saturated further.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
until grep -q E08B_DONE $S/e08b_growth_gpu1.log; do sleep 60; done
mkdir -p $H/oolong-lib-base-full && ln -sfn $H/oolong-lib-base-full $S/oolong-lib-base-full
log "starting oolong-lib-base-full"
$PY oolong_eval.py --out $S/oolong-lib-base-full --policies rlm --harness skills --controller-model qwen3-4b --lens 1024,2048,4096,8192,16384,32768 --per-cell 50 --concurrency 6 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 8
log E09_DONE
