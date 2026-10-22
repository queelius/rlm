#!/bin/bash
# E08d (12:15 UTC): the rejected form (busiest_day) with the 8B proposer (Qwen3-8B-AWQ on GPU 0, thinking on,
# 6000 output tokens), two repair rounds; if accepted, stats v3 and the busiest_day after-eval with the controller.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
mkdir -p $S/e08-librarian-busiest_day-8b
$PY librarian.py --qt busiest_day --cache $S/e08-events-cache/teach --out $S/e08-librarian-busiest_day-8b --model qwen3-8b-awq --base-url http://127.0.0.1:8002/v1 --max-output-tokens 6000 --candidates 8 --repair-rounds 2 --register 2>&1 | grep -v -iE "warn" | tail -n 30
if [ -f $S/e08-librarian-busiest_day-8b/rows.jsonl ]; then
  $PY build_stats.py --out $S/stats-teach-v3.json $S/teach-lib-009 $S/sweep-001-t0 $S/teach-lib-base-stats $S/e08-librarian-top_buyer:lib:answer_top_buyer $S/e08-librarian-purchase_span:lib:answer_purchase_span $S/e08-librarian-busiest_day-8b:lib:answer_busiest_day
  mkdir -p $H/e08d-dev-heldout-after && ln -sfn $H/e08d-dev-heldout-after $S/e08d-dev-heldout-after
  $PY run_sweep.py --out $S/e08d-dev-heldout-after --policies rlm --harness skills --stats-file $S/stats-teach-v3.json --controller-model qwen3-4b --ns 8,32,64,128,256,512 --qts busiest_day --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2
fi
log E08D_DONE
