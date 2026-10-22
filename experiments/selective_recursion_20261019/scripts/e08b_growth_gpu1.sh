#!/bin/bash
# E08b (10:05 UTC): the harder held-out type (busiest_day, tie rule). Cache its teaching extractions and
# run the before-growth baseline now; after the first growth chain finishes, run the librarian for it,
# rebuild the statistics (v2) and evaluate after growth.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
$PY extract_cache.py --out $S/e08-events-cache/teach --split teach --seeds 16-23 --qts busiest_day --concurrency 3 > $H/e08-events-cache/teach-busiest.log 2>&1 &
mkdir -p $H/e08b-dev-heldout-before && ln -sfn $H/e08b-dev-heldout-before $S/e08b-dev-heldout-before
$PY run_sweep.py --out $S/e08b-dev-heldout-before --policies rlm,direct --harness skills --stats-file $S/stats-teach-v0.json --controller-model qwen3-4b --ns 8,32,64,128,256,512 --qts busiest_day --seeds 1000-1003 --split dev --concurrency 3 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2
wait
log "busiest_day cache: $(ls $S/e08-events-cache/teach | grep -c busiest_day) tasks; before-eval rows: $(wc -l < $S/e08b-dev-heldout-before/rows.jsonl)"
until grep -q E08_DONE $S/e08_growth_gpu1.log; do sleep 60; done
mkdir -p $S/e08-librarian-busiest_day
$PY librarian.py --qt busiest_day --cache $S/e08-events-cache/teach --out $S/e08-librarian-busiest_day --candidates 8 --register 2>&1 | grep -v -iE "warn" | tail -n 10
$PY build_stats.py --out $S/stats-teach-v2.json $S/teach-lib-009 $S/sweep-001-t0 $S/teach-lib-base-stats $S/e08-librarian-top_buyer:lib:answer_top_buyer $S/e08-librarian-purchase_span:lib:answer_purchase_span $S/e08-librarian-busiest_day:lib:answer_busiest_day
mkdir -p $H/e08b-dev-heldout-after && ln -sfn $H/e08b-dev-heldout-after $S/e08b-dev-heldout-after
$PY run_sweep.py --out $S/e08b-dev-heldout-after --policies rlm --harness skills --stats-file $S/stats-teach-v2.json --controller-model qwen3-4b --ns 8,32,64,128,256,512 --qts busiest_day --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2
log E08B_DONE
