#!/bin/bash
# E08c (11:50 UTC): busiest_day with repair rounds; then stats v2, the busiest_day after-eval, and the warm bandit arm.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
mkdir -p $S/e08-librarian-busiest_day
$PY librarian.py --qt busiest_day --cache $S/e08-events-cache/teach --out $S/e08-librarian-busiest_day --candidates 8 --repair-rounds 2 --register 2>&1 | grep -v -iE "warn" | tail -n 30
SRC="$S/teach-lib-009 $S/sweep-001-t0 $S/teach-lib-base-stats $S/e08-librarian-top_buyer:lib:answer_top_buyer $S/e08-librarian-purchase_span:lib:answer_purchase_span"
[ -f $S/e08-librarian-busiest_day/rows.jsonl ] && SRC="$SRC $S/e08-librarian-busiest_day:lib:answer_busiest_day"
$PY build_stats.py --out $S/stats-teach-v2.json $SRC
log "learned skills: $(ls skills/learned/ | tr '\n' ' ')"
mkdir -p $H/e08b-dev-heldout-after && ln -sfn $H/e08b-dev-heldout-after $S/e08b-dev-heldout-after
$PY run_sweep.py --out $S/e08b-dev-heldout-after --policies rlm --harness skills --stats-file $S/stats-teach-v2.json --controller-model qwen3-4b --ns 8,32,64,128,256,512 --qts busiest_day --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2 &
rm -rf $H/e10-bandit-greedy-warm/*; mkdir -p $H/e10-bandit-greedy-warm
$PY online_bandit.py --out $S/e10-bandit-greedy-warm --policy greedy --warm $S/stats-teach-v2.json --steps 240 --batch 4 > $H/e10-bandit-greedy-warm/run.log 2>&1 &
wait
log E08C_DONE
