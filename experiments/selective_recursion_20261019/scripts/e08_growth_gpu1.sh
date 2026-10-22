#!/bin/bash
# E08 library growth (10:00 UTC): wait for the extraction cache, run the librarian for both held-out
# types with registration, rebuild the statistics table with the verification rows, then evaluate the
# untrained controller with the grown library and the note on the held-out dev tasks.
S=/project/alex_phd/runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
until grep -q "^done" /home/atowell/research-runs/selective-recursion-20261019/e08-events-cache/teach.log; do sleep 30; done
log "cache complete: $(ls $S/e08-events-cache/teach | wc -l) tasks"
for qt in top_buyer purchase_span; do
  mkdir -p $S/e08-librarian-$qt
  $PY librarian.py --qt $qt --cache $S/e08-events-cache/teach --out $S/e08-librarian-$qt --candidates 8 --register 2>&1 | grep -v -iE "warn" | tail -n 4
done
log "learned skills: $(ls skills/learned/)"
$PY build_stats.py --out $S/stats-teach-v1.json $S/teach-lib-009 $S/sweep-001-t0 $S/teach-lib-base-stats $S/e08-librarian-top_buyer:lib:answer_top_buyer $S/e08-librarian-purchase_span:lib:answer_purchase_span
mkdir -p /home/atowell/research-runs/selective-recursion-20261019/e08-dev-heldout-after && ln -sfn /home/atowell/research-runs/selective-recursion-20261019/e08-dev-heldout-after $S/e08-dev-heldout-after
$PY run_sweep.py --out $S/e08-dev-heldout-after --policies rlm --harness skills --stats-file $S/stats-teach-v1.json --controller-model qwen3-4b --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 3
log E08_DONE
