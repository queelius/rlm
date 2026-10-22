#!/bin/bash
# E11 (12:35 UTC): the join form (same_day_cross). Cache teaching extractions and the before-growth baseline;
# 4B librarian with two repair rounds; if rejected, the 8B librarian; stats v4 and the after-eval with the
# untrained 4B caller.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
$PY extract_cache.py --out $S/e08-events-cache/teach --split teach --seeds 16-23 --qts same_day_cross --concurrency 3 > $H/e08-events-cache/teach-cross.log 2>&1 &
mkdir -p $H/e11-dev-heldout-before && ln -sfn $H/e11-dev-heldout-before $S/e11-dev-heldout-before
$PY run_sweep.py --out $S/e11-dev-heldout-before --policies rlm,direct --harness skills --stats-file $S/stats-teach-v3.json --controller-model qwen3-4b --ns 8,32,64,128,256,512 --qts same_day_cross --seeds 1000-1003 --split dev --concurrency 3 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2
wait
log "cache: $(ls $S/e08-events-cache/teach | grep -c same_day_cross) tasks; before rows: $(wc -l < $S/e11-dev-heldout-before/rows.jsonl)"
mkdir -p $S/e11-librarian-4b
$PY librarian.py --qt same_day_cross --cache $S/e08-events-cache/teach --out $S/e11-librarian-4b --candidates 8 --repair-rounds 2 --register 2>&1 | grep -v -iE "warn" | tail -n 30
SRC="$S/teach-lib-009 $S/sweep-001-t0 $S/teach-lib-base-stats $S/e08-librarian-top_buyer:lib:answer_top_buyer $S/e08-librarian-purchase_span:lib:answer_purchase_span $S/e08-librarian-busiest_day-8b:lib:answer_busiest_day"
if [ -f $S/e11-librarian-4b/rows.jsonl ]; then
  SRC="$SRC $S/e11-librarian-4b:lib:answer_same_day_cross"; WHO=4b
else
  until grep -q "Application startup complete" $S/serving/vllm_gpu0_8b_awq.log; do sleep 20; done
  mkdir -p $S/e11-librarian-8b
  $PY librarian.py --qt same_day_cross --cache $S/e08-events-cache/teach --out $S/e11-librarian-8b --model qwen3-8b-awq --base-url http://127.0.0.1:8002/v1 --max-output-tokens 6000 --candidates 8 --repair-rounds 2 --register 2>&1 | grep -v -iE "warn" | tail -n 30
  [ -f $S/e11-librarian-8b/rows.jsonl ] && SRC="$SRC $S/e11-librarian-8b:lib:answer_same_day_cross"; WHO=8b
fi
$PY build_stats.py --out $S/stats-teach-v4.json $SRC
log "learned skills: $(ls skills/learned/ | tr '\n' ' ') (proposer $WHO)"
mkdir -p $H/e11-dev-heldout-after && ln -sfn $H/e11-dev-heldout-after $S/e11-dev-heldout-after
$PY run_sweep.py --out $S/e11-dev-heldout-after --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model qwen3-4b --ns 8,32,64,128,256,512 --qts same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2
log E11_DONE
