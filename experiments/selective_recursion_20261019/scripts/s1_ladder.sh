#!/bin/bash
# S1 caller-capacity ladder (2026-10-21 10:40 UTC). Controller served on GPU 0 (non-thinking Qwen3 0.6B, 1.7B,
# 8B-AWQ); helpers stay the 4B on GPU 1. Per caller: event-log dev without a library (default harness), with the
# library + statistics note (v4), the four grown held-out forms, and the 72-row OOLONG sample with the library.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
one() {  # name model port
  local n=$1 m=$2 u=http://127.0.0.1:$3/v1
  until grep -q "Application startup complete" $S/serving/vllm_gpu0_$m.log; do sleep 20; done
  for d in s1-$n-dev-nolib s1-$n-dev-lib s1-$n-heldout s1-$n-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
  $PY run_sweep.py --out $S/s1-$n-dev-nolib --policies rlm --harness default --controller-model $m --controller-base-url $u --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1
  $PY run_sweep.py --out $S/s1-$n-dev-lib --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model $m --controller-base-url $u --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1
  $PY run_sweep.py --out $S/s1-$n-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model $m --controller-base-url $u --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1
  $PY oolong_eval.py --out $S/s1-$n-oolong --policies rlm --harness skills --controller-model $m --controller-base-url $u --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 4 2>&1 | tail -n 1
  log "S1 $n done"
}
one 1.7b qwen3-1.7b-nt 8003 > $S/s1_1.7b.log 2>&1 &
one 8b qwen3-8b-awq-nt 8002 > $S/s1_8b.log 2>&1 &
one 0.6b qwen3-0.6b-nt 8004 > $S/s1_0.6b.log 2>&1 &
wait
log S1_DONE
