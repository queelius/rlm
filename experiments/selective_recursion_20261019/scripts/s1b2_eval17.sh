#!/bin/bash
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
U=http://127.0.0.1:8003/v1
cd $E
until curl -s -m 10 $U/models -H "Authorization: Bearer local" | grep -q ctrl-17-lib; do sleep 15; done
for d in s1b-17lib-dev s1b-17lib-heldout s1b-17lib-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
$PY run_sweep.py --out $S/s1b-17lib-dev --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-17-lib --controller-base-url $U --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
$PY run_sweep.py --out $S/s1b-17lib-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-17-lib --controller-base-url $U --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
$PY oolong_eval.py --out $S/s1b-17lib-oolong --policies rlm --harness skills --controller-model ctrl-17-lib --controller-base-url $U --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 4 2>&1 | tail -n 1 &
wait
echo "[$(date -u +%FT%TZ)] S1B2_DONE"
