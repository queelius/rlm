#!/bin/bash
# E13 data (11:35 UTC): per-task route outcomes on the teaching side for the learned retrieval key.
# OOLONG teaching windows (train split): direct and library routes, 25 rows per length; event-log dev: direct.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
cd $E
for d in e13-oolong-teach-direct e13-oolong-teach-lib e13-dev-direct e13-teach-direct; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
$PY run_sweep.py --out $S/e13-dev-direct --policies direct --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
$PY run_sweep.py --out $S/e13-teach-direct --policies direct --ns 8,32,64,128,256,512 --seeds 16-23 --split teach --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
$PY oolong_eval.py --out $S/e13-oolong-teach-direct --split train --policies direct --lens 1024,2048,4096,8192,16384,32768 --per-cell 25 --concurrency 4 2>&1 | tail -n 1 &
$PY oolong_eval.py --out $S/e13-oolong-teach-lib --split train --policies rlm --harness skills --controller-model qwen3-4b --lens 1024,2048,4096,8192,16384,32768 --per-cell 25 --concurrency 4 2>&1 | tail -n 1 &
wait
echo "[$(date -u +%FT%TZ)] E13_DATA_DONE"
