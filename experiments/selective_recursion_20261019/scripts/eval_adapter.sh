#!/bin/bash
# Usage: eval_adapter.sh <adapter_dir> <name>   -> loads the adapter into vLLM as <name>, evaluates on
# the OOLONG held-out rows (all lengths) and the event-log dev split, into oolong-<name>/ and dev-<name>/.
set -u
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
A=$1; N=$2; S=/project/alex_phd/runs/selective-recursion-20261019; H=/home/atowell/research-runs/selective-recursion-20261019
mkdir -p $H/oolong-$N $H/dev-$N; ln -sfn $H/oolong-$N $S/oolong-$N; ln -sfn $H/dev-$N $S/dev-$N
curl -sS -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d "{\"lora_name\": \"$N\", \"lora_path\": \"$A\"}"; echo
../../.venv/bin/python oolong_eval.py --out $S/oolong-$N --policies rlm --controller-model $N --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 10 &
../../.venv/bin/python run_sweep.py --out $S/dev-$N --policies rlm --harness default --controller-model $N --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 8 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 3 &
wait
echo "EVAL_DONE $N"
