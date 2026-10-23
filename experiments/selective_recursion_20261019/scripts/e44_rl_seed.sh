#!/bin/bash
# E44: second RL seed for the E41 B recipe (round five's SFT adapter, mixed code and helper RL, round four's settings),
# fresh seed 17, so the best balanced result (36 of 74, helper mean best 0.46) rests on two RL seeds. No 14B server is
# running on GPU1 at this point, so nothing is stopped. Held-out evaluations queue themselves. gpu1 queue job.
set -u
H=/home/atowell/research-runs/selective-recursion-20261019
S=/project/alex_phd/runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
log() { echo "[$(date -u +%FT%TZ)] $*"; }
curl -s -H "Authorization: Bearer local" http://127.0.0.1:8001/v1/models | grep -q '"proposer-4b-r5"' || {
  r=$(curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" \
    -d "{\"lora_name\": \"proposer-4b-r5\", \"lora_path\": \"$H/adapters/proposer-4b-r5/final\"}"); log "loaded start adapter: $r"; }
cd $X
OUT=$H/e44-rl-r5sft-mixed-s17; mkdir -p $OUT
nohup bash $S/e26h_queue_evals.sh $OUT 4br5b-s17 > $S/e44_queue_evals.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $OUT --init-adapter $H/adapters/proposer-4b-r5/final --start-name proposer-4b-r5 \
  --with-helper-families --sample-ids --steps 40 --families-per-step 8 --group 4 --seed 17 > $OUT/run.log 2>&1 || log "RL FAILED: $(tail -n 3 $OUT/run.log)"
log "RL finished: $(tail -n 1 $OUT/run.log); E44 DONE"
