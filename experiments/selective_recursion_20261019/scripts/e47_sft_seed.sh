#!/bin/bash
# E47: is round four's spread from SFT as well as RL? Round four's SFT data and settings with SFT seed 1 (data order;
# the original used seed 0), then RL with round four's settings and its original RL seed 5. Compare with the original
# (SFT seed 0, RL seed 5: 42 of 74) and the RL-seed replications (E43, E45, E46). Held-out evaluations queue themselves.
# RL adapters are named proposer-rl-5-<step> as in round four; the original round-four adapter is not loaded on this
# server, and the evaluation runs are tagged 4br4sft1-s5. gpu1 queue job.
set -u
H=/home/atowell/research-runs/selective-recursion-20261019
S=/project/alex_phd/runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $X
A=$H/adapters/proposer-4b-r4-sft1; mkdir -p $A
CUDA_VISIBLE_DEVICES=$G1 $TE sft_train.py --data $H/sft-data/proposer-r4.jsonl --out $A --epochs 3 --lr 1e-4 --lora-r 32 \
  --max-seq-len 12000 --batch-tokens 16000 --seed 1 > $A/train.log 2>&1 || { log "SFT FAILED"; tail -n 20 $A/train.log; exit 1; }
r=$(curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" \
  -d "{\"lora_name\": \"proposer-4b-r4-sft1\", \"lora_path\": \"$A/final\"}"); log "SFT done; load: $r"
curl -s -H "Authorization: Bearer local" http://127.0.0.1:8001/v1/models | grep -q '"proposer-rl-5-' && { log "a proposer-rl-5-* adapter is already loaded; refusing to collide"; exit 1; }
OUT=$H/e47-rl-r4sft1-s5; mkdir -p $OUT
nohup bash $S/e26h_queue_evals.sh $OUT 4br4sft1-s5 > $S/e47_queue_evals.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $OUT --init-adapter $A/final --start-name proposer-4b-r4-sft1 --with-helper-families \
  --sample-ids --steps 40 --families-per-step 8 --group 4 --seed 5 > $OUT/run.log 2>&1 || log "RL FAILED: $(tail -n 3 $OUT/run.log)"
log "RL finished: $(tail -n 1 $OUT/run.log); E47 DONE"
