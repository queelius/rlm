#!/bin/bash
# E26e: round three of proposer training with helper-dependent families. Waits for both helper-family data runs,
# builds SFT data (round-two pairs + helper-family libraries at teaching score >= 0.7), trains a fresh LoRA, then RL
# (40 steps over code-only and helper families) from it; GPU1 as before (hybrid-4B server paused, restarted at the end).
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
log() { echo "[$(date -u +%FT%TZ)] $*"; }
until [ -f $S/e26e-disc-8b-helper-train/bench.jsonl ] && [ -f $S/e26e-disc-4bh-helper-train/bench.jsonl ]; do sleep 60; done
log "helper-family data runs finished"
cd $X
$PY build_proposer_sft.py --runs $S/e26e-disc-8b-helper-train $S/e26e-disc-4bh-helper-train --threshold 0.7 --out $H/sft-data/proposer-helper-07.jsonl
cat $H/sft-data/proposer-8b-code-r2.jsonl $H/sft-data/proposer-helper-07.jsonl > $H/sft-data/proposer-r3.jsonl
log "round-three SFT pairs: $(wc -l < $H/sft-data/proposer-r3.jsonl) (helper families: $(wc -l < $H/sft-data/proposer-helper-07.jsonl))"
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8007") && !index($0,"awk") {print $1}')
[ -n "$srv" ] && { kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5; }
log "8007 stopped"
A=$H/adapters/proposer-4b-r3; mkdir -p $A
CUDA_VISIBLE_DEVICES=$G1 $TE sft_train.py --data $H/sft-data/proposer-r3.jsonl --out $A --epochs 3 --lr 1e-4 --lora-r 32 \
  --max-seq-len 12000 --batch-tokens 16000 > $A/train.log 2>&1 || { log "SFT FAILED"; tail -n 20 $A/train.log; exit 1; }
r=$(curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" \
  -d "{\"lora_name\": \"proposer-4b-r3\", \"lora_path\": \"$A/final\"}"); log "SFT done; load: $r"
OUT=$H/e26e-proposer-rl-r3; mkdir -p $OUT
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $OUT --init-adapter $A/final --start-name proposer-4b-r3 --with-helper-families \
  --steps 40 --families-per-step 8 --group 4 --seed 3 > $OUT/run.log 2>&1 || log "RL FAILED: $(tail -n 3 $OUT/run.log)"
log "RL finished: $(tail -n 1 $OUT/run.log)"
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-4B --served-model-name qwen3-4b-hybrid --host 127.0.0.1 \
  --port 8007 --api-key local --dtype bfloat16 --max-model-len 24576 --gpu-memory-utilization 0.38 --max-num-seqs 8 \
  --enable-prefix-caching --seed 0 > $S/serve-8007-hybrid4b-restart6.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8007/v1/models | grep -q qwen3-4b-hybrid; do
  kill -0 $sp 2>/dev/null || { log "8007 restart FAILED"; exit 1; }; sleep 5; done
log "8007 restarted; E26K DONE"
