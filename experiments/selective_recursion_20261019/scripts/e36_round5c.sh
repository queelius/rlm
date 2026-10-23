#!/bin/bash
# E36: proposer round five = round four's recipe with more helper-family teacher data: 40 new helper training families
# (seeds 2-5) from the thinking 8B and 20 (seeds 6-7) from the thinking 14B, both with the worked helper example
# stripped from the saved prompts by build_proposer_sft. SFT and RL settings are round four's (RL families unchanged),
# so the comparison with round four isolates the extra helper-family pairs (RL seed 15, not 5: adapter names are
# proposer-rl-<seed>-<step> and round four's proposer-rl-5-40 is still loaded on 8001). Runs as a gpu1 queue job.
# Copy c also waits for the 14B OOLONG group bootstraps before stopping 8008.
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
NEW="e36-8b-exemplar-helper-train-s23 e36-8b-exemplar-helper-train-s45 e36-14b-exemplar-helper-train-s67"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for d in $NEW; do until [ -f $S/$d/bench.jsonl ]; do sleep 60; done; done
log "helper-family data runs finished"
# the 14B OOLONG group bootstraps (E35c) use 8008, which is stopped below: wait for them, at most until 10:00 UTC
oolong14_done() { for g in counting timeline user; do [ -f $S/e35c-boot-14b-oolong-$g-s0/summary.json ] || return 1; done; }
until oolong14_done || [ "$(date -u +%Y%m%d%H%M)" -ge 202610231000 ]; do sleep 60; done
log "14B OOLONG group bootstraps finished (or 10:00 UTC reached)"
cd $X
runs="$S/e26g-8b-exemplar-helper-train-b0 $S/e26g-8b-exemplar-helper-train-b1 $S/e26e-disc-8b-helper-train $S/e26e-disc-4bh-helper-train"
for d in $NEW; do runs="$runs $S/$d"; done
$PY build_proposer_sft.py --runs $runs --threshold 0.7 --out $H/sft-data/proposer-helper-07-r5.jsonl
cat $H/sft-data/proposer-8b-code-r2.jsonl $H/sft-data/proposer-helper-07-r5.jsonl > $H/sft-data/proposer-r5.jsonl
log "round-five SFT pairs: $(wc -l < $H/sft-data/proposer-r5.jsonl) (helper families: $(wc -l < $H/sft-data/proposer-helper-07-r5.jsonl))"
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8008") && !index($0,"awk") {print $1}')
[ -n "$srv" ] && { kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5; }
log "8008 stopped"
A=$H/adapters/proposer-4b-r5; mkdir -p $A
CUDA_VISIBLE_DEVICES=$G1 $TE sft_train.py --data $H/sft-data/proposer-r5.jsonl --out $A --epochs 3 --lr 1e-4 --lora-r 32 \
  --max-seq-len 12000 --batch-tokens 16000 > $A/train.log 2>&1 || { log "SFT FAILED"; tail -n 20 $A/train.log; exit 1; }
r=$(curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" \
  -d "{\"lora_name\": \"proposer-4b-r5\", \"lora_path\": \"$A/final\"}"); log "SFT done; load: $r"
OUT=$H/e36-proposer-rl-r5; mkdir -p $OUT
nohup bash $S/e26h_queue_evals.sh $OUT 4brl5 > $S/e36_queue_evals.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $OUT --init-adapter $A/final --start-name proposer-4b-r5 --with-helper-families --sample-ids \
  --steps 40 --families-per-step 8 --group 4 --seed 15 > $OUT/run.log 2>&1 || log "RL FAILED: $(tail -n 3 $OUT/run.log)"
log "RL finished: $(tail -n 1 $OUT/run.log)"
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-14B-AWQ --served-model-name qwen3-14b-awq --host 127.0.0.1 \
  --port 8008 --api-key local --max-model-len 24576 --gpu-memory-utilization 0.42 --max-num-seqs 8 --enable-prefix-caching --seed 0 \
  --kv-cache-dtype fp8 > $S/serve-8008-qwen3-14b-restart1.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8008/v1/models | grep -q qwen3-14b-awq; do
  kill -0 $sp 2>/dev/null || { log "8008 restart FAILED"; exit 1; }; sleep 5; done
log "8008 restarted; E36 DONE"
