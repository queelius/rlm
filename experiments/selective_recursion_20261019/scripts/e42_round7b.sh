#!/bin/bash
# E42: proposer round seven = round five's SFT data plus helper libraries for three new helper question types
# (E40: semantic_distinct, semantic_last_city, semantic_nth; 14B and 8B with the worked example, stripped), with the
# 68 code pairs included twice (E41: round five's residual code gap traced to its 51 percent code share), then RL with
# round four's settings on RL_FAMILIES (set below), seed 26. Evaluates the SFT-only adapter and the RL adapter
# (MRCR seed, two held-out discovery seeds each). Waits for the 14B data run, and for the 8B one until 17:00 UTC
# (a partial run contributes its finished families). 8008 is stopped for training and restarted. gpu1 queue job.
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
RL_FAMILIES="${RL_FAMILIES:-}"   # empty: code-only training families; "--with-helper-families": code and helper
log() { echo "[$(date -u +%FT%TZ)] $*"; }
until [ -f $S/e40-14b-exemplar-helper-extra/bench.jsonl ]; do sleep 60; done
# copy b: also the 14B on new-question seeds 2-3 (e40b); both partial runs contribute what finished by 17:00 UTC
until { [ -f $S/e40-8b-exemplar-helper-extra/bench.jsonl ] && [ -f $S/e40b-14b-exemplar-helper-extra-s23/bench.jsonl ]; } \
      || [ "$(date -u +%Y%m%d%H%M)" -ge 202610231700 ]; do sleep 60; done
log "E40 data runs finished (or 17:00 UTC reached)"
cd $X
runs="$S/e26g-8b-exemplar-helper-train-b0 $S/e26g-8b-exemplar-helper-train-b1 $S/e26e-disc-8b-helper-train $S/e26e-disc-4bh-helper-train"
runs="$runs $S/e36-8b-exemplar-helper-train-s23 $S/e36-8b-exemplar-helper-train-s45 $S/e36-14b-exemplar-helper-train-s67"
runs="$runs $S/e40-14b-exemplar-helper-extra $S/e40-8b-exemplar-helper-extra $S/e40b-14b-exemplar-helper-extra-s23"
$PY build_proposer_sft.py --runs $runs --threshold 0.7 --out $H/sft-data/proposer-helper-07-r7.jsonl
cat $H/sft-data/proposer-8b-code-r2.jsonl $H/sft-data/proposer-8b-code-r2.jsonl $H/sft-data/proposer-helper-07-r7.jsonl > $H/sft-data/proposer-r7.jsonl
log "round-seven SFT pairs: $(wc -l < $H/sft-data/proposer-r7.jsonl) (helper families: $(wc -l < $H/sft-data/proposer-helper-07-r7.jsonl))"
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8008") && !index($0,"awk") {print $1}')
[ -n "$srv" ] && { kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5; }
log "8008 stopped"
A=$H/adapters/proposer-4b-r7; mkdir -p $A
CUDA_VISIBLE_DEVICES=$G1 $TE sft_train.py --data $H/sft-data/proposer-r7.jsonl --out $A --epochs 3 --lr 1e-4 --lora-r 32 \
  --max-seq-len 12000 --batch-tokens 16000 > $A/train.log 2>&1 || { log "SFT FAILED"; tail -n 20 $A/train.log; exit 1; }
r=$(curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" \
  -d "{\"lora_name\": \"proposer-4b-r7\", \"lora_path\": \"$A/final\"}"); log "SFT done; load: $r"
SO=$H/e42-sft-only-r7; mkdir -p $SO; echo proposer-4b-r7 > $SO/FINAL
bash $S/e26h_queue_evals.sh $SO 4bsft7 > $S/e42_queue_evals_sft.log 2>&1
OUT=$H/e42-rl-r7; mkdir -p $OUT
nohup bash $S/e26h_queue_evals.sh $OUT 4brl7 > $S/e42_queue_evals_rl.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $OUT --init-adapter $A/final --start-name proposer-4b-r7 $RL_FAMILIES --sample-ids \
  --steps 40 --families-per-step 8 --group 4 --seed 26 > $OUT/run.log 2>&1 || log "RL FAILED: $(tail -n 3 $OUT/run.log)"
log "RL finished: $(tail -n 1 $OUT/run.log)"
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-14B-AWQ --served-model-name qwen3-14b-awq --host 127.0.0.1 \
  --port 8008 --api-key local --max-model-len 24576 --gpu-memory-utilization 0.42 --max-num-seqs 8 --enable-prefix-caching --seed 0 \
  --kv-cache-dtype fp8 > $S/serve-8008-qwen3-14b-restart3.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8008/v1/models | grep -q qwen3-14b-awq; do
  kill -0 $sp 2>/dev/null || { log "8008 restart FAILED"; exit 1; }; sleep 5; done
log "8008 restarted; E42 DONE"
