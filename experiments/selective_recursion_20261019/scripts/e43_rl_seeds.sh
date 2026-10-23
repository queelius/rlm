#!/bin/bash
# E43: RL seed replications. RL results moved by about four held-out families between seeds (E41), and the
# headline round-four result (42 of 74) rests on one RL seed. A: round four's SFT adapter, round four's RL settings
# (code and helper families), fresh seed 6. B: round seven's SFT adapter, its RL settings, fresh seed 27. Each run's
# held-out evaluations queue themselves; 8008 is stopped for training and restarted at the end. gpu1 queue job.
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
log() { echo "[$(date -u +%FT%TZ)] $*"; }
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8008") && !index($0,"awk") {print $1}')
[ -n "$srv" ] && { kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5; }
log "8008 stopped"
cd $X
for spec in "r4:6:proposer-4b-r4:$H/adapters/proposer-4b-r4/final" "r7:27:proposer-4b-r7:$H/adapters/proposer-4b-r7/final"; do
  IFS=: read -r tag seed start init <<< "$spec"
  OUT=$H/e43-rl-$tag-s$seed; mkdir -p $OUT
  nohup bash $S/e26h_queue_evals.sh $OUT 4b$tag-s$seed > $S/e43_queue_evals_$tag.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $OUT --init-adapter $init --start-name $start --with-helper-families --sample-ids \
    --steps 40 --families-per-step 8 --group 4 --seed $seed > $OUT/run.log 2>&1 || log "RL $tag FAILED: $(tail -n 3 $OUT/run.log)"
  log "RL $tag finished: $(tail -n 1 $OUT/run.log)"
done
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-14B-AWQ --served-model-name qwen3-14b-awq --host 127.0.0.1 \
  --port 8008 --api-key local --max-model-len 24576 --gpu-memory-utilization 0.42 --max-num-seqs 8 --enable-prefix-caching --seed 0 \
  --kv-cache-dtype fp8 > $S/serve-8008-qwen3-14b-restart4.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8008/v1/models | grep -q qwen3-14b-awq; do
  kill -0 $sp 2>/dev/null || { log "8008 restart FAILED"; exit 1; }; sleep 5; done
log "8008 restarted; E43 DONE"
