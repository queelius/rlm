#!/bin/bash
# E26c replication and ablation: RL seed 1 from the round-two SFT adapter, and RL from the untrained 4B (no SFT).
# Same GPU1 arrangement as e26e (pause the hybrid-4B runs, stop 8007, train next to 8001, restart 8007).
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for job in e26-disc-4bh-heldout e26-disc-4bh-train; do
  if [ -f $H/$job/PID ]; then pid=$(cat $H/$job/PID); kill -TERM -- -$pid 2>/dev/null
    echo "[$(date -u +%FT%TZ)] paused (process group $pid stopped) for the E26c RL replication and ablation; resumes when requeued" >> $H/$job/NOTE.txt; fi
done
sleep 5
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8007") && !index($0,"awk") {print $1}')
[ -n "$srv" ] && { kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5; }
log "hybrid-4B runs paused, 8007 stopped"
cd $X
mkdir -p $H/e26c-proposer-rl-s1 $H/e26c-proposer-rl-base
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $H/e26c-proposer-rl-s1 --init-adapter $H/adapters/proposer-4b-8bcode-r2/final \
  --steps 40 --families-per-step 8 --group 4 --seed 1 > $H/e26c-proposer-rl-s1/run.log 2>&1 || log "RL seed 1 FAILED: $(tail -n 3 $H/e26c-proposer-rl-s1/run.log)"
log "RL seed 1 finished: $(tail -n 1 $H/e26c-proposer-rl-s1/run.log)"
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $H/e26c-proposer-rl-base --init-adapter none --start-name qwen3-4b \
  --steps 40 --families-per-step 8 --group 4 --seed 2 > $H/e26c-proposer-rl-base/run.log 2>&1 || log "RL from base FAILED: $(tail -n 3 $H/e26c-proposer-rl-base/run.log)"
log "RL from base finished: $(tail -n 1 $H/e26c-proposer-rl-base/run.log)"
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-4B --served-model-name qwen3-4b-hybrid --host 127.0.0.1 \
  --port 8007 --api-key local --dtype bfloat16 --max-model-len 24576 --gpu-memory-utilization 0.38 --max-num-seqs 8 \
  --enable-prefix-caching --seed 0 > $S/serve-8007-hybrid4b-restart3.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8007/v1/models | grep -q qwen3-4b-hybrid; do
  kill -0 $sp 2>/dev/null || { log "8007 restart FAILED"; exit 1; }; sleep 5; done
log "8007 restarted; E26G DONE"
