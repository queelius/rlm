#!/bin/bash
# E26c with sampled token ids (idea-lab I-019): the round-two RL repeated with on-policy token ids, from the round-two
# SFT adapter. GPU1: stop 8007 (idle), smoke step, 40 steps, restart 8007.
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
INIT=$H/adapters/proposer-4b-8bcode-r2/final
log() { echo "[$(date -u +%FT%TZ)] $*"; }
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8007") && !index($0,"awk") {print $1}')
[ -n "$srv" ] && { kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5; }
log "8007 stopped"
cd $X; mkdir -p $H/e26c-rl-ids-smoke $H/e26c-proposer-rl-ids
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $H/e26c-rl-ids-smoke --init-adapter $INIT --start-name proposer-4b-8bcode-r2 \
  --sample-ids --steps 1 --families-per-step 2 --group 2 --seed 98 > $H/e26c-rl-ids-smoke/run.log 2>&1 || { log "SMOKE FAILED"; tail -n 25 $H/e26c-rl-ids-smoke/run.log; exit 1; }
log "smoke ok: $(grep '^step' $H/e26c-rl-ids-smoke/run.log)"
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $H/e26c-proposer-rl-ids --init-adapter $INIT --start-name proposer-4b-8bcode-r2 \
  --sample-ids --steps 40 --families-per-step 8 --group 4 --seed 4 > $H/e26c-proposer-rl-ids/run.log 2>&1 || log "RL FAILED: $(tail -n 3 $H/e26c-proposer-rl-ids/run.log)"
log "RL finished: $(tail -n 1 $H/e26c-proposer-rl-ids/run.log)"
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-4B --served-model-name qwen3-4b-hybrid --host 127.0.0.1 \
  --port 8007 --api-key local --dtype bfloat16 --max-model-len 24576 --gpu-memory-utilization 0.38 --max-num-seqs 8 \
  --enable-prefix-caching --seed 0 > $S/serve-8007-hybrid4b-restart7.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8007/v1/models | grep -q qwen3-4b-hybrid; do
  kill -0 $sp 2>/dev/null || { log "8007 restart FAILED"; exit 1; }; sleep 5; done
log "8007 restarted; E26L DONE"
