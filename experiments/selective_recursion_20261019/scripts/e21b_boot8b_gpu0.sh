#!/bin/bash
# E21b: when the E18b window rerun ends, replace the RLM-Qwen3-8B server on GPU0 with Qwen3-8B-AWQ (thinking on)
# and bootstrap the MRCR library with it as proposer (helpers: 4B on GPU1).
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
until [ "$(pgrep -fc 'e18_official_rl[m].py')" -eq 0 ]; do sleep 60; done
log "E18b runners finished: $(cat $S/e18b-official-yarn/raw-shard*.jsonl | wc -l)/48 rows"
pid=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"rlm-qwen3-8b-v0") && !index($0,"awk") {print $1; exit}'); [ -n "$pid" ] && kill $pid; sleep 25
cd $S/serving && CUDA_VISIBLE_DEVICES=$G0 PATH=/project/alex_phd/envs/vllm/bin:$PATH nohup /project/alex_phd/envs/vllm/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-8B-AWQ --served-model-name qwen3-8b-awq --host 127.0.0.1 --port 8002 --api-key local --max-model-len 24576 --gpu-memory-utilization 0.45 --max-num-seqs 16 --enable-prefix-caching --seed 0 > $S/serving/vllm_gpu0_8b_awq.log 2>&1 &
cd $E; sleep 30
until curl -s -m 10 http://127.0.0.1:8002/v1/models -H "Authorization: Bearer local" | grep -q qwen3-8b-awq; do sleep 15; if grep -qE "ValueError|out of memory" $S/serving/vllm_gpu0_8b_awq.log; then log "8B server FAILED"; exit 1; fi; done
log "8B-AWQ served"
mkdir -p $H/e21-boot-8b && ln -sfn $H/e21-boot-8b $S/e21-boot-8b
$PY bootstrap_library.py --family mrcr --out $S/e21-boot-8b --model qwen3-8b-awq --base-url http://127.0.0.1:8002/v1 --max-output-tokens 16000 --candidates 8 --repair-rounds 2 --timeout 1200 2>&1 | tail -n 30
log E21B_DONE
