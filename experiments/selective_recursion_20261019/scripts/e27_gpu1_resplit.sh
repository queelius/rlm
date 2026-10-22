#!/bin/bash
# Re-split GPU1 (MIG-43a3375a): 4B-Instruct helper at 0.50 on 8001 (args otherwise unchanged) and the
# hybrid-thinking Qwen3-4B at 0.38 on 8007, for the E21b proposer ladder (reasoning vs scale).
S=/project/alex_phd/runs/selective-recursion-20261019
V=/project/alex_phd/envs/vllm/bin/vllm
M=/project/alex_phd/research-cache/models
export CUDA_VISIBLE_DEVICES=MIG-43a3375a-1942-592d-accf-d0884164dcb8 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for pid in 1828134 1844151; do while kill -0 $pid 2>/dev/null; do sleep 20; done; done
log "GPU1 evaluations finished; stopping 8001 (pid 1802081)"
kill 1802081; while kill -0 1802081 2>/dev/null; do sleep 2; done; sleep 5
nohup $V serve $M/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554 --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8001 --api-key local --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.50 \
  --max-num-seqs 64 --enable-prefix-caching --seed 0 --enable-lora --max-lora-rank 64 --max-loras 2 > $S/serve-8001-resplit.log 2>&1 &
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8001/v1/models | grep -q qwen3-4b; do sleep 5; done
log "8001 up"
nohup $V serve $M/Qwen--Qwen3-4B --served-model-name qwen3-4b-hybrid --host 127.0.0.1 --port 8007 --api-key local \
  --dtype bfloat16 --max-model-len 24576 --gpu-memory-utilization 0.38 --max-num-seqs 8 --enable-prefix-caching --seed 0 \
  > $S/serve-8007-hybrid4b.log 2>&1 &
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8007/v1/models | grep -q qwen3-4b-hybrid; do
  sleep 5; if ! pgrep -f "port 800[7]" > /dev/null; then log "8007 FAILED to start"; tail -n 20 $S/serve-8007-hybrid4b.log; exit 1; fi; done
log "8007 up"
for spec in "8001 qwen3-4b" "8007 qwen3-4b-hybrid"; do set -- $spec
  out=$(curl -s http://127.0.0.1:$1/v1/responses -H "Authorization: Bearer local" -H "Content-Type: application/json" \
    -d "{\"model\": \"$2\", \"input\": \"What is 17*23? Reply with the number only.\", \"max_output_tokens\": 1500}" | head -c 600)
  log "canary $1: $(echo "$out" | grep -o '391' | head -n 1 || echo FAILED) ${out:0:120}"
done
log "RESPLIT DONE"
