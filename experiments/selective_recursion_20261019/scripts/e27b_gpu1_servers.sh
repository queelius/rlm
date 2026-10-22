#!/bin/bash
# Retry of e27 (which failed: vLLM needs `ninja` on PATH at engine startup and the script did not put the
# vLLM environment's bin directory on PATH). GPU1: 4B-Instruct helper at 0.50 on 8001, hybrid Qwen3-4B at
# 0.38 on 8007. Fails fast if a server process exits before it is ready.
S=/project/alex_phd/runs/selective-recursion-20261019
E=/project/alex_phd/envs/vllm
M=/project/alex_phd/research-cache/models
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=MIG-43a3375a-1942-592d-accf-d0884164dcb8 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
log() { echo "[$(date -u +%FT%TZ)] $*"; }
ready() {  # port model pid logfile
  until curl -s -H "Authorization: Bearer local" http://127.0.0.1:$1/v1/models | grep -q "\"$2\""; do
    if ! kill -0 $3 2>/dev/null; then log "$1 FAILED to start"; grep -m 3 -E "Error|error" $4 | tail -n 3; exit 1; fi
    sleep 5
  done; log "$1 up"; }
nohup $E/bin/vllm serve $M/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554 --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8001 --api-key local --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.50 \
  --max-num-seqs 64 --enable-prefix-caching --seed 0 --enable-lora --max-lora-rank 64 --max-loras 2 > $S/serve-8001-resplit.log 2>&1 &
ready 8001 qwen3-4b $! $S/serve-8001-resplit.log
nohup $E/bin/vllm serve $M/Qwen--Qwen3-4B --served-model-name qwen3-4b-hybrid --host 127.0.0.1 --port 8007 --api-key local \
  --dtype bfloat16 --max-model-len 24576 --gpu-memory-utilization 0.38 --max-num-seqs 8 --enable-prefix-caching --seed 0 \
  > $S/serve-8007-hybrid4b.log 2>&1 &
ready 8007 qwen3-4b-hybrid $! $S/serve-8007-hybrid4b.log
for spec in "8001 qwen3-4b" "8007 qwen3-4b-hybrid"; do set -- $spec
  out=$(curl -s http://127.0.0.1:$1/v1/responses -H "Authorization: Bearer local" -H "Content-Type: application/json" \
    -d "{\"model\": \"$2\", \"input\": \"What is 17*23? Reply with the number only.\", \"max_output_tokens\": 1500}")
  echo "$out" | grep -q 391 && log "canary $1 ok" || log "canary $1 FAILED: ${out:0:300}"
done
log "RESPLIT DONE"
