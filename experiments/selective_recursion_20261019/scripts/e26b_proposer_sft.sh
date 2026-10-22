#!/bin/bash
# E26b: distil library writing from the 8B into the non-thinking 4B proposer (code-only targets, hindsight
# relabelled). GPU1: pause the hybrid-4B server (8007) after its held-out discovery run, train a LoRA on the
# freed memory, load it into 8001, check it answers, then restart 8007.
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
A=$H/adapters/proposer-4b-8bcode
log() { echo "[$(date -u +%FT%TZ)] $*"; }
until [ -f $S/e26-disc-4bh-heldout/bench.jsonl ]; do sleep 30; done
log "hybrid-4B held-out discovery finished"
pid=$(cat $H/e26-disc-4bh-train/PID); kill -TERM -- -$pid 2>/dev/null; sleep 5
echo "[$(date -u +%FT%TZ)] paused (process group stopped) to free GPU1 for the E26b proposer SFT; families with summary.json are kept and the run resumes by requeueing the same spec" >> $H/e26-disc-4bh-train/NOTE.txt
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8007") && !index($0,"awk") {print $1}')
kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5
log "8007 stopped"
cd $X
$PY build_proposer_sft.py --runs $S/e26-disc-8b-train $S/e26-disc-8b-train-s12 --out $H/sft-data/proposer-8b-code.jsonl || { log "FAILED building pairs"; exit 1; }
mkdir -p $A
CUDA_VISIBLE_DEVICES=$G1 $TE sft_train.py --data $H/sft-data/proposer-8b-code.jsonl --out $A --epochs 3 --lr 1e-4 \
  --lora-r 32 --max-seq-len 12000 --batch-tokens 16000 > $A/train.log 2>&1 || { log "FAILED training"; tail -n 20 $A/train.log; exit 1; }
log "training done: $(grep -v -iE 'warn|Loading' $A/train.log | tail -n 2 | tr '\n' ' ')"
ad=$(ls -d $A/final 2>/dev/null || ls -d $A/checkpoint* 2>/dev/null | tail -n 1 || echo $A)
[ -f $ad/adapter_config.json ] || ad=$A
r=$(curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" \
  -d "{\"lora_name\": \"proposer-4b-8bcode\", \"lora_path\": \"$ad\"}")
log "load adapter from $ad: $r"
out=$(curl -s http://127.0.0.1:8001/v1/responses -H "Authorization: Bearer local" -H "Content-Type: application/json" \
  -d '{"model": "proposer-4b-8bcode", "input": "Write a Python function add(a, b) that returns a + b. Reply with one ```python block.", "max_output_tokens": 300}')
echo "$out" | grep -q "def add" && log "adapter canary ok" || log "adapter canary FAILED: ${out:0:300}"
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-4B --served-model-name qwen3-4b-hybrid --host 127.0.0.1 \
  --port 8007 --api-key local --dtype bfloat16 --max-model-len 24576 --gpu-memory-utilization 0.38 --max-num-seqs 8 \
  --enable-prefix-caching --seed 0 > $S/serve-8007-hybrid4b-restart.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8007/v1/models | grep -q qwen3-4b-hybrid; do
  kill -0 $sp 2>/dev/null || { log "8007 restart FAILED"; exit 1; }; sleep 5; done
log "8007 restarted; E26B DONE"
