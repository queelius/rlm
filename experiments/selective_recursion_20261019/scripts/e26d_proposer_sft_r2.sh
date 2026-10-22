#!/bin/bash
# E26d, second round (adds the 8B's chat-format training libraries): distil library writing from the
# 8B into the non-thinking 4B proposer (code-only targets, hindsight relabelled). GPU1: pause both hybrid-4B
# discovery runs and their server (8007), train a LoRA on the freed memory, load it into 8001, check it answers,
# restart 8007. The paused runs resume (finished families kept, partial ones continue from saved candidates)
# when their queue specs are added again.
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
A=$H/adapters/proposer-4b-8bcode-r2
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for job in e26-disc-4bh-heldout e26-disc-4bh-train; do
  pid=$(cat $H/$job/PID)
  kill -TERM -- -$pid 2>/dev/null
  echo "[$(date -u +%FT%TZ)] paused (process group $pid stopped) to free GPU1 for the proposer SFT; resumes when requeued" >> $H/$job/NOTE.txt
done
sleep 5
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8007") && !index($0,"awk") {print $1}')
kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5
log "hybrid-4B runs paused, 8007 stopped"
cd $X
$PY build_proposer_sft.py --runs $S/e26-disc-8b-train $S/e26-disc-8b-train-s12 $S/e26-disc-8b-train-chat --out $H/sft-data/proposer-8b-code-r2.jsonl || { log "FAILED building pairs"; exit 1; }
mkdir -p $A
CUDA_VISIBLE_DEVICES=$G1 $TE sft_train.py --data $H/sft-data/proposer-8b-code-r2.jsonl --out $A --epochs 3 --lr 1e-4 \
  --lora-r 32 --max-seq-len 12000 --batch-tokens 16000 > $A/train.log 2>&1 || { log "FAILED training"; tail -n 20 $A/train.log; exit 1; }
log "training done: $(grep -v -iE 'warn|Loading' $A/train.log | tail -n 2 | tr '\n' ' ')"
r=$(curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" \
  -d "{\"lora_name\": \"proposer-4b-8bcode-r2\", \"lora_path\": \"$A/final\"}")
log "load adapter: $r"
out=$(curl -s http://127.0.0.1:8001/v1/responses -H "Authorization: Bearer local" -H "Content-Type: application/json" \
  -d '{"model": "proposer-4b-8bcode-r2", "input": "Write a Python function add(a, b) that returns a + b. Reply with one ```python block.", "max_output_tokens": 300}')
echo "$out" | grep -q "def add" && log "adapter canary ok" || log "adapter canary FAILED: ${out:0:300}"
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-4B --served-model-name qwen3-4b-hybrid --host 127.0.0.1 \
  --port 8007 --api-key local --dtype bfloat16 --max-model-len 24576 --gpu-memory-utilization 0.38 --max-num-seqs 8 \
  --enable-prefix-caching --seed 0 > $S/serve-8007-hybrid4b-restart.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8007/v1/models | grep -q qwen3-4b-hybrid; do
  kill -0 $sp 2>/dev/null || { log "8007 restart FAILED"; exit 1; }; sleep 5; done
log "8007 restarted; E26D DONE"
