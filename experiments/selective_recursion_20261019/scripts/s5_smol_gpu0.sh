#!/bin/bash
# S5 (19:05 UTC): how small can the caller be? SmolLM2-360M-Instruct (Llama architecture, 8k window), untrained
# and LoRA-tuned on the same 436 one-cell traces; event-log dev, four grown forms, OOLONG-72; helpers 4B on GPU1.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
M=/project/alex_phd/research-cache/models/HuggingFaceTB--SmolLM2-360M-Instruct
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
CUDA_VISIBLE_DEVICES=$G0 SR_MODEL=$M $TE sft_train.py --data $S/sft-data/onecell-4b-lib.jsonl --out $S/adapters/sft-smol360-onecell --epochs 3 --lr 3e-4 --lora-r 32 --max-seq-len 7000 --batch-tokens 8000 2>&1 | grep -v -iE "warn|Loading weights" | tail -n 2
[ -d $S/adapters/sft-smol360-onecell/final ] && log "smol adapter trained" || { log "smol adapter FAILED"; exit 1; }
cd $S/serving && CUDA_VISIBLE_DEVICES=$G0 PATH=/project/alex_phd/envs/vllm/bin:$PATH nohup /project/alex_phd/envs/vllm/bin/vllm serve $M --served-model-name smol360 --host 127.0.0.1 --port 8008 --api-key local --max-model-len 8192 --gpu-memory-utilization 0.1 --max-num-seqs 16 --enable-prefix-caching --seed 0 --enable-lora --max-lora-rank 64 --lora-modules ctrl-smol360-onecell=$S/adapters/sft-smol360-onecell/final > $S/serving/vllm_gpu0_smol360.log 2>&1 &
cd $E; sleep 30
until curl -s -m 10 http://127.0.0.1:8008/v1/models -H "Authorization: Bearer local" | grep -q ctrl-smol360-onecell; do sleep 15; if grep -qE "ValueError|out of memory" $S/serving/vllm_gpu0_smol360.log; then log "smol server FAILED"; exit 1; fi; done
log "smol served"
U=http://127.0.0.1:8008/v1
for t in s5-smol s5-smolone; do for d in $t-dev $t-heldout $t-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done; done
ev() { local t=$1 m=$2
  $PY run_sweep.py --out $S/$t-dev --policies rlm --harness skills --stats-file $S/stats-teach-v5.json --controller-model $m --controller-base-url $U --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 > /dev/null 2>&1 &
  $PY run_sweep.py --out $S/$t-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v5.json --controller-model $m --controller-base-url $U --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 > /dev/null 2>&1 &
  $PY oolong_eval.py --out $S/$t-oolong --policies rlm --harness skills --stats-file $S/stats-teach-v5.json --controller-model $m --controller-base-url $U --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 4 > /dev/null 2>&1 &
}
ev s5-smol smol360
ev s5-smolone ctrl-smol360-onecell
log "S5 evaluations launched"
