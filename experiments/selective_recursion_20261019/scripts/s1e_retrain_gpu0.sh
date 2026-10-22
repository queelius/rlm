#!/bin/bash
# S1e (13:10 UTC): retrain the small-caller adapters with the serving template (empty think block), then evaluate.
# 1.7B: lib-v2 (scripted three-cell library teacher, 651 pairs) and onecell-v2 (untrained 4B one-cell calls, 436 pairs);
# 0.6B: onecell-v2. Evaluations: event-log dev (library + note v4), four grown forms, OOLONG 72-row sample.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
M17=/project/alex_phd/research-cache/models/Qwen--Qwen3-1.7B
M06=/project/alex_phd/research-cache/models/Qwen--Qwen3-0.6B
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
train() {  # model data out epochs
  CUDA_VISIBLE_DEVICES=$G0 SR_NO_THINK=1 SR_MODEL=$1 $TRAIN_PY sft_train.py --data $2 --out $S/adapters/$3 --epochs $4 --lr 2e-4 --lora-r 32 --max-seq-len 8000 --batch-tokens 12000 2>&1 | grep -v -iE "warn|Loading weights" | tail -n 2
  [ -d $S/adapters/$3/final ] && log "$3 trained" || { log "$3 FAILED"; exit 1; }
}
serve() {  # model name port util adapters...
  local m=$1 n=$2 p=$3 u=$4; shift 4
  pid=$(ps -eo pid,args | awk -v pat="vllm serv[e] .*$(basename $m)\$|vllm serv[e] .*$(basename $m) " '$0 ~ pat {print $1; exit}'); [ -n "$pid" ] && kill $pid; sleep 25
  cd $S/serving && CUDA_VISIBLE_DEVICES=$G0 PATH=/project/alex_phd/envs/vllm/bin:$PATH nohup /project/alex_phd/envs/vllm/bin/vllm serve $m --served-model-name $n --host 127.0.0.1 --port $p --api-key local --max-model-len 16384 --gpu-memory-utilization $u --max-num-seqs 16 --enable-prefix-caching --seed 0 --default-chat-template-kwargs '{"enable_thinking": false}' --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules "$@" > $S/serving/vllm_gpu0_$n.log 2>&1 &
  cd $E; sleep 30
  local last=${@: -1}; local name=${last%%=*}
  until curl -s -m 10 http://127.0.0.1:$p/v1/models -H "Authorization: Bearer local" | grep -q "\"$name\""; do sleep 15; done
  log "$n served with $*"
}
evalset() {  # tag model port
  local t=$1 m=$2 U=http://127.0.0.1:$3/v1
  for d in $t-dev $t-heldout $t-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
  $PY run_sweep.py --out $S/$t-dev --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model $m --controller-base-url $U --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
  $PY run_sweep.py --out $S/$t-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model $m --controller-base-url $U --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
  $PY oolong_eval.py --out $S/$t-oolong --policies rlm --harness skills --controller-model $m --controller-base-url $U --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 4 2>&1 | tail -n 1 &
}
train $M17 $S/sft-data/onecell-4b-lib.jsonl sft-17-onecell-v2 3
train $M17 $S/teach-lib-009/sft.jsonl sft-17-lib-v2 2
train $M06 $S/sft-data/onecell-4b-lib.jsonl sft-06-onecell-v2 3
serve $M17 qwen3-1.7b-nt 8003 0.18 ctrl-17-lib-v2=$S/adapters/sft-17-lib-v2/final ctrl-17-onecell-v2=$S/adapters/sft-17-onecell-v2/final
evalset s1e-17one ctrl-17-onecell-v2 8003
evalset s1e-17lib ctrl-17-lib-v2 8003
until grep -q "S1 0.6b done" $S/s1_0.6b.log; do sleep 60; done
serve $M06 qwen3-0.6b-nt 8004 0.12 ctrl-06-onecell-v2=$S/adapters/sft-06-onecell-v2/final
evalset s1e-06one ctrl-06-onecell-v2 8004
wait
log S1E_DONE
