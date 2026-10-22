#!/bin/bash
# S1d (12:55 UTC): distil the untrained 4B's one-cell library calls (teaching splits of both families,
# 436 correct runs) into the 1.7B caller; evaluate like S1b (dev with library + note, four grown forms, OOLONG).
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
M=/project/alex_phd/research-cache/models/Qwen--Qwen3-1.7B
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
CUDA_VISIBLE_DEVICES=$G0 SR_MODEL=$M $TRAIN_PY sft_train.py --data $S/sft-data/onecell-4b-lib.jsonl --out $S/adapters/sft-17-onecell --epochs 3 --lr 2e-4 --lora-r 32 --max-seq-len 8000 --batch-tokens 12000 2>&1 | grep -v -iE "warn|Loading weights" | tail -n 3
[ -d $S/adapters/sft-17-onecell/final ] || { log "SFT-17-onecell failed"; exit 1; }
log "SFT-17-onecell trained"
pid=$(ps -eo pid,args | awk '/vllm serv[e] .*Qwen--Qwen3-1.7B/ {print $1; exit}'); [ -n "$pid" ] && kill $pid; sleep 25
cd $S/serving && CUDA_VISIBLE_DEVICES=$G0 PATH=/project/alex_phd/envs/vllm/bin:$PATH nohup /project/alex_phd/envs/vllm/bin/vllm serve $M --served-model-name qwen3-1.7b-nt --host 127.0.0.1 --port 8003 --api-key local --max-model-len 16384 --gpu-memory-utilization 0.18 --max-num-seqs 16 --enable-prefix-caching --seed 0 --default-chat-template-kwargs '{"enable_thinking": false}' --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules ctrl-17-lib=$S/adapters/sft-17-lib/final ctrl-17-onecell=$S/adapters/sft-17-onecell/final > $S/serving/vllm_gpu0_qwen3-1.7b-nt.log 2>&1 &
cd $E
U=http://127.0.0.1:8003/v1
sleep 30; until curl -s -m 10 $U/models -H "Authorization: Bearer local" | grep -q ctrl-17-onecell; do sleep 15; done
log "server ready with ctrl-17-onecell"
for d in s1d-17one-dev s1d-17one-heldout s1d-17one-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
$PY run_sweep.py --out $S/s1d-17one-dev --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-17-onecell --controller-base-url $U --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
$PY run_sweep.py --out $S/s1d-17one-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-17-onecell --controller-base-url $U --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
$PY oolong_eval.py --out $S/s1d-17one-oolong --policies rlm --harness skills --controller-model ctrl-17-onecell --controller-base-url $U --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 4 2>&1 | tail -n 1 &
wait
log S1D_DONE
