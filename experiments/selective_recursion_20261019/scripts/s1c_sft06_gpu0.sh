#!/bin/bash
# S1c (10:45 UTC): teach the 0.6B caller to call the library: LoRA SFT on teach-lib-009 (651 library-calling
# pairs from the scripted library caller); after the S1 1.7B evaluations finish, restart the 1.7B server with
# LoRA and evaluate the adapter on event-log dev (library + note), the four grown forms, and the OOLONG sample.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
M=/project/alex_phd/research-cache/models/Qwen--Qwen3-0.6B
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
CUDA_VISIBLE_DEVICES=$G0 SR_MODEL=$M $TRAIN_PY sft_train.py --data $S/teach-lib-009/sft.jsonl --out $S/adapters/sft-06-lib --epochs 2 --lr 2e-4 --lora-r 32 --max-seq-len 8000 --batch-tokens 12000 2>&1 | grep -v -iE "warn|Loading weights" | tail -n 5
[ -d $S/adapters/sft-06-lib/final ] || { log "SFT-06 failed"; exit 1; }
log "SFT-06 trained"
until grep -q "S1 0.6b done" $S/s1_0.6b.log; do sleep 60; done
pid=$(ps -eo pid,args | awk '/vllm serv[e] .*Qwen--Qwen3-0.6B/ {print $1; exit}'); [ -n "$pid" ] && kill $pid; sleep 20
cd $S/serving && CUDA_VISIBLE_DEVICES=$G0 PATH=/project/alex_phd/envs/vllm/bin:$PATH nohup /project/alex_phd/envs/vllm/bin/vllm serve $M --served-model-name qwen3-0.6b-nt --host 127.0.0.1 --port 8004 --api-key local --max-model-len 16384 --gpu-memory-utilization 0.12 --max-num-seqs 16 --enable-prefix-caching --seed 0 --default-chat-template-kwargs '{"enable_thinking": false}' --enable-lora --max-lora-rank 64 --lora-modules ctrl-06-lib=$S/adapters/sft-06-lib/final > $S/serving/vllm_gpu0_qwen3-0.6b-nt.log 2>&1 &
cd $E
sleep 30; until curl -s -m 10 http://127.0.0.1:8004/v1/models -H "Authorization: Bearer local" | grep -q ctrl-06-lib; do sleep 15; done
U=http://127.0.0.1:8004/v1
for d in s1c-06lib-dev s1c-06lib-heldout s1c-06lib-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
$PY run_sweep.py --out $S/s1c-06lib-dev --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-06-lib --controller-base-url $U --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1
$PY run_sweep.py --out $S/s1c-06lib-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-06-lib --controller-base-url $U --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1
$PY oolong_eval.py --out $S/s1c-06lib-oolong --policies rlm --harness skills --controller-model ctrl-06-lib --controller-base-url $U --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 4 2>&1 | tail -n 1
log S1C_DONE
