#!/bin/bash
# S2 (14:25 UTC): confirm the distilled small callers. After S1e: 293-row held-out OOLONG for the 0.6B and
# 1.7B one-cell adapters; a seed-1 retrain of the 0.6B one-cell adapter with dev, grown forms and OOLONG-72.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
M06=/project/alex_phd/research-cache/models/Qwen--Qwen3-0.6B
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
CUDA_VISIBLE_DEVICES=$G0 SR_NO_THINK=1 SR_MODEL=$M06 $TRAIN_PY sft_train.py --data $S/sft-data/onecell-4b-lib.jsonl --out $S/adapters/sft-06-onecell-v2-s1 --epochs 3 --lr 2e-4 --lora-r 32 --max-seq-len 8000 --batch-tokens 12000 --seed 1 2>&1 | grep -v -iE "warn|Loading weights" | tail -n 1
[ -d $S/adapters/sft-06-onecell-v2-s1/final ] && log "seed-1 adapter trained" || log "seed-1 FAILED"
until grep -q S1E_DONE $S/s1e_retrain_gpu0.log; do sleep 60; done
for d in s2-06one-oolong-full s2-17one-oolong-full; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
$PY oolong_eval.py --out $S/s2-06one-oolong-full --policies rlm --harness skills --controller-model ctrl-06-onecell-v2 --controller-base-url http://127.0.0.1:8004/v1 --lens 1024,2048,4096,8192,16384,32768 --per-cell 50 --concurrency 6 2>&1 | tail -n 1 &
$PY oolong_eval.py --out $S/s2-17one-oolong-full --policies rlm --harness skills --controller-model ctrl-17-onecell-v2 --controller-base-url http://127.0.0.1:8003/v1 --lens 1024,2048,4096,8192,16384,32768 --per-cell 50 --concurrency 6 2>&1 | tail -n 1 &
wait
log "full OOLONG done; loading seed-1 adapter"
pid=$(ps -eo pid,args | awk '/vllm serv[e] .*Qwen--Qwen3-0.6B / {print $1; exit}'); [ -n "$pid" ] && kill $pid; sleep 25
cd $S/serving && CUDA_VISIBLE_DEVICES=$G0 PATH=/project/alex_phd/envs/vllm/bin:$PATH nohup /project/alex_phd/envs/vllm/bin/vllm serve $M06 --served-model-name qwen3-0.6b-nt --host 127.0.0.1 --port 8004 --api-key local --max-model-len 16384 --gpu-memory-utilization 0.12 --max-num-seqs 16 --enable-prefix-caching --seed 0 --default-chat-template-kwargs '{"enable_thinking": false}' --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules ctrl-06-onecell-v2=$S/adapters/sft-06-onecell-v2/final ctrl-06-onecell-v2-s1=$S/adapters/sft-06-onecell-v2-s1/final > $S/serving/vllm_gpu0_qwen3-0.6b-nt.log 2>&1 &
cd $E; sleep 30
until curl -s -m 10 http://127.0.0.1:8004/v1/models -H "Authorization: Bearer local" | grep -q ctrl-06-onecell-v2-s1; do sleep 15; done
log "0.6B served with the seed-1 adapter"
U=http://127.0.0.1:8004/v1
for d in s2-06one-s1-dev s2-06one-s1-heldout s2-06one-s1-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
$PY run_sweep.py --out $S/s2-06one-s1-dev --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-06-onecell-v2-s1 --controller-base-url $U --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
$PY run_sweep.py --out $S/s2-06one-s1-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-06-onecell-v2-s1 --controller-base-url $U --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
$PY oolong_eval.py --out $S/s2-06one-s1-oolong --policies rlm --harness skills --controller-model ctrl-06-onecell-v2-s1 --controller-base-url $U --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 4 2>&1 | tail -n 1 &
wait
log S2_DONE
