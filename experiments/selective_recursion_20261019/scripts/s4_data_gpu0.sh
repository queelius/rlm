#!/bin/bash
# S4 (15:05 UTC): what does the 0.6B learn from the 4B's one-cell calls? (i) event-log-only traces (220),
# tested on OOLONG which it never saw; (ii) data efficiency: random mixed subsets of 55, 110 and 220 traces.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
M06=/project/alex_phd/research-cache/models/Qwen--Qwen3-0.6B
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
for v in eventlog-only n55 n110 n220; do
  ep=3; [ $v = n55 ] && ep=6; [ $v = n110 ] && ep=4
  CUDA_VISIBLE_DEVICES=$G0 SR_NO_THINK=1 SR_MODEL=$M06 $TRAIN_PY sft_train.py --data $S/sft-data/onecell-4b-lib-$v.jsonl --out $S/adapters/sft-06-$v --epochs $ep --lr 2e-4 --lora-r 32 --max-seq-len 8000 --batch-tokens 12000 2>&1 | grep -v -iE "warn|Loading weights" | tail -n 1
  [ -d $S/adapters/sft-06-$v/final ] && log "sft-06-$v trained" || log "sft-06-$v FAILED"
done
until grep -q "S2-SEED1-EVALS-DONE" $S/s2_confirm.log || [ "$(cat $S/s2-06one-s1-dev/rows.jsonl $S/s2-06one-s1-heldout/rows.jsonl $S/s2-06one-s1-oolong/rows.jsonl 2>/dev/null | wc -l)" -ge 288 ]; do sleep 60; done
pid=$(ps -eo pid,args | awk '/vllm serv[e] .*Qwen--Qwen3-0.6B / {print $1; exit}'); [ -n "$pid" ] && kill $pid; sleep 25
cd $S/serving && CUDA_VISIBLE_DEVICES=$G0 PATH=/project/alex_phd/envs/vllm/bin:$PATH nohup /project/alex_phd/envs/vllm/bin/vllm serve $M06 --served-model-name qwen3-0.6b-nt --host 127.0.0.1 --port 8004 --api-key local --max-model-len 16384 --gpu-memory-utilization 0.12 --max-num-seqs 32 --enable-prefix-caching --seed 0 --default-chat-template-kwargs '{"enable_thinking": false}' --enable-lora --max-lora-rank 64 --max-loras 4 --lora-modules ctrl-06-eventlog-only=$S/adapters/sft-06-eventlog-only/final ctrl-06-n55=$S/adapters/sft-06-n55/final ctrl-06-n110=$S/adapters/sft-06-n110/final ctrl-06-n220=$S/adapters/sft-06-n220/final > $S/serving/vllm_gpu0_qwen3-0.6b-nt.log 2>&1 &
cd $E; sleep 30
until curl -s -m 10 http://127.0.0.1:8004/v1/models -H "Authorization: Bearer local" | grep -q ctrl-06-n220; do sleep 15; done
log "0.6B served with the S4 adapters"
U=http://127.0.0.1:8004/v1
for v in eventlog-only n55 n110 n220; do
  t=s4-06-$v; for d in $t-dev $t-heldout $t-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
  $PY run_sweep.py --out $S/$t-dev --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-06-$v --controller-base-url $U --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 3 --temperature 0 > /dev/null 2>&1 &
  $PY run_sweep.py --out $S/$t-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model ctrl-06-$v --controller-base-url $U --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 3 --temperature 0 > /dev/null 2>&1 &
  $PY oolong_eval.py --out $S/$t-oolong --policies rlm --harness skills --controller-model ctrl-06-$v --controller-base-url $U --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 3 > /dev/null 2>&1 &
done
log "S4 evaluations launched"
