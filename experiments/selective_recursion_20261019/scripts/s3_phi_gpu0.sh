#!/bin/bash
# S3 (15:10 UTC): is the recipe model-agnostic? Phi-4-mini-instruct (3.8B, another family) as caller:
# one-cell distillation adapter on the same 436 traces, then untrained and distilled Phi evaluated on
# event-log dev (library + note), the four grown forms and OOLONG-72; helpers stay the Qwen3-4B.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
MP=/project/alex_phd/research-cache/models/microsoft--Phi-4-mini-instruct--cfbefacb99257ffa30c83adab238a50856ac3083
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
pid=$(ps -eo pid,args | awk '/vllm serv[e] .*Qwen--Qwen3-8B-AWQ / {print $1; exit}'); [ -n "$pid" ] && kill $pid && log "8B server stopped"; sleep 20
CUDA_VISIBLE_DEVICES=$G0 SR_MODEL=$MP SR_LORA_TARGETS=qkv_proj,o_proj,gate_up_proj,down_proj $TRAIN_PY sft_train.py --data $S/sft-data/onecell-4b-lib.jsonl --out $S/adapters/sft-phi-onecell --epochs 3 --lr 2e-4 --lora-r 32 --max-seq-len 6000 --batch-tokens 12000 2>&1 | grep -v -iE "warn|Loading weights" | tail -n 2
[ -d $S/adapters/sft-phi-onecell/final ] && log "phi adapter trained" || { log "phi adapter FAILED"; }
cd $S/serving && CUDA_VISIBLE_DEVICES=$G0 PATH=/project/alex_phd/envs/vllm/bin:$PATH nohup /project/alex_phd/envs/vllm/bin/vllm serve $MP --served-model-name phi4-mini --host 127.0.0.1 --port 8005 --api-key local --max-model-len 16384 --gpu-memory-utilization 0.3 --max-num-seqs 16 --enable-prefix-caching --seed 0 --enable-lora --max-lora-rank 64 --lora-modules ctrl-phi-onecell=$S/adapters/sft-phi-onecell/final > $S/serving/vllm_gpu0_phi4-mini.log 2>&1 &
cd $E; sleep 30
until curl -s -m 10 http://127.0.0.1:8005/v1/models -H "Authorization: Bearer local" | grep -q ctrl-phi-onecell; do sleep 15; if grep -qE "Error|error:" $S/serving/vllm_gpu0_phi4-mini.log && ! pgrep -f "served-model-name phi4-mini" >/dev/null; then log "phi server FAILED"; exit 1; fi; done
log "phi served"
U=http://127.0.0.1:8005/v1
for t in s3-phi s3-phione; do for d in $t-dev $t-heldout $t-oolong; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done; done
ev() { local t=$1 m=$2
  $PY run_sweep.py --out $S/$t-dev --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model $m --controller-base-url $U --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
  $PY run_sweep.py --out $S/$t-heldout --policies rlm --harness skills --stats-file $S/stats-teach-v4.json --controller-model $m --controller-base-url $U --ns 8,32,64,128,256,512 --qts top_buyer,purchase_span,busiest_day,same_day_cross --seeds 1000-1003 --split dev --concurrency 4 --temperature 0 2>&1 | tail -n 1 &
  $PY oolong_eval.py --out $S/$t-oolong --policies rlm --harness skills --controller-model $m --controller-base-url $U --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 4 2>&1 | tail -n 1 &
}
ev s3-phi phi4-mini
ev s3-phione ctrl-phi-onecell
wait
log S3_DONE
