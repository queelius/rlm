#!/bin/bash
# After the no-cost ablation: evaluate its final adapter; then a second seed of the lambda=0.1 run,
# evaluate; then a stronger cost term (lambda=0.3), evaluate.
set -u
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
export CUDA_VISIBLE_DEVICES=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
S=/project/alex_phd/runs/selective-recursion-20261019
log() { echo "[$(date -u +%FT%TZ)] $*"; }
COMMON="--init-adapter $S/adapters/sft-005/final --task-source mixed --oolong-max-len 8192 --oolong-dev-per-len 4 --ns 8,32,64,128,256,512 --dev-per-cell 1 --harness default --steps 30 --tasks-per-step 16 --group 4 --temperature 1.0 --lr 2e-5 --eval-every 5 --rollout-concurrency 16 --deadline 1200 --exec-timeout 900 --max-total-tokens 600000"
while ! grep -q "PIPELINE8_DONE" $S/pipeline8_gpu0.log; do sleep 60; done
F=$(ls -d $S/rl-mixed-nocost-001/adapters/step-* | sort | tail -n 1); log "nocost final: $F"
bash $S/eval_adapter.sh $F ctrl-rlnocost > $S/eval_rlnocost.log 2>&1 &
$TRAIN_PY rl_train.py --out $S/rl-mixed-002 --lam 0.1 --seed 1 $COMMON 2>&1 | grep -v -E "UserWarning|warn\("
wait
F=$(ls -d $S/rl-mixed-002/adapters/step-* | sort | tail -n 1); log "seed-2 final: $F"
bash $S/eval_adapter.sh $F ctrl-rl2 > $S/eval_rl2.log 2>&1 &
$TRAIN_PY rl_train.py --out $S/rl-mixed-lam03-001 --lam 0.3 --seed 0 $COMMON 2>&1 | grep -v -E "UserWarning|warn\("
wait
F=$(ls -d $S/rl-mixed-lam03-001/adapters/step-* | sort | tail -n 1); log "lam03 final: $F"
bash $S/eval_adapter.sh $F ctrl-rllam03 > $S/eval_rllam03.log 2>&1
log PIPELINE9_DONE
