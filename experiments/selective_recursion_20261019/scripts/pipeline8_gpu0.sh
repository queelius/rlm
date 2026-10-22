#!/bin/bash
# After rl-mixed-001 finishes: evaluate its final adapter on the held-out sets (GPU 1), and run the
# no-cost-term ablation (lambda = 0, otherwise identical) on GPU 0.
set -u
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
export CUDA_VISIBLE_DEVICES=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
S=/project/alex_phd/runs/selective-recursion-20261019
log() { echo "[$(date -u +%FT%TZ)] $*"; }
while ! grep -q "PIPELINE7_DONE" $S/pipeline7_gpu0.log; do sleep 60; done
FINAL=$(ls -d $S/rl-mixed-001/adapters/step-* | sort | tail -n 1)
log "rl-mixed-001 final adapter: $FINAL"
bash $S/eval_adapter.sh $FINAL ctrl-rl1 > $S/eval_rl1.log 2>&1 &
$TRAIN_PY rl_train.py --out $S/rl-mixed-nocost-001 --init-adapter $S/adapters/sft-005/final --task-source mixed --oolong-max-len 8192 --oolong-dev-per-len 4 --ns 8,32,64,128,256,512 --dev-per-cell 1 --harness default --steps 30 --tasks-per-step 16 --group 4 --lam 0.0 --temperature 1.0 --lr 2e-5 --eval-every 5 --rollout-concurrency 16 --deadline 1200 --exec-timeout 900 --max-total-tokens 600000 2>&1 | grep -v -E "UserWarning|warn\("
wait
log PIPELINE8_DONE
