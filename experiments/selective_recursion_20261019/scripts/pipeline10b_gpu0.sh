#!/bin/bash
# After pipeline 9 and the route-diverse teaching: SFT-006 on all unified teaching (incl. taught direct
# route on short inputs) -> evals -> RL with lambda 0.1 and 0.3 from SFT-006 -> evals.
set -u
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
export CUDA_VISIBLE_DEVICES=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
RLM_PY=../../.venv/bin/python
S=/project/alex_phd/runs/selective-recursion-20261019
log() { echo "[$(date -u +%FT%TZ)] $*"; }
loadlora() { curl -sS -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d "{\"lora_name\": \"$1\", \"lora_path\": \"$2\"}"; echo; log "loaded $1"; }
COMMON="--task-source mixed --oolong-max-len 8192 --oolong-dev-per-len 4 --ns 8,32,64,128,256,512 --dev-per-cell 1 --harness default --steps 30 --tasks-per-step 16 --group 4 --temperature 1.0 --lr 2e-5 --eval-every 5 --rollout-concurrency 16 --deadline 1200 --exec-timeout 900 --max-total-tokens 600000"
# Resumed from the RL stage: SFT-006 evals (oolong-sft6, dev-sft6) continue as orphan processes on GPU 1.
$TRAIN_PY rl_train.py --out $S/rl-route-lam01 --init-adapter $S/adapters/sft-006/final --lam 0.1 --seed 0 $COMMON 2>&1 | grep -v -E "UserWarning|warn\("
F=$(ls -d $S/rl-route-lam01/adapters/step-* | sort | tail -n 1); log "route lam01 final: $F"
bash $S/eval_adapter.sh $F ctrl-rlroute01 > $S/eval_rlroute01.log 2>&1 &
$TRAIN_PY rl_train.py --out $S/rl-route-lam03 --init-adapter $S/adapters/sft-006/final --lam 0.3 --seed 0 $COMMON 2>&1 | grep -v -E "UserWarning|warn\("
wait
F=$(ls -d $S/rl-route-lam03/adapters/step-* | sort | tail -n 1); log "route lam03 final: $F"
bash $S/eval_adapter.sh $F ctrl-rlroute03 > $S/eval_rlroute03.log 2>&1
log PIPELINE10_DONE
