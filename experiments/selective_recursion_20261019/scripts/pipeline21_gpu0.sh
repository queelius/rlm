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
# Pipeline 21b: after the stratified plain run, its evals, then the balance and strong-KL arms from SFT-007.
t1() { $RLM_PY run_sweep.py --out $S/dev-$1-t1 --policies rlm --harness default --controller-model $1 --ns 8,32 --seeds 1000-1007 --split dev --concurrency 8 --temperature 1.0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2; }
evalall() { bash $S/eval_adapter.sh $1 $2 > $S/eval_$2.log 2>&1; t1 $2 > $S/t1_$2.log 2>&1; }
COMMONS="--cost-mode relative --lam 0.5 --seq-norm --sample-mode stratified"
BAL="--balance-target 0.3 --balance-coef 1.0 --balance-anneal-from 20"
SHORT="--train-ns 8,8,8,32,32,64,128,256,512 --oolong-train-lens 1024,1024,1024,2048,2048,4096,8192"
run() { name=$1; init=$2; shift 2; $TRAIN_PY rl_train.py --out $S/$name --init-adapter $init $COMMONS "$@" $COMMON 2>&1 | grep -v -E "UserWarning|warn\("; F=$(ls -d $S/$name/adapters/step-* | sort | tail -n 1); log "$name final: $F"; evalall $F ctrl-$name & }
while [ ! -d $S/rl-s7-strat/adapters/step-0030 ]; do sleep 60; done
while pgrep -f "rl_train.py --out $S/rl-s7-strat " > /dev/null; do sleep 30; done
F=$(ls -d $S/rl-s7-strat/adapters/step-* | sort | tail -n 1); log "rl-s7-strat final: $F"; evalall $F ctrl-rl-s7-strat &
run rl-s7-strat-bal $S/adapters/sft-007/final --seed 0 $BAL
run rl-s7-strat-kl5 $S/adapters/sft-007/final --seed 0 --kl-coef 5
wait
log PIPELINE21_DONE
