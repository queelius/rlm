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
# Pipeline 15: after pipeline 14, a 1-step smoke test of the KL option, then relative-0.5 + KL 0.1 (uniform
# lengths) and relative-0.5 + KL 0.1 with short lengths over-weighted, each evaluated and route-sampled.
t1() { $RLM_PY run_sweep.py --out $S/dev-$1-t1 --policies rlm --harness default --controller-model $1 --ns 8,32 --seeds 1000-1007 --split dev --concurrency 8 --temperature 1.0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2; }
evalall() { bash $S/eval_adapter.sh $1 $2 > $S/eval_$2.log 2>&1; t1 $2 > $S/t1_$2.log 2>&1; }
SHORT="--train-ns 8,8,8,32,32,64,128,256,512 --oolong-train-lens 1024,1024,1024,2048,2048,4096,8192"
# (2026-10-20 04:55) started concurrently with pipeline 14 to overlap rollouts and updates across the two GPUs
rm -rf $S/rl-kl-smoke; $TRAIN_PY rl_train.py --out $S/rl-kl-smoke --init-adapter $S/adapters/sft-007/final --cost-mode relative --lam 0.5 --kl-coef 0.1 --seed 7 --task-source mixed --oolong-max-len 8192 --oolong-dev-per-len 1 --ns 8,32 --dev-per-cell 1 --harness default --steps 1 --tasks-per-step 4 --group 2 --temperature 1.0 --lr 2e-5 --eval-every 100 --rollout-concurrency 8 --deadline 1200 --exec-timeout 900 --max-total-tokens 600000 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 5
if [ -f $S/rl-kl-smoke/adapters/step-0001/adapter_config.json ] && ! grep -q "Traceback" $S/pipeline15_gpu0.log; then log "KL smoke ok"; else log "KL smoke failed"; exit 1; fi
$TRAIN_PY rl_train.py --out $S/rl-route7-rel05-kl --init-adapter $S/adapters/sft-007/final --cost-mode relative --lam 0.5 --kl-coef 0.1 --seed 0 $COMMON 2>&1 | grep -v -E "UserWarning|warn\("
F=$(ls -d $S/rl-route7-rel05-kl/adapters/step-* | sort | tail -n 1); log "route7 rel05-kl final: $F"
evalall $F ctrl-rlroute7-rel05-kl &
$TRAIN_PY rl_train.py --out $S/rl-route7-rel05-short-kl --init-adapter $S/adapters/sft-007/final --cost-mode relative --lam 0.5 --kl-coef 0.1 --seed 0 $SHORT $COMMON 2>&1 | grep -v -E "UserWarning|warn\("
wait
F=$(ls -d $S/rl-route7-rel05-short-kl/adapters/step-* | sort | tail -n 1); log "route7 rel05-short-kl final: $F"
evalall $F ctrl-rlroute7-rel05-short-kl
log PIPELINE15_DONE
