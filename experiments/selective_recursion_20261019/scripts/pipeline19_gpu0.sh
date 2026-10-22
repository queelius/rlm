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
# Pipeline 19 (third process on GPU 0, short): SFT-008 on the consistent-teacher set (direct only at 8
# records, decomposition elsewhere), then temperature-0 evals and temperature-1 route sampling.
t1() { $RLM_PY run_sweep.py --out $S/dev-$1-t1 --policies rlm --harness default --controller-model $1 --ns 8,32 --seeds 1000-1007 --split dev --concurrency 8 --temperature 1.0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2; }
$TRAIN_PY sft_train.py --data $S/teach-consistent-008/sft.jsonl --out $S/adapters/sft-008 --epochs 2 --lr 1e-4 --lora-r 32 2>&1 | grep -v -iE "warn|Loading weights" || { log "SFT-008 failed"; exit 1; }
[ -d $S/adapters/sft-008/final ] || { log "SFT-008 missing"; exit 1; }
loadlora ctrl-sft8 $S/adapters/sft-008/final
$RLM_PY oolong_eval.py --out $S/oolong-sft8 --policies rlm --controller-model ctrl-sft8 --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 5 &
$RLM_PY run_sweep.py --out $S/dev-sft8 --policies rlm --harness default --controller-model ctrl-sft8 --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 8 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 3 &
t1 ctrl-sft8 > $S/t1_ctrl-sft8.log 2>&1 &
wait
log PIPELINE19_DONE
