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
# Pipeline 23 (08:25 UTC): the library-trained controller. (1) native replay of the library-calling expert on the
# event-log teaching split with the prelude harness; (2) SFT-009 on those pairs (third process on GPU 0);
# (3) temperature-0 dev evaluation and temperature-1 route sampling with the prelude harness; also the untrained
# base with the prelude, for the no-training datapoint.
$RLM_PY run_sweep.py --out $S/dev-lib-base --policies rlm --harness skills --controller-model qwen3-4b --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 6 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 3 &
$RLM_PY teach.py --out $S/teach-lib-009 --harness skills --cells-dir expert_cells_library --ns 8,32,64,128,256,512 --seeds 16-23 --split teach --diversify --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 3
wait
$TRAIN_PY sft_train.py --data $S/teach-lib-009/sft.jsonl --out $S/adapters/sft-009 --epochs 2 --lr 1e-4 --lora-r 32 2>&1 | grep -v -iE "warn|Loading weights" || { log "SFT-009 failed"; exit 1; }
[ -d $S/adapters/sft-009/final ] || { log "SFT-009 missing"; exit 1; }
loadlora ctrl-sft9 $S/adapters/sft-009/final
$RLM_PY run_sweep.py --out $S/dev-sft9 --policies rlm --harness skills --controller-model ctrl-sft9 --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 8 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 3 &
$RLM_PY run_sweep.py --out $S/dev-ctrl-sft9-t1 --policies rlm --harness skills --controller-model ctrl-sft9 --ns 8,32 --seeds 1000-1007 --split dev --concurrency 8 --temperature 1.0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 2 &
wait
log PIPELINE23_DONE
