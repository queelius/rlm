#!/bin/bash
# GPU 0 pipeline (revised 03:35 UTC): wait for Track A, then
#  1. SFT-001 on event-log expert trajectories; load; dev-eval (event-log) and OOLONG zero-shot eval
#  2. wait for OOLONG teaching data; SFT-002 on event-log + OOLONG; load; OOLONG test eval
#  3. RL on OOLONG train split from SFT-002 (reward: correct - 0.1*tokens/10k), dev on OOLONG test split
set -u
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
export CUDA_VISIBLE_DEVICES=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
RLM_PY=../../.venv/bin/python
S=/project/alex_phd/runs/selective-recursion-20261019
log() { echo "[$(date -u +%FT%TZ)] $*"; }
while pgrep -f "run_independent_queu[e].py" >/dev/null; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | grep -q .; do sleep 30; done
log "GPU0 free"
$TRAIN_PY sft_train.py --data $S/teach-001/sft.jsonl --out $S/adapters/sft-001 --epochs 2 --lr 1e-4 --lora-r 32 2>&1 | grep -v -iE "warn|Loading weights" || { log "SFT-001 failed"; exit 1; }
curl -sS -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d "{\"lora_name\": \"ctrl-sft1\", \"lora_path\": \"$S/adapters/sft-001/final\"}"; echo; log "loaded ctrl-sft1"
$RLM_PY run_sweep.py --out $S/dev-sft1 --policies rlm --harness default --controller-model ctrl-sft1 --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 12 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 40
$RLM_PY oolong_eval.py --out $S/oolong-sft1 --policies rlm --controller-model ctrl-sft1 --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 16
while [ ! -f $S/teach-oolong-001/DONE ]; do sleep 60; done
$TRAIN_PY sft_train.py --data $S/teach-001/sft.jsonl $S/teach-oolong-001/sft.jsonl --out $S/adapters/sft-002 --epochs 2 --lr 1e-4 --lora-r 32 2>&1 | grep -v -iE "warn|Loading weights" || { log "SFT-002 failed"; exit 1; }
curl -sS -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d "{\"lora_name\": \"ctrl-sft2\", \"lora_path\": \"$S/adapters/sft-002/final\"}"; echo; log "loaded ctrl-sft2"
$RLM_PY oolong_eval.py --out $S/oolong-sft2 --policies rlm --controller-model ctrl-sft2 --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 16
curl -sS -X POST http://127.0.0.1:8001/v1/unload_lora_adapter -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d '{"lora_name": "ctrl-sft1"}'; echo
$TRAIN_PY rl_train.py --out $S/rl-oolong-001 --init-adapter $S/adapters/sft-002/final --task-source oolong --oolong-max-len 8192 --oolong-dev-per-len 6 --harness default --steps 30 --tasks-per-step 16 --group 4 --lam 0.1 --temperature 1.0 --lr 2e-5 --eval-every 5 --rollout-concurrency 16 --deadline 1200 --exec-timeout 900 --max-total-tokens 600000 2>&1 | grep -v -E "UserWarning|warn\("
log PIPELINE_DONE
