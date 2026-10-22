#!/bin/bash
# GPU 0 pipeline, part 7 (07:40 UTC): SFT-003 ablation (diversified, family-specific first cell) while
# OOLONG re-teaching with composed aggregate cells runs; then SFT-005 on unified teaching with composed
# aggregates -> load -> evals -> mixed-task RL from SFT-005.
set -u
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
export CUDA_VISIBLE_DEVICES=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
RLM_PY=../../.venv/bin/python
S=/project/alex_phd/runs/selective-recursion-20261019
log() { echo "[$(date -u +%FT%TZ)] $*"; }
loadlora() { curl -sS -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d "{\"lora_name\": \"$1\", \"lora_path\": \"$2\"}"; echo; log "loaded $1"; }
unloadlora() { curl -sS -X POST http://127.0.0.1:8001/v1/unload_lora_adapter -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d "{\"lora_name\": \"$1\"}"; echo; }
$TRAIN_PY sft_train.py --data $S/teach-001/sft.jsonl $S/teach-002/sft.jsonl $S/teach-oolong-001/sft.jsonl $S/teach-oolong-002/sft.jsonl --out $S/adapters/sft-003 --epochs 1 --lr 1e-4 --lora-r 32 2>&1 | grep -v -iE "warn|Loading weights" || log "SFT-003 ablation failed (continuing)"
while [ ! -f $S/teach-oolong-004/DONE ]; do sleep 60; done
$TRAIN_PY sft_train.py --data $S/teach-003/sft.jsonl $S/teach-oolong-004/sft.jsonl --out $S/adapters/sft-005 --epochs 2 --lr 1e-4 --lora-r 32 2>&1 | grep -v -iE "warn|Loading weights" || { log "SFT-005 failed"; exit 1; }
[ -d $S/adapters/sft-005/final ] || { log "SFT-005 missing"; exit 1; }
unloadlora ctrl-sft4
loadlora ctrl-sft5 $S/adapters/sft-005/final
$RLM_PY oolong_eval.py --out $S/oolong-sft5 --policies rlm --controller-model ctrl-sft5 --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 10 &
$RLM_PY run_sweep.py --out $S/dev-sft5 --policies rlm --harness default --controller-model ctrl-sft5 --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 8 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 3 &
wait
log "SFT-005 evals done"
$TRAIN_PY rl_train.py --out $S/rl-mixed-001 --init-adapter $S/adapters/sft-005/final --task-source mixed --oolong-max-len 8192 --oolong-dev-per-len 4 --ns 8,32,64,128,256,512 --dev-per-cell 1 --harness default --steps 30 --tasks-per-step 16 --group 4 --lam 0.1 --temperature 1.0 --lr 2e-5 --eval-every 5 --rollout-concurrency 16 --deadline 1200 --exec-timeout 900 --max-total-tokens 600000 2>&1 | grep -v -E "UserWarning|warn\("
log PIPELINE7_DONE
