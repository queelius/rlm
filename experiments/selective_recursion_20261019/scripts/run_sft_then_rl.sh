#!/bin/bash
# GPU 0 pipeline: SFT warm start from expert trajectories, load adapter into vLLM, dev-evaluate, then RL.
set -u
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
export CUDA_VISIBLE_DEVICES=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
TRAIN_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
RLM_PY=../../.venv/bin/python
S=/project/alex_phd/runs/selective-recursion-20261019
# wait until GPU 0 has no compute process (Track A queue finished)
while nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | grep -q .; do sleep 60; done
while pgrep -f run_independent_queue.py >/dev/null; do sleep 60; done
echo "GPU0 free at $(date -u +%FT%TZ)"
$TRAIN_PY sft_train.py --data $S/teach-001/sft.jsonl --out $S/adapters/sft-001 --epochs 2 --lr 1e-4 --lora-r 32 || exit 1
curl -sS -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H 'Authorization: Bearer local' -H 'Content-Type: application/json' -d "{\"lora_name\": \"ctrl-sft1\", \"lora_path\": \"$S/adapters/sft-001/final\"}" && echo " loaded ctrl-sft1"
$RLM_PY run_sweep.py --out $S/dev-sft1 --policies rlm --harness default --controller-model ctrl-sft1 --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 12 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\("
$RLM_PY run_sweep.py --out $S/dev-base --policies direct,rlm,fixed --harness default --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 12 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\("
$TRAIN_PY rl_train.py --out $S/rl-001 --init-adapter $S/adapters/sft-001/final --harness default --steps 40 --tasks-per-step 16 --group 4 --lam 0.1 --temperature 1.0 --lr 2e-5 --eval-every 5 --dev-seed 1000 --dev-per-cell 2 --rollout-concurrency 16 2>&1 | grep -v -E "UserWarning|warn\("
echo PIPELINE_DONE
