#!/usr/bin/env bash
# Finite exploratory screen: baseline128chat/raw, then three fresh-base arms.
# Never launch until the current whole255-update owner and its GPU children release.
set -euo pipefail
W=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007
X="$W/experiments/r1_replication_20261007"
P=/project/alex_phd/envs/r1-zero-dfca49d/bin/python
if [[ "${1:-}" != --inside-screen-owner ]]; then
  exec "$P" "$X/scoped_session_owner.py" --seconds 36000 --grace 55 \
    -- bash "$0" --inside-screen-owner
fi
R=/project/alex_phd/runs/r1-zero-replication-20261007
RUN="$R/followups/quickscreen512-v1"
NATIVE=/home/atowell/research-runs/r1-zero-replication-20261007/followups/quickscreen512-v1
S=/project/alex_phd/research-cache/repos/understand-r1-zero
BASE=/project/alex_phd/research-cache/models/Qwen--Qwen2.5-Math-1.5B--4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
DATA="$R/data-oat/quick512-lvl3to5-v1"

gpu_released() {
  local active_pids
  active_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [[ -z "$active_pids" ]]
}
for prior_pid in 1861599 1861602; do
  if [[ -e "/proc/$prior_pid" ]]; then
    printf 'Refusing screen: prior whole owner/trainerPID%s still exists.\n' "$prior_pid" >&2
    exit 1
  fi
done
gpu_released
allocation_end=1791621574
launch_now=$(date -u +%s)
(( allocation_end - launch_now >= 10 * 3600 + 60 ))
test ! -e "$RUN"
test ! -e "$NATIVE"
test -f "$DATA/train/data-00000-of-00001.arrow"
test -f "$R/quick512-data-manifest.json"
test -f "$R/data-oat/heldout128/math/data-00000-of-00001.arrow"
test -f "$R/runtime-memory-v1/manifest.json"
# Verify this small new input once per owner, before any model/GPU job.
"$P" - "$R/quick512-data-manifest.json" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
p = Path(sys.argv[1])
assert hashlib.sha256(p.read_bytes()).hexdigest() == '7ae7327441dc09ad557a203dac9de2e8116992c31dbe073212b6f1926102a490'
m = json.loads(p.read_text())
assert m['n'] == m['checks']['native_PromptDataset_used'] == 512
assert m['checks']['drop_last_batches'] == 32 and m['checks']['drop_last_discarded'] == 0
for name, digest in m['files'].items():
    assert hashlib.sha256((Path(m['path']) / name).read_bytes()).hexdigest() == digest
PY
mkdir -p "$R/followups"
mkdir "$RUN"
mkdir -p "$NATIVE"
export PYTHONPATH="$R/runtime-memory-v1${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-12.5
unset PYTORCH_CUDA_ALLOC_CONF
export PATH="/project/alex_phd/envs/r1-zero-dfca49d/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/home/atowell/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1 VLLM_USE_V1=0
export TORCH_EXTENSIONS_DIR="$R/torch-extensions"
ARM=baseline ROOT="$RUN/baseline" MODEL="$BASE" UPDATES=0 PHASE=baseline RECEIPT_WRITTEN=0

record_failure() {
  local code=$?
  if [[ "$RECEIPT_WRITTEN" == 0 && -d "$ROOT" ]]; then
    local status=failed
    if [[ "$code" == 124 || "$code" == 137 || "$code" == 143 ]]; then status=capped; fi
    "$P" "$X/quickscreen_receipt.py" record --root "$ROOT" --native "$NATIVE/$ARM" \
      --updates "$UPDATES" --arm "$ARM" --status "$status" --phase "$PHASE" \
      --exit-code "$code" --model "$MODEL" || true
  fi
}
trap record_failure EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

evaluate_pair() {
  local condition template
  for condition in chat raw; do
    case "$condition" in chat) template=qwen_math ;; raw) template=no ;; esac
    PHASE="eval-$condition"
    mkdir "$ROOT/eval-$condition"
    cd "$ROOT/eval-$condition"
    timeout --signal=TERM --kill-after=60s 20m "$P" "$S/evaluate_model.py" \
      --model_name "$MODEL" --tasks '["math"]' --template "$template" \
      --dataset_name "$R/data-oat/heldout128" --temperature 0 --top_p 1 \
      --max_tokens 3000 --max_model_len 4096 --n_samples 1 --max_test 128 --save True \
      > "$ROOT/eval-$condition.log" 2>&1
    gpu_released
  done
  PHASE=receipt-check
  "$P" "$X/quickscreen_receipt.py" record --root "$ROOT" --native "$NATIVE/$ARM" \
    --updates "$UPDATES" --arm "$ARM" --status completed --phase completed \
    --exit-code 0 --model "$MODEL"
  RECEIPT_WRITTEN=1
}

mkdir "$ROOT"
evaluate_pair
for ARM in control lr5e6 reuse; do
  gpu_released
  ROOT="$RUN/$ARM" MODEL="$BASE" RECEIPT_WRITTEN=0 UPDATES=32
  LR=0.000001 PPO_EPOCHS=1
  case "$ARM" in lr5e6) LR=0.000005 ;; reuse) PPO_EPOCHS=2 UPDATES=64 ;; esac
  mkdir "$ROOT"
  cd "$ROOT"
  PHASE=training
  # After collection32 the response querycounter4096 exceeds4095; steps is already33.
  # The same forced natural-end33 labels32 updates in PPO1 and64 updates in PPO2.
  timeout --signal=TERM --kill-after=60s 2h "$P" "$S/train_zero_math.py" \
    --critic_type drgrpo --gpus 1 --enable_prefix_caching --collocate --vllm_sleep \
    --vllm_gpu_ratio 0.35 --max_model_len 4096 --gradient-checkpointing --flash-attn \
    --bf16 --no-rnd-seed --seed 42 --learning_rate "$LR" --lr_scheduler constant \
    --num_ppo_epochs "$PPO_EPOCHS" --beta 0 --oracle_type reward --oracle math --pretrain "$BASE" \
    --prompt_template qwen_math --verifier_version math_verify --zero-stage 2 --ref_offload \
    --prompt_data "$DATA" --train_split train --input_key problem --output_key answer \
    --max-train 512 --max-queries 4095 --num_prompt_epoch 1 --prompt_max_length 1024 \
    --num_samples 8 --temperature 1 --top_p 1 --generate_max_length 3000 \
    --save_steps 8 --save-ckpt --max_save_num 2 --train_batch_size 128 \
    --train_batch_size_per_device 1 --rollout_batch_size 16 --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 128 --eval_batch_size 64 --eval_steps 16 \
    --eval_temperature 0 --eval_generate_max_length 3000 --eval_data "$R/data-oat/monitor64" \
    --eval_input_key input --save_path "$NATIVE/$ARM" --dump_replay_every 1 \
    > "$ROOT/training.log" 2>&1
  gpu_released
  PHASE=endpoint-check
  MODEL=$("$P" "$X/quickscreen_receipt.py" endpoint --root "$ROOT" \
    --native "$NATIVE/$ARM" --updates "$UPDATES")
  evaluate_pair
done
trap - EXIT TERM INT
