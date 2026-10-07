#!/usr/bin/env bash
# Configuration-only launcher: all scientific logic is the authors' code.
set -euo pipefail
R=/project/alex_phd/runs/r1-zero-replication-20261007
P=/project/alex_phd/envs/r1-zero-dfca49d/bin/python
S=/project/alex_phd/research-cache/repos/understand-r1-zero
M=/project/alex_phd/research-cache/models/Qwen--Qwen2.5-Math-1.5B--4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
export CUDA_HOME=/usr/local/cuda-12.5
export PATH="/project/alex_phd/envs/r1-zero-dfca49d/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/home/atowell/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export TORCH_EXTENSIONS_DIR="$R/torch-extensions"
cd "$R"
exec timeout --signal=TERM --kill-after=60s 90m "$P" "$S/train_zero_math.py" \
  --critic_type drgrpo --gpus 1 --enable_prefix_caching --collocate --vllm_sleep \
  --vllm_gpu_ratio 0.35 --max_model_len 4096 --gradient-checkpointing --flash-attn \
  --bf16 --no-rnd-seed --seed 42 --learning_rate 0.000001 --lr_scheduler constant \
  --num_ppo_epochs 1 --beta 0 --oracle_type reward --oracle math --pretrain "$M" \
  --prompt_template qwen_math --zero-stage 2 --ref_offload \
  --prompt_data "$R/data-oat/train64" --train_split train --input_key problem --output_key answer \
  --max-train 9999999 --max-queries 512 --num_prompt_epoch 1 --prompt_max_length 1024 \
  --num_samples 8 --temperature 1 --top_p 1 --generate_max_length 3000 \
  --save_steps 4 --save-ckpt --max_save_num 2 --train_batch_size 128 \
  --train_batch_size_per_device 1 --rollout_batch_size 16 --rollout_batch_size_per_device 16 \
  --pi_buffer_maxlen_per_device 128 --eval_batch_size 64 --eval_steps 4 \
  --eval_temperature 0 --eval_generate_max_length 3000 --eval_data "$R/data-oat/monitor64" \
  --eval_input_key input --save_path "$R/pilot-attempt2" --dump_replay_every 1
