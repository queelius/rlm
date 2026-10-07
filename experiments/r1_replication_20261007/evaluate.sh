#!/usr/bin/env bash
# Run separately for base and the prespecified final checkpoint, not the best.
set -euo pipefail
MODEL=${1:?Supply an immutable model or final checkpoint directory}
OUTPUT=${2:?Supply a new output directory}
R=/project/alex_phd/runs/r1-zero-replication-20261007
EVAL_DATA=${3:-$R/data-oat/heldout128}
EVAL_LIMIT=${4:-128}
P=/project/alex_phd/envs/r1-zero-dfca49d/bin/python
S=/project/alex_phd/research-cache/repos/understand-r1-zero
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_USE_V1=0
cd "$OUTPUT"
exec timeout --signal=TERM --kill-after=60s 30m "$P" "$S/evaluate_model.py" \
  --model_name "$MODEL" --tasks '["math"]' --template qwen_math \
  --dataset_name "$EVAL_DATA" --temperature 0 --top_p 1 \
  --max_tokens 3000 --max_model_len 4096 --n_samples 1 --max_test "$EVAL_LIMIT" --save True
