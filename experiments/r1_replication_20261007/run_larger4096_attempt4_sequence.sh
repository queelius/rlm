#!/usr/bin/env bash
# One owner: fresh training, final MATH500 in both formats, then AMC/Minerva in both.
set -euo pipefail
W=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007
R=/project/alex_phd/runs/r1-zero-replication-20261007
S=/project/alex_phd/research-cache/repos/understand-r1-zero
P=/project/alex_phd/envs/r1-zero-dfca49d/bin/python
export CUDA_VISIBLE_DEVICES=0
bash "$W/experiments/r1_replication_20261007/run_larger4096_attempt4.sh" > "$R/larger4096-chat-drgrpo42-attempt4.log" 2>&1
MODEL=$("$P" - <<'PY'
from pathlib import Path
import json
import re
r = Path('/project/alex_phd/runs/r1-zero-replication-20261007')
h = Path('/home/atowell/research-runs/r1-zero-replication-20261007/larger4096-chat-drgrpo42-attempt4')
log = (r / 'larger4096-chat-drgrpo42-attempt4.log').read_text(errors='replace')
assert 'Traceback (most recent call last)' not in log, 'Training failure is not a completed endpoint'
assert log.count('finish learn()') == 255, 'Expected 255 completed collection-level learning rounds'
assert re.search(r"misc/policy_sgd_step['\"]?:\s*255(?:\.0)?\b", log), 'Expected 255 actual optimizer updates'
models = list(h.glob('debug_*/saved_models/step_00256/model.safetensors'))
assert len(models) == 1, 'Expected unique prescribed natural-end checkpoint'
run = models[0].parents[2]
assert len(json.loads((run / 'eval_results/256_math.json').read_text())) == 64
print(models[0].parent)
PY
)
bash "$W/experiments/r1_replication_20261007/evaluate.sh" "$MODEL" \
  "$R/eval500-larger4096-attempt4-final-chat" "$S/datasets/evaluation_suite" 500 qwen_math \
  > "$R/eval500-larger4096-attempt4-final-chat.log" 2>&1
bash "$W/experiments/r1_replication_20261007/evaluate.sh" "$MODEL" \
  "$R/eval500-larger4096-attempt4-final-raw" "$S/datasets/evaluation_suite" 500 no \
  > "$R/eval500-larger4096-attempt4-final-raw.log" 2>&1

# Broader checks are fixed before any new endpoint is inspected.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_USE_V1=0
for condition in chat raw; do
  case "$condition" in chat) TEMPLATE=qwen_math ;; raw) TEMPLATE=no ;; esac
  OUTPUT="$R/eval-broader-larger4096-attempt4-final-$condition"
  test ! -e "$OUTPUT"
  mkdir "$OUTPUT"
  cd "$OUTPUT"
  timeout --signal=TERM --kill-after=60s 30m "$P" "$S/evaluate_model.py" \
    --model_name "$MODEL" --tasks '["amc","minerva"]' --template "$TEMPLATE" \
    --dataset_name "$S/datasets/evaluation_suite" --temperature 0 --top_p 1 \
    --max_tokens 3000 --max_model_len 4096 --n_samples 1 --max_test 999999 --save True \
    > "$R/eval-broader-larger4096-attempt4-final-$condition.log" 2>&1
done
