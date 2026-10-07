#!/usr/bin/env bash
# Prepared successor only: no waiting, automatic polling, or checkpoint selection.
# One isolated13h session; per-jobGNUtimeouts retain their process-group cleanup.
set -euo pipefail
if [[ "${1:-}" != --inside-owner-cap ]]; then
  exec /project/alex_phd/envs/r1-zero-dfca49d/bin/python \
    /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/r1_replication_20261007/scoped_session_owner.py \
    --seconds 46800 --grace 55 -- bash "$0" --inside-owner-cap
fi
W=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007
R=/project/alex_phd/runs/r1-zero-replication-20261007
RUN="$R/followups/fresh-lr5e6-fixed255"
S=/project/alex_phd/research-cache/repos/understand-r1-zero
P=/project/alex_phd/envs/r1-zero-dfca49d/bin/python

# Fail immediately until the entire current owner and its trainer have gone.
# Refuse even a recycledPID; resolve it manually rather than assuming release.
for prior_pid in 1861599 1861602; do
  if [[ -e "/proc/$prior_pid" ]]; then
    printf 'Refusing successor: prior owner/trainerPID%s still exists.\n' "$prior_pid" >&2
    exit 1
  fi
done
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$GPU_PROCESSES" ]]; then
  printf 'Refusing successor: GPU compute processes remain.\n' >&2
  exit 1
fi
# Authoritative allocation epoch, verified by root; calendar summaries were stale.
allocation_end=1791621574
launch_now=$(date -u +%s)
if (( allocation_end - launch_now < 13 * 3600 + 60 )); then
  printf 'Refusing successor: less than13h plus cleanup remains in the allocation.\n' >&2
  exit 1
fi
test ! -e "$RUN"
test ! -e /home/atowell/research-runs/r1-zero-replication-20261007/followups/fresh-lr5e6-fixed255
mkdir -p "$R/followups"
mkdir "$RUN"
export CUDA_VISIBLE_DEVICES=0
bash "$W/experiments/r1_replication_20261007/run_larger4096_lr5e6.sh" --inside-ready-owner \
  > "$RUN/training.log" 2>&1
MODEL=$("$P" - <<'PY'
from pathlib import Path
import json
import re
r = Path('/project/alex_phd/runs/r1-zero-replication-20261007/followups/fresh-lr5e6-fixed255')
h = Path('/home/atowell/research-runs/r1-zero-replication-20261007/followups/fresh-lr5e6-fixed255')
log = (r / 'training.log').read_text(errors='replace')
assert 'Traceback (most recent call last)' not in log, 'Training failure is not a completed endpoint'
assert log.count('finish learn()') == 255, 'Expected255 completed collection-level learning rounds'
assert re.search(r"misc/policy_sgd_step['\"]?:\s*255(?:\.0)?\b", log), 'Expected255 actual optimizer updates'
models = list(h.glob('debug_*/saved_models/step_00256/model.safetensors'))
assert len(models) == 1, 'Expected unique prescribed natural-end checkpoint'
run = models[0].parents[2]
assert len(json.loads((run / 'eval_results/256_math.json').read_text())) == 64
print(models[0].parent)
PY
)
bash "$W/experiments/r1_replication_20261007/evaluate.sh" "$MODEL" \
  "$RUN/eval500-final-chat" "$S/datasets/evaluation_suite" 500 qwen_math \
  > "$RUN/eval500-final-chat.log" 2>&1
bash "$W/experiments/r1_replication_20261007/evaluate.sh" "$MODEL" \
  "$RUN/eval500-final-raw" "$S/datasets/evaluation_suite" 500 no \
  > "$RUN/eval500-final-raw.log" 2>&1

# Same fixed broader conditions and decoding as attempt4.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_USE_V1=0
for condition in chat raw; do
  case "$condition" in chat) TEMPLATE=qwen_math ;; raw) TEMPLATE=no ;; esac
  OUTPUT="$RUN/eval-broader-final-$condition"
  test ! -e "$OUTPUT"
  mkdir "$OUTPUT"
  cd "$OUTPUT"
  timeout --signal=TERM --kill-after=60s 30m "$P" "$S/evaluate_model.py" \
    --model_name "$MODEL" --tasks '["amc","minerva"]' --template "$TEMPLATE" \
    --dataset_name "$S/datasets/evaluation_suite" --temperature 0 --top_p 1 \
    --max_tokens 3000 --max_model_len 4096 --n_samples 1 --max_test 999999 --save True \
    > "$RUN/eval-broader-final-$condition.log" 2>&1
done
