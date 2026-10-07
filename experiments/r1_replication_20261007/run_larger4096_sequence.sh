#!/usr/bin/env bash
# One GPU owner: training, then both prescribed-final tests; never best-checkpoint selection.
set -euo pipefail
W=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007
R=/project/alex_phd/runs/r1-zero-replication-20261007
S=/project/alex_phd/research-cache/repos/understand-r1-zero
P=/project/alex_phd/envs/r1-zero-dfca49d/bin/python
export CUDA_VISIBLE_DEVICES=0
bash "$W/experiments/r1_replication_20261007/run_larger4096.sh" > "$R/larger4096-chat-drgrpo42.log" 2>&1
MODEL=$("$P" - <<'PY'
from pathlib import Path
import json
import re
r = Path('/project/alex_phd/runs/r1-zero-replication-20261007')
h = Path('/home/atowell/research-runs/r1-zero-replication-20261007/larger4096-chat-drgrpo42')
log = (r / 'larger4096-chat-drgrpo42.log').read_text(errors='replace')
assert log.count('finish learn()') == 32, 'Expected 32 completed collection-level learning rounds'
assert re.search(r"misc/policy_sgd_step['\"]?:\s*256(?:\.0)?\b", log), 'Expected 256 actual optimizer updates'
models = list(h.glob('debug_*/saved_models/step_00033/model.safetensors'))
assert len(models) == 1, 'Expected unique prescribed final checkpoint'
run = models[0].parents[2]
assert len(json.loads((run / 'eval_results/33_math.json').read_text())) == 64
print(models[0].parent)
PY
)
bash "$W/experiments/r1_replication_20261007/evaluate.sh" "$MODEL" \
  "$R/eval500-larger4096-final-chat" "$S/datasets/evaluation_suite" 500 qwen_math \
  > "$R/eval500-larger4096-final-chat.log" 2>&1
bash "$W/experiments/r1_replication_20261007/evaluate.sh" "$MODEL" \
  "$R/eval500-larger4096-final-raw" "$S/datasets/evaluation_suite" 500 no \
  > "$R/eval500-larger4096-final-raw.log" 2>&1
