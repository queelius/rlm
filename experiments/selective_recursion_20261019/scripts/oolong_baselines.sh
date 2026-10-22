#!/bin/bash
# After OOLONG teaching (train split) and the sweep follow-up finish: expert on the test split, then
# direct and base-controller baselines on the same test rows (GPU 1 inference only).
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=../../.venv/bin/python
while [ ! -f /project/alex_phd/runs/selective-recursion-20261019/teach-oolong-001/DONE ]; do sleep 60; done
while ! grep -q FOLLOWUP_DONE /project/alex_phd/runs/selective-recursion-20261019/sweep-001-t0/followup.log 2>/dev/null; do sleep 60; done
$PY teach_oolong.py --out /project/alex_phd/runs/selective-recursion-20261019/oolong-expert-test --split test --max-len 32768 --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 12
$PY oolong_eval.py --out /project/alex_phd/runs/selective-recursion-20261019/oolong-base --policies direct,rlm --controller-model qwen3-4b --lens 1024,2048,4096,8192,16384,32768 --per-cell 6 --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 16
echo OOLONG_BASELINES_DONE
