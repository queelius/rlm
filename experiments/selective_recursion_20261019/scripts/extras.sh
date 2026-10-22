#!/bin/bash
# Extra evaluations on GPU 1 inference: expert on the event-log dev split; SFT-002 on the event-log dev split.
cd /project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=../../.venv/bin/python
while ! grep -q FOLLOWUP_DONE /project/alex_phd/runs/selective-recursion-20261019/sweep-001-t0/followup.log 2>/dev/null; do sleep 60; done
$PY teach.py --out /project/alex_phd/runs/selective-recursion-20261019/dev-expert --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 8 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 5
while ! grep -q "loaded ctrl-sft2" /project/alex_phd/runs/selective-recursion-20261019/pipeline_gpu0.log 2>/dev/null; do sleep 60; done
$PY run_sweep.py --out /project/alex_phd/runs/selective-recursion-20261019/dev-sft2 --policies rlm --harness default --controller-model ctrl-sft2 --ns 8,32,64,128,256,512 --seeds 1000-1003 --split dev --concurrency 8 --temperature 0 2>&1 | grep -v -E "UserWarning|warn\(" | tail -n 40
echo EXTRAS_DONE
