#!/bin/bash
# E10 restart 2 (13:35 UTC): applicability-restricted candidates and a hierarchical prior across lengths.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
log "learned skills: $(ls skills/learned/ | tr '\n' ' ')"
for arm in thompson greedy random; do
  mkdir -p $H/e10-bandit-$arm && ln -sfn $H/e10-bandit-$arm $S/e10-bandit-$arm
  $PY online_bandit.py --out $S/e10-bandit-$arm --policy $arm --steps 240 --batch 4 > $H/e10-bandit-$arm/run.log 2>&1 &
done
mkdir -p $H/e10-bandit-greedy-warm && ln -sfn $H/e10-bandit-greedy-warm $S/e10-bandit-greedy-warm
$PY online_bandit.py --out $S/e10-bandit-greedy-warm --policy greedy --warm $S/stats-teach-v3.json --steps 240 --batch 4 > $H/e10-bandit-greedy-warm/run.log 2>&1 &
wait
log E10_DONE
