#!/bin/bash
# E10 (11:20 UTC): online routine selection (L3c). After the growth chains register the learned routines,
# run four 240-task streams from a fresh seed namespace with the untrained controller: Thompson sampling,
# greedy and uniform-random from an empty table, and greedy warm-started from the teaching statistics (v2).
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
until grep -q E08B_DONE $S/e08b_growth_gpu1.log; do sleep 60; done
log "starting bandit streams; learned skills: $(ls skills/learned/ | tr '\n' ' ')"
for arm in thompson greedy random; do
  mkdir -p $H/e10-bandit-$arm && ln -sfn $H/e10-bandit-$arm $S/e10-bandit-$arm
  $PY online_bandit.py --out $S/e10-bandit-$arm --policy $arm --steps 240 --batch 4 > $H/e10-bandit-$arm/run.log 2>&1 &
done
mkdir -p $H/e10-bandit-greedy-warm && ln -sfn $H/e10-bandit-greedy-warm $S/e10-bandit-greedy-warm
$PY online_bandit.py --out $S/e10-bandit-greedy-warm --policy greedy --warm $S/stats-teach-v2.json --steps 240 --batch 4 > $H/e10-bandit-greedy-warm/run.log 2>&1 &
wait
log E10_DONE
