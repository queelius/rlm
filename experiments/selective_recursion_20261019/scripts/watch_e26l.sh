#!/bin/bash
# Watch: sampled-id RL progress (every ten steps), full LongBench and helper-data completions, idle alarms, exits.
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
declare -A seen
seen[s10]=1
for f in $H/*/EXIT; do seen[X$(basename $(dirname $f))]=1; done
seen[G]="$(grep -E 'smoke|FAILED|finished|DONE' $S/e26l_rl_sampled_ids.log 2>/dev/null | tail -n 1)"
block() {  # mean reward and solved count of steps $1-9 .. $1
  python3 - "$H/e26c-proposer-rl-ids/steps.jsonl" "$1" <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])][int(sys.argv[2]) - 10:int(sys.argv[2])]
print(f"sampled-id RL steps {int(sys.argv[2]) - 9}-{sys.argv[2]}: mean reward {sum(r['mean_reward'] for r in rs) / len(rs):.3f}, "
      f"solved {sum(r['solved'] for r in rs)} of 320")
PY
}
while true; do
  l=$(grep -E "FAILED|finished|DONE" $S/e26l_rl_sampled_ids.log 2>/dev/null | tail -n 1)
  if [ -n "$l" ] && [ "$l" != "${seen[G]}" ]; then seen[G]="$l"; echo "$(date -u +%H:%M) $l" | cut -c1-250; fi
  n=$(wc -l < $H/e26c-proposer-rl-ids/steps.jsonl 2>/dev/null || echo 0)
  for k in 20 30 40; do
    if [ "$n" -ge $k ] && [ -z "${seen[s$k]}" ]; then seen[s$k]=1; block $k; fi
  done
  for d in e32e-longbench-all-direct e32e-longbench-all-retrieve e32e-longbench-all-06-ret; do
    c=$(wc -l < $S/$d/rows.jsonl 2>/dev/null || echo 0)
    if [ "$c" -ge 503 ] && [ -z "${seen[$d]}" ]; then seen[$d]=1; echo "$(date -u +%H:%M) COMPLETE $d"; fi
  done
  for d in e26g-8b-exemplar-helper-train-b0 e26g-8b-exemplar-helper-train-b1 e26-disc-4brlids-heldout e26-disc-4brlids-heldout-r2; do
    if [ -f $S/$d/bench.jsonl ] && [ -z "${seen[$d]}" ]; then
      seen[$d]=1; echo "$(date -u +%H:%M) FINISHED $d: $(grep -A1 success $S/$d/launch.log | tr '\n' ' ' | cut -c1-200)"
    fi
  done
  if [ -f $S/queue/IDLE-ALARM ]; then
    k=IA$(date -u +%H)
    if [ -z "${seen[$k]}" ]; then seen[$k]=1; echo "$(date -u +%H:%M) $(cat $S/queue/IDLE-ALARM)"; fi
  fi
  for f in $H/*/EXIT; do
    d=$(basename $(dirname $f)); c=$(cat $f)
    if [ -z "${seen[X$d]}" ]; then seen[X$d]=1; if [ "$c" != "0" ]; then echo "$(date -u +%H:%M) NONZERO EXIT $d: $c"; fi; fi
  done
  sleep 120
done
