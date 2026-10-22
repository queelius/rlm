#!/bin/bash
# E26d: test-time RL for library discovery on MRCR. RL on MRCR's own teaching rows (held-out rows untouched), from the
# first RL-trained proposer. Any sample that solves its 12 unseen teaching tasks is kept; the first kept solution is then
# checked on all 30 teaching rows and on the held-out 2- and 4-needle sets (scripted, no model in the loop).
set -u
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
X=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
E=/project/alex_phd/envs/vllm
G1=MIG-43a3375a-1942-592d-accf-d0884164dcb8
OUT=$H/e26d-mrcr-ttrl
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for job in e26-disc-4bh-heldout e26-disc-4bh-train; do
  if [ -f $H/$job/PID ]; then pid=$(cat $H/$job/PID); kill -TERM -- -$pid 2>/dev/null
    echo "[$(date -u +%FT%TZ)] paused (process group $pid stopped) for the E26d MRCR test-time RL; resumes when requeued" >> $H/$job/NOTE.txt; fi
done
sleep 5
srv=$(ps -eo pid,args | awk 'index($0,"vllm serve") && index($0,"--port 8007") && !index($0,"awk") {print $1}')
[ -n "$srv" ] && { kill $srv; while kill -0 $srv 2>/dev/null; do sleep 2; done; sleep 5; }
log "hybrid-4B runs paused, 8007 stopped"
cd $X; mkdir -p $OUT
CUDA_VISIBLE_DEVICES=$G1 $TE proposer_rl.py --out $OUT --init-adapter $H/e26c-proposer-rl/step-40 --start-name proposer-rl-0-40 \
  --families mrcr --steps 30 --families-per-step 8 --group 4 --seed 7 --keep-solutions > $OUT/run.log 2>&1 || log "TTRL FAILED: $(tail -n 3 $OUT/run.log)"
log "TTRL finished: $(tail -n 1 $OUT/run.log); solutions: $(ls $OUT/solutions 2>/dev/null | wc -l)"
first=$(ls $OUT/solutions/*.py 2>/dev/null | sort -V | head -n 1)
if [ -n "$first" ]; then
  { printf '"""MRCR library found by test-time RL (E26d).\n\nanswer_mrcr(request) -> str    # final answer text\nTypical use: FINAL_TEXT(answer_mrcr(request)).\n"""\n'; sed 's/answer_mrcr_[A-Za-z0-9_]*/answer_mrcr/g' $first; } > $OUT/first_solution.py
  for n in 2 4; do $PY run_routine.py --family mrcr --needles $n --module $OUT/first_solution.py --entry answer_mrcr --out $OUT/heldout-mrcr$n \
      --split test --concurrency 8 > $OUT/heldout-mrcr$n.log 2>&1; log "held-out $n needles: $(tail -n 1 $OUT/heldout-mrcr$n.log)"; done
  $PY run_routine.py --family mrcr --needles 2 --module $OUT/first_solution.py --entry answer_mrcr --out $OUT/teach-all --split teach \
      --concurrency 8 > $OUT/teach-all.log 2>&1; log "all 30 teaching rows: $(tail -n 1 $OUT/teach-all.log)"
fi
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=$G1
nohup $E/bin/vllm serve /project/alex_phd/research-cache/models/Qwen--Qwen3-4B --served-model-name qwen3-4b-hybrid --host 127.0.0.1 \
  --port 8007 --api-key local --dtype bfloat16 --max-model-len 24576 --gpu-memory-utilization 0.38 --max-num-seqs 8 \
  --enable-prefix-caching --seed 0 > $S/serve-8007-hybrid4b-restart4.log 2>&1 &
sp=$!
until curl -s -H "Authorization: Bearer local" http://127.0.0.1:8007/v1/models | grep -q qwen3-4b-hybrid; do
  kill -0 $sp 2>/dev/null || { log "8007 restart FAILED"; exit 1; }; sleep 5; done
log "8007 restarted; E26I DONE"
