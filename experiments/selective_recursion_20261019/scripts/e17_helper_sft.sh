#!/bin/bash
# E17 (18:25 UTC): train the helper, not the controller. LoRA on the 4B helper with SMS-spam and TREC items
# that appear in no OOLONG window (320 classify-format chunks); then per-row accuracy on held-out windows and
# the 293-row held-out OOLONG run with the tuned helper (controller: untrained 4B with the library).
S=/project/alex_phd/runs/selective-recursion-20261019
H=/home/atowell/research-runs/selective-recursion-20261019
E=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/experiments/selective_recursion_20261019
TE=/project/alex_phd/envs/selective-recursion-train-5b1e2b4/bin/python
PY=/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007/.venv/bin/python
G0=MIG-cfcfe369-9742-5104-84d9-7e0bfc536bc5
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cd $E
CUDA_VISIBLE_DEVICES=$G0 $TE sft_train.py --data $S/sft-data/helper-disjoint-labels.jsonl --out $S/adapters/helper-ft --epochs 2 --lr 1e-4 --lora-r 16 --max-seq-len 4000 --batch-tokens 12000 2>&1 | grep -v -iE "warn|Loading weights" | tail -n 2
[ -d $S/adapters/helper-ft/final ] && log "helper adapter trained" || { log "helper adapter FAILED"; exit 1; }
curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" -d "{\"lora_name\": \"helper-ft\", \"lora_path\": \"$S/adapters/helper-ft/final\"}"; echo
until curl -s -m 10 http://127.0.0.1:8001/v1/models -H "Authorization: Bearer local" | grep -q '"helper-ft"'; do sleep 10; done
log "helper-ft loaded on GPU1"
for d in e17-rowacc-base e17-rowacc-ft e17-oolong-ft; do mkdir -p $H/$d; ln -sfn $H/$d $S/$d; done
PYTHONPATH=$E:$E/../../src $TE row_accuracy.py --out $S/e17-rowacc-base --split test --per-len 4 --base-url http://127.0.0.1:8001/v1 --model qwen3-4b > $H/e17-rowacc-base/run.log 2>&1 &
PYTHONPATH=$E:$E/../../src $TE row_accuracy.py --out $S/e17-rowacc-ft --split test --per-len 4 --base-url http://127.0.0.1:8001/v1 --model helper-ft > $H/e17-rowacc-ft/run.log 2>&1 &
$PY oolong_eval.py --out $S/e17-oolong-ft --policies rlm --harness skills --stats-file $S/stats-teach-v5.json --controller-model qwen3-4b --public-model helper-ft --lens 1024,2048,4096,8192,16384,32768 --per-cell 50 --concurrency 6 > $H/e17-oolong-ft/run.log 2>&1 &
wait
log E17_DONE
