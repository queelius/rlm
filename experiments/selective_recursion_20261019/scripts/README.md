# Pipeline scripts (snapshot of the run store)

Copies of the shell pipelines that launched servers, training and evaluation runs for this study, taken
from the run store `/project/alex_phd/runs/selective-recursion-20261019/` on 2026-10-22 (the copies there are the ones that ran). They hard-code this
cluster's paths (run store, model cache, environments, MIG device ids) and the local vLLM servers' placeholder
key `local`; read them as a record of what was run, not as portable entry points.

Naming: `eNN_*` belongs to the experiment of that number in `docs/RESULTS.md`; `pipelineNN_gpu0.sh` are the
earlier chained GPU0 pipelines; `watch_*.sh` are monitoring loops. Later runs were launched through the job
queue (`experiments/ops/queue.py`); their exact commands are in each run directory's `MANIFEST.json`.
