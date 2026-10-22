# Research operations tools

Small tools that keep the GPUs busy and make failures visible (built 22 October 2026 after the costliest
failures of the campaign turned out to be operational).

- `queue.py`: a job queue. Specs are JSON (`name`, `lane` = gpu0 or gpu1, `priority`, `cmd`, optional `env` and
  `after`) in `RUNS/queue/pending`; `queue.py run` (a detached dispatcher) starts the highest-priority job whenever
  its lane has a free slot (capacities in `RUNS/queue/slots.json`, re-read every minute), tracks the job's own PID,
  and writes `RUNS/queue/IDLE-ALARM` when a lane has run nothing for 10 minutes. `queue.py add SPEC`, `queue.py ls`.
- `launch.py`: checked launch. Dry-runs the command's argument parsing, writes `MANIFEST.json` (command, commit,
  dirty flag, lanes), starts the job detached with its exit code recorded in `EXIT`, and marks the run FAILED if
  it dies or every early row is an error.
- `status.py`: one table of every recent run (rows, score per arm, errors, age, nonzero exits).

Check `IDLE-ALARM` and `queue.py ls` first when resuming a session.
