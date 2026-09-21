# Affordance-assisted ALFWorld screen

Exploratory frozen-model scaffolding, not novel hierarchy, RL, or learned depth.
CPU implementation only until the main agent accepts and launches a GPU run.

Accepted15:54UTC under `POST-RL-STOP-DECISION-001.json`: sealed source
`source-022-alfworld`, output `alfworld-screen-001`, supervisor17243. It follows
the fresh Hotpot comparison and separately labeled stopped-checkpoint21 readout.
The output PLAN is CPU-prepared; no ALF model outcomes existed at acceptance.
Eight focused fixtures passed, including a real isolated environment reset/step
and the base-only native-generation receipt path. Main remains sole GPU launcher.

Use the eight seen-development games in the hash-bound readiness manifest, seeds
2026092178 and 2026092179, and two policies: 32 planned episodes. The first game
and first seed run both policies before the rest. This smoke gate checks that
each arm returns a valid executed native action, not that either arm wins.

Both policies use the identical frozen base Qwen 4B, with no QA adapters. Flat
policy emits exactly `{"action": "native command"}`. The hierarchical arm first
emits exactly `{"goal": "short current goal"}`, then its worker emits the same
action object. A manager refresh occurs before actions 1, 5, 9, and so on. It
receives current public evidence but no previous private manager reasoning. A
worker receives the latest goal and the same public context as the flat policy.

Public context is the complete initial room/task observation, current feedback,
executed action/feedback history, and the current native admissible commands.
These lists provide affordance assistance, including information absent from
visible feedback; they are identical in both policies at an identical state.
No game path, scene metadata, native reward/won, expert plan, PDDL fact, or
trajectory annotation enters any policy prompt. Native won remains the host-side
success metric, distinct from done. Do not execute generated code.

Each episode has at most 50 executed environment actions, 2,048 generated tokens
across all policy calls, and 128 output tokens per request. Manager calls cost
the same budget as workers. Sampling temperature is 0.5, top-p 1, top-k 0.
Action-attempt seeds are shared across policies; manager seeds use a separate
stream. An invalid JSON object or inadmissible command consumes its actual
tokens/call but does not advance the environment. Three consecutive invalid
policy responses terminate the episode. Valid manager output is not an executed
action. There are no invisible retries, repairs, or fallback commands.

Use a deterministic oldest-first history suffix. The initial observation,
current state, and current commands are never removed. The retained suffix must
fit the largest flat/manager/worker prompt for that state, reserving space for
the bounded manager goal; record full and retained token counts and dropped
history count. Do not treat history truncation as new evidence access.

The ALFWorld engine runs in its existing isolated environment as a line-JSON
subprocess; the model runs in the existing training environment. A tiny bridge
projects only public fields while reporting native status separately. Every
model request/output and every native reset/action is immutable. Partial episode
state is checkpointed for inspection, but no implicit mid-episode reconstruction
or retry is supported. A failed owner requires a separately reviewed new run.

One-hour global cap, allocation end minus 600 seconds, native first-response
check within 90 seconds, authenticated owner, and the existing exclusive GPU
coordinator lock. An inference/bridge exception halts the owner and preserves
unobserved slots; it is not converted into an observed task failure. Report all
32 planned slots, missing outcomes, protocol stops, native success, action/model
costs, and paired game/seed changes. This tiny exposed panel cannot establish
benchmark superiority or a causal benefit of recursion. The package contrast
includes explicit short-term goals and extra managerial calls under an equal
total generated-token ceiling.

## CPU readiness and launch interface

Implemented in `alfworld_probe.py` and `alfworld_bridge.py`; only new files, no
shared collector or training-environment changes. Eight focused fixtures cover
strict parsing, host-field exclusion, a real isolated engine reset/step, manager
refresh and common action seeds, protocol stopping, total manager-token charging,
role-independent history truncation, and base-only native receipts using a tiny
CPU fake model. This is not a 4B model smoke result.

The manager goal is bounded to 240 characters. Common history selection always
reserves 512 goal tokens plus the full 128-token request allowance, even for the
last smaller-budget call, so the remaining budget does not change visibility.
Native requests retain actual prompt token IDs, output IDs, sampling parameters,
seed, timestamps, trimming counts, and explicit `adapter_enabled=false`.

Use the existing training interpreter. After sealing these files and dependencies,
prepare a fresh output on CPU, then let the main agent launch the same command
without `--prepare-only`:

```bash
TRAINPY <sealed>/alfworld_probe.py --output R/alfworld-screen-001 --hours 1 --prepare-only
```

`--validate-only` is an alias. The actual worktree preflight is
`R/alfworld-cpu-preflight-001/PLAN.json` (32 slots; no model loaded); do not reuse
that output after source paths change during sealing. The launch writes its own
OWNER/LOAD/STATUS/TERMINAL, per-call receipts, public/host-separated observations,
decisions, partial episode snapshots, immutable completed/failed episode records,
SMOKE result, and a planned-denominator SUMMARY. Inference or bridge failures
halt collection with preserved missing outcomes rather than task-failure zeros.
