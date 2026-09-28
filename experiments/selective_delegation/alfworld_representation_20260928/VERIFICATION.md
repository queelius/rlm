# CPU preparation verification — September 28, 2026

Ready for parent review/dispatch. No GPU launch, environment mutation, accepted-source edit,
Git commit or push. This is a runnable exploratory comparison, not a scientific result.

External study:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/alfworld-representation-20260928-001`.

- `PREPARED-JOBS.json` SHA256:
  `56497c642c69fc3a2e4cb822e358ddf77a8ce04bb9cffb74e9d462f2dee42b71`.
  Six descriptors share 48 verified immutable pins. Command checkpoint identity is deliberately
  null until a complete, authenticated step 33 exists; the original index endpoint is bound now.
- Eight focused tests passed using the accepted GPU interpreter with `CUDA_VISIBLE_DEVICES=`.
  Direct `/project/alex_phd/envs/rlm/bin/ruff check` and `ruff format --check` passed for all 12
  Python files. No `uv run`, dependency install or environment synchronization was invoked.
- All 12 prospectively selected unseen games reset successfully, without outcome-based replacement.
  Maximum initial prompt plus 128-token response is 1,127 index / 1,133 command, below 8,192.
  This qualifies initial context only: later full-history overflow remains an unknown outcome.
- The final `runtime-fixture-002/VERIFICATION.json` SHA256 is
  `cd7360a3b4c3fad106ce47d9c2a187bc5bb1d66233a0f7fd6db6c853d66e6316`.
  Both saved-response/token paths execute the same 35 native TRAIN actions and win; command
  additionally rejects one uppercase command without advancing the native state. Each saved
  trajectory independently replays in a fresh native process. All 46 fixture source-pin checks
  pass. Responses are scripted, not a model competence test; zero scientific model calls/GPU use.
- Original sealed source 041b's seven files and source 042b's 14 files still match their manifests.
  Original TRAIN data and checkpoint hashes were rechecked by the matched-endpoint admission.
  Failed/stale fixture 001 remains preserved; fixture 002 binds the final production source set.

Exact verification entry points (run from the active worktree):

```bash
CUDA_VISIBLE_DEVICES= /project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python -m pytest -q experiments/selective_delegation/alfworld_representation_20260928/test_alf_rep.py
/project/alex_phd/envs/rlm/bin/ruff check experiments/selective_delegation/alfworld_representation_20260928
/project/alex_phd/envs/rlm/bin/ruff format --check experiments/selective_delegation/alfworld_representation_20260928
CUDA_VISIBLE_DEVICES= /project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python experiments/selective_delegation/alfworld_representation_20260928/alf_rep_jobs.py
```

GPU scientific caps total 116 minutes: 20 minutes for new command SFT, then four independent
24-minute readouts. Supervisor padding and a final 20-minute CPU audit/comparison cap total
146 minutes. Expected useful GPU time is approximately 1–2 hours, not a promised completion time.
Missing training must not suppress index/base controls, and incomplete readouts remain unknown.

Inference is limited to this representation/training package: identical 524 TRAIN demonstrations
and 33 optimizer updates do not match exposure. Command targets contain 5,362 supervised tokens,
versus 4,060 index tokens (+32.1%), and token-mean loss changes per-example weighting. The new
games span only four scene IDs. Prior symbol-binding work already establishes the broad mechanism;
the primary decision is the paired difference in own-interface learning gain.
