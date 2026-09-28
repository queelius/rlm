# RL collection signal census

CPU-only, read-only analysis of the September 28 raw/binder collection. It imports
the existing trainer's `load_batch` and `batch_credits`; it changes neither reward,
credit, sampled targets, nor any live run. The native audit must be complete and
all 32 episodes must have known rewards. Partial collections are rejected, not
turned into failures.

[Raw result](RAW-RESULT.md) and [proposed gradient diagnostic](MASK-DIAGNOSTIC.md).

```bash
PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
E=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/rl_signal_20260928
R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921

# Raw final report already exists; do not overwrite it.
CUDA_VISIBLE_DEVICES= "$PY" "$E/signal_analysis.py" \
  --collection "$R/textcraft-rl-assist-20260928-001/binder/collect-0001" \
  --output "$R/analysis-rl-signal-20260928-001/binder.json"
CUDA_VISIBLE_DEVICES= "$PY" "$E/signal_analysis.py" \
  --raw-report "$R/analysis-rl-signal-20260928-001/raw-final.json" \
  --binder-report "$R/analysis-rl-signal-20260928-001/binder.json" \
  --output "$R/analysis-rl-signal-20260928-001/COMPARISON.json"
```

The analyzer records absolute advantage-weighted token mass as well as signed
mass and raw counts. Longer trajectories legitimately contain more terms in the
full score-function objective; these totals do not measure gradient norms or
demonstrate an objective defect. Positive trajectory credit to a failed action is
not causal action-level credit.

Diversity at task start requires identical saved input-token prefixes. Later
matching prefixes are reported separately; differing history, remaining budget,
or inventory is never merged. Semantic action keys discard formatting and notes,
and ignore finish wording, but preserve query order and all execution arguments.
Projected craft keys additionally omit ingredients; they are not equivalent raw
actions and are not automatically identical future policy states.

Craft-token attribution requires native re-encoding to match every non-special
emitted ID exactly. Whole-member tokens, cross-boundary tokens, JSON punctuation,
and actual special tokens are counted separately. No token-fraction claim comes
from character counts or newly normalized JSON. Both schema-valid crafts and
native-successful crafts are reported.

Verification: five focused tests pass; `ruff check` and `ruff format --check` pass
for this folder. The actual raw integration verifies 1,644 native receipts and
790 generation-probability sidecars. No full-project suite or GPU run was used.
`raw.json` is the initial identical-science report; its pre-format source is
preserved externally as `signal_analysis-initial.py`. Prefer `raw-final.json`,
whose source hash matches the finalized analyzer.
