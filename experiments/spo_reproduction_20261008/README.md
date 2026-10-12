# SPO experiment snapshots

These are the small owner, monitor, diagnostic and runtime adapters used for
the single-A100 SPO reproduction. The current run is documented in
[the research report](../../docs/spo-reproduction-2026-10-08/diagnosis-and-restart-20261012.md).
They are research snapshots, not a portable installer. Absolute paths identify
the external pinned SPO source, environment, public model/data cache and native
run artifacts on our cluster. Model weights are not in this repository.

## October 12 fresh run

The published copies of `bos_only_adapter.py`, `bf16_zero_norm.py`,
`fixed_endpoint.py`, `launch_bos_only.py` and `fresh_bos_only.jsonnet` match the
sealed live-source hashes in the [preparation manifest](../../docs/spo-reproduction-2026-10-08/evidence/bos-preparation-20261012.json).
The manifest's unlaunched admission label describes preparation time;
the [restart receipt](../../docs/spo-reproduction-2026-10-08/evidence/restart-20261012.json)
records the later actual launch. Do not edit the external sealed copy while its
owner is live, or launch this entry point beside the owner.

The BOS adapter changes query tokenization before native likelihoods and MC
preparation, not just the final PPO input. It leaves EOS behavior unchanged.
The BF16 adapter changes only a strictly-positive gradient-norm assertion to
allow finite zero, retaining native optimizer updates and nonfinite rejection.
The endpoint hook validates full checkpoint 690 / 11,040 updates, then invokes
native evaluation without another optimizer update. These are explicit changes
to the public implementation, not proof of the unavailable historical source.

The focused tests run on CPUs with the existing external SPO environment:

```bash
cd experiments/spo_reproduction_20261008
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  /project/alex_phd/envs/spo-6c5f947/bin/python -m unittest -v \
  test_bos_alignment test_endpoint test_bf16_zero_norm
```

The fixtures use the real pinned tokenizer/native preparation paths and small
synthetic probabilities or optimizer tensors. They do not measure model quality.
The saved-model FusedAdam restoration probe is a separate GPU diagnostic; it
must be admitted through the scoped owner, not run beside active training.

The fresh owner uses direct single-rank Python distributed settings to preserve
the A100 MIG UUID. Its exact environment, 68-hour cap, fixed endpoint, checkpoint
policy and artifact destination live in
`/project/alex_phd/runs/rl-paper-reproduction-20261012/training-queue-bos-only-001.json`.
The 1,000-iteration learning-rate horizon is not shortened to the 690 endpoint.

Future resumes require the latest complete **model and optimizer** state, not
HF-only weights. An interrupted run remains incomplete even if its earlier
score was favorable. The observer requests human-level analysis from this
research session; it does not interpret results or authorize new GPU work.
