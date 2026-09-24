# Allocation-tail handoff — September 24, 01:22 UTC

The GPU is still running the RL-trained model's final memory condition. Queue020
waits for that scientific owner to finish and release the GPU; it does not interrupt it.

## Accepted next work

1. Audit the last memory condition and the complete memory comparison.
2. Evaluate exactly four previously unavailable positive-only RL cases. Do not retry
   the 28 completed outcomes. Charge the interrupted prefix and all extra compute.
3. Compare quantity-corrected original teaching examples with public-discovery
   teaching examples in changed world47, using 16 episodes per model.
4. Audit and compare the saved native outcomes.

The RL supplement has a 30-minute collection cap. Each world47 arm has a
30-minute cap. The runner refuses jobs whose cap cannot fit before the allocation
cutoff; accepted does not mean completed. Missing outcomes remain unknown.

## Scientific rationale

The recovered RL result is 15 successes among 28 observed cases; four cases are
unknown. Among known paired cases, positive-only credit wins three and loses
three against signed credit. There is no established RL improvement. The
supplement resolves coverage, not matched one-hour efficiency.

The public-discovery teacher still wins after correcting original demonstrations'
quantity handling in the original world. It also wins in changed worlds against
the *uncorrected* original teacher. World47 crosses those two controls, which is
more informative than another repetition of the uncorrected comparison.

The completed memory comparisons disagree across trained models. Adding a
structured memory is not yet a generally beneficial harness change. See
[the latest report](RESEARCH-UPDATE-20260924.md) for counts and limitations.

## Resume pointers

External root: `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- Accepted receipt: `INDEPENDENT-TRAINING-QUEUE-020.json`, SHA256
  `239cf0d1deb441d3cfacd84c47c6d73c5df705bbea91862bcdaf25905618b065`.
- Successor supervisor: PID646625; invocation in `independent-training-queue-020/`.
- Preserved scientific owner: PID638942, create time1790211111.36;
  output `textcraft-memory-rl_cp2-ledger_recent4-001`.
- Old supervisor019 was stopped **alone**; its scientific child remains running.
  The old current-job wrapper receipt may therefore be absent. Native scientific
  terminal receipts remain authoritative. Do not restart019.
- Cancellation record: `REPRIORITIZE-019-TO-020.json`. Old world47–49 tail is deferred.
- Frozen source: `source-positive-credit-supplement-002` and
  `source-textcraft-quantity-world47-003`; prior proposals were not launched.
- Allocation5879 ends around03:59 UTC; the runner stops accepting work ten minutes
  earlier. No promise that every queued comparison fits.

Four focused CPU tests passed before queue acceptance. All131 unique queue pins
were checked. Working-copy line wrapping may differ from frozen execution sources;
no live or accepted source was edited. Check native results before making claims.
Shared Codex quota was6% at01:21; no reset was observed. Jobs run without Codex tokens.

Git backs up source and this handoff, not external model weights or raw run artifacts.
