---
date: 2026-09-12
cutoff_utc: "2026-09-12T07:32:00Z"
question: rq:rl-effective-feedback
status: prospective_runtime_probe_complete
evidence_level: exploratory_one_prompt_one_device
---

# A serving option stabilized the probabilities in our small RL diagnostic

We found a practical explanation for part of our RL difficulty: the fast model
server could report substantially different probabilities at the same point in
the same answer, depending on repeated execution and concurrent requests.

In a new comparison, enabling vLLM's existing **batch-invariant mode** eliminated
the observed variation at that answer position. This is useful experimental
infrastructure evidence, not a new algorithm or proof that the old training data
is valid. vLLM documents this mode as a beta reproducibility feature with a possible
performance cost. [Official documentation](https://docs.vllm.ai/en/stable/features/batch_invariance/).

## What we compared

Both runs sent the same 32 requests with the same model weights, sampling seeds
and A100 device. Twenty-four were helper requests, covering serial execution,
four concurrent helper requests, and requests mixed with another model adapter.
Eight additional requests created the mixed-model workload.

Every helper answer reached the same inspected prefix. We examined two possible
next tokens: the selected `entity` token and the alternative `numeric` token.
These are **next-token probabilities**, not confidence that the whole answer is
correct.

| Probability at the inspected answer position | Default serving | Batch-invariant serving |
|---|---:|---:|
| Selected `entity` token | 72.89%–99.55% | 99.50% in all 24 calls |
| Alternative `numeric` token | 0.192%–26.81% | 0.247% in all 24 calls |

The alternative token's probability varied by a factor of about 140 in the
default run. In the batch-invariant run, repeated-seed and cross-scheduling
comparisons showed zero variation at the inspected position. The sampled helper
answers nevertheless matched between runs in all 24 cases. Matching answer text
alone would therefore have missed this problem.

One of the eight extra mixed-model answers did differ. We are not claiming that
every trajectory or every token is now reproducible.

## Why this matters for RL

Our paused fast-collection approach uses the recorded probability of a sampled
answer to account for differences between the serving model and the training
model. Unexplained probability variation makes that calculation difficult to
interpret. The new result motivates collecting a **fresh** batch under the
stabilized serving configuration and testing it against the original training
acceptance rules.

It does not justify repairing old probabilities after the fact, relaxing an
acceptance threshold, or assuming the serving and training implementations agree.
The [same-implementation helper RL experiment](2026-09-12-helper-rl-first-readout.md)
remains a separate approach that already completed one qualified update.

## Cost and limits

The stabilized run took 162 seconds versus 84 seconds for the default run,
including startup and cleanup. Token totals were identical. Summed request times
also increased, but those requests overlap, so their sum is not elapsed GPU time.
This is a small operational comparison, not a general performance benchmark.

The result concerns one helper prompt, one inspected answer position, one model
configuration and one A100. Current public issue reports describe remaining
variation under other workloads; we have not reproduced those reports locally.
That reinforces the need to verify the configuration actually used rather than
treat the flag as a universal guarantee.
[Example issue report](https://github.com/vllm-project/vllm/issues/51187).

## Evidence and the two earlier startup failures

The authoritative external report is
`/project/alex_phd/runs/rlm-research-r4/analyses/qs6-fixed-helper-batch-invariant-attempt003-2026-09-12/REPORT_V2.md`.
Its result SHA-256 is
`b6eb18c09b37a8fd8c3948b2d7dc82a58cca54c5422fda01cf079f3ac8d92435`.
MAIN reproduced the result exactly and separately checked all 48 numeric-token
probabilities against the two runs' recorded top-20 alternatives.

The audit authenticates the requests, model/device identities and service
processes. The actual owned inference engine logged execution of the
batch-invariant matrix-multiplication code. Both runs completed and released
their processes.

Two earlier attempts sent no scientific requests. The first lost the flag in a
child-process environment wrapper. The second activated the feature but failed
an unreliable check of the process environment after the process renamed itself.
The completed third attempt preserved the environment fix and replaced only
that faulty check. All failed attempts remain recorded; they are not model-quality
observations.

This report and its evidence pointers are preserved in Git. The external raw
run store is not backed up by a GitHub push.
