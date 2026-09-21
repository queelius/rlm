# Frozen-trace aggregation diagnostic

## Question and prior art

Does allowing the final answerer to read every original document weaken the relationship
between an executed plan and its reward, or does that access mainly rescue erroneous helper
answers? Higher reward variance by itself is not better credit assignment.

PyRAG already uses intermediate variable binding, no-document final synthesis, frozen-role
reinforcement learning, and answerer-first training. It also reports correct intermediate
answers followed by wrong aggregation. This is a local interface diagnostic, not a claim that
any of those ideas or failure modes are new.

Primary references: <https://arxiv.org/html/2605.12975v1> and
<https://github.com/GasolSun36/PyRAG/tree/5d8ab2ea10b9bf3da1ab2581c8ad5aa93d5df263>.
The external checkout is `/project/alex_phd/research-cache/repos/PyRAG-inspect-20260921`,
retrieved 2026-09-21. No project-level license declaration was found. No third-party code
is executed, installed, or copied by this experiment.

## Frozen source and eligibility

Use the complete `rl-planner-001/batch-0001/BATCH.json`, the source run's immutable PLAN,
all 64 planned episode records, and every referenced native call receipt. Verify source
request digests, receipt timestamps, frozen helper flags, source dependency hashes, and
the cases checksum. Reconstruct each eligible plan from the actual root response,
bind each dependency to its actual preceding helper prediction, and verify the saved
helper requests, trace, original final prompt, and score. Gold and aliases are host-side
scoring inputs only. No annotated decomposition or initial checkpoint enters new prompts.

The source policy may continue later batches; this diagnostic needs only completed batch one.
It does not require the entire RL owner to terminate for CPU preparation. Main is the sole GPU
launcher and must wait for the exclusive GPU owner to release its lock.

All 16 parents and four candidates per parent remain in the denominator. Invalid plans,
dependencies, and helper responses become explicit zero rows without further model calls.
An invalid original final with a complete valid helper trace is eligible: the original final
is not the treatment baseline. Missing generations or an incomplete source batch are errors,
not zero-reward examples. Source batch one currently contains 58 scored traces and six invalid
helper paths, giving 232 new calls and 24 explicit source-zero rows.

## Matched intervention

For every eligible candidate, sample two fresh final repeats under both conditions:

- `full_source`: original question, generated plan, actual helper trace, and original documents.
- `trace_only`: exactly the same serialized fields and instruction, except documents are omitted.

Both conditions use one common neutral evidence instruction and the shared short-answer JSON
contract. There is no forced copying of the last helper answer and no gold-dependent repair.
Both finals are newly sampled; the source RL final is provenance, never the new full-source
baseline. Its instruction differs from this experiment's common instruction.

Temperature is 0.5 and the final token cap is 128. Seeds are
`2026092108 + 50000 + parent_index * 100 + repeat`, shared across candidates and conditions
within each parent/repeat. Condition order alternates by parent and repeat. Maximum planned
work is 256 final calls, four collector threads, one A100, and one hour per bounded invocation,
also capped at allocation end minus 600 seconds. The existing native service/client and
exclusive coordinator lock are reused. The first scientific response must return within
90 seconds. There are no helper regenerations, retries, answer fallbacks, or optimizer steps.

## Outputs and interpretation

The immutable PLAN binds source paths/hashes, dependency hashes, cases, model manifest,
all source request digests, new prompt digests, seeds, caps, and candidate eligibility.
Per-call receipts preserve native inputs, outputs, usage and timing. Per-candidate rows
separate new physical cost from hypothetical deployment cost: root plus helpers plus one
new final. The source original final is excluded from hypothetical deployment cost.

Report EM and F1 over 128 planned candidate-repeat outcomes per condition, explicit source
zeros, new protocol failures, transport failures, and pending outcomes separately. Until all
rows finish, all-planned rates are only lower bounds. Also report paired wins/losses among
returned eligible final pairs. Parent identity, not the repeated final call, is the statistical
cluster for any later uncertainty analysis.

For eligible candidates with both repeats returned, report the variance of candidate mean
rewards within each parent and the average variance across a candidate's two final repeats.
These are descriptive, not unbiased estimates of latent plan quality. Candidate variation
also reflects frozen helper realizations and can include duplicate plans. Two repeats cannot
fully separate final sampling noise from plan-content effects. Six invalid-helper paths remain
in headline reward denominators but are excluded from this conditional variation description.

Promote the interface hypothesis only if trace-only access improves useful differentiation
without a material average-EM loss. More variation caused solely by propagating helper mistakes
is a negative result. Full-source access can legitimately rescue bad plans. The diagnostic
does not prove planner faithfulness, establish causal mediation, or demonstrate an RL gain.

## Launch (main only)

Run from the sealed experiment source directory using the existing native-inference environment:

```sh
/project/alex_phd/envs/prime-rl-5990b1b/bin/python aggregation_probe.py \
  --source-output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/rl-planner-001 \
  --cases /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/inputs-001/cases.jsonl \
  --output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/aggregation-probe-001 \
  --hours 1
```

CPU verification uses only the focused aggregation tests and saved batch reconstruction.
The GPU-critical bounded research scope takes precedence over a broad test-suite run.
