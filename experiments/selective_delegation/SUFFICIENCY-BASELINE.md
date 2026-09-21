# Full-context evidence-sufficiency baseline

This is an existing-benchmark baseline, not a new decomposition method. The question is whether
the frozen base 4B model can distinguish sufficient from insufficient supplied evidence before
we introduce constrained retrieval or additional training. GPU execution remains conditional on
main-agent review; the preparation below made no model calls.

## Frozen panel

The official MuSiQue full-development archive and metric implementation come from repository
commit `922ac98f19a201998dbdae6d7f2887a5258dbdeb` (CC BY 4.0 data). The exact archive, member,
metric, exclusion-input, and preparer hashes are in
`sufficiency-inputs-001/MANIFEST.json` under the active September 21 research store.

All previously selected original384/fresh003/breadth MuSiQue parent IDs, normalized questions,
and atomic component IDs are excluded. Component matching uses the union of both official
variants. There are438 eligible parents, unchanged by checking260 prior Hotpot rows representing
228 distinct normalized questions (prepared explorer32, canonical128, and original explorer100).
The first32 parents ordered by SHA256 of `2026092180:` plus original ID are retained, with
both official variants. No hop or outcome filter was applied. All32 happen to be two-hop;
this is not a balanced-hop or whole-development estimate. The remaining eligible pool is
already predominantly two-hop, so longer-chain sufficiency is not tested here.

Each prompt contains only the question and documents with contiguous new document IDs, title,
and text. Documents are ordered by a label-blind hash of seed, question, title, and text.
Original source IDs, original paragraph indices, support flags, answers, aliases, decomposition,
hop counts, and variant labels remain host-only. This removes index/order leakage, not the
natural title, content, distractor, and length differences between official paired variants.
The comparison is not a pure support-deletion intervention. Known-component disjointness does
not establish semantic or pretraining independence.

## Frozen inference and accounting

There are128 planned native calls:32 parents × two variants × seeds2026092181/2026092182.
The frozen base model has no adapters. Every call has temperature0.5, top-p1, top-k0, and a
128-token generation cap. The exact instruction is in `sufficiency_probe.INSTRUCTION` and PLAN.
It asks for an answer only if the supplied documents support the entire question, without
answering from memory. The response must be exactly
`{"answerable": boolean, "answer": string}`. Duplicate or extra fields, non-boolean labels,
fences, or malformed JSON are protocol-invalid; there is no repair or answer fallback.
The instruction requests an empty string when abstaining, but scoring does not add a
non-official empty-string validity condition.

CPU tokenization of every exact chat-template prompt passed: maximum4,215 input tokens;
adding128 generated tokens stays below8,192. There is no truncation or replacement selection.
`sufficiency-cpu-preflight-001/PLAN.json` records the per-variant token counts and128 jobs.
The collector has an exclusive coordinator lock, authenticated owner,20-minute cumulative cap,
allocation-end minus600-second guard, native starts/returns, immutable episode checkpoints,
and terminal accounting. First scientific response is bounded by90seconds. An inference error
halts the run without retry. An existing owner is refused rather than silently resumed.

The official `GroupAnswerSufficiencyMetric` is called directly for each parent/repeat. Both
availability predictions must be correct; only then is the supported variant's answer EM/F1
credited. Its own positive aliases are used, never substituted by the insufficient sibling's
potentially different aliases. Primary denominator is64 paired attempts. Separate supported
answer EM/F1, false abstentions, insufficient-context overanswers, returned protocol failures,
missing/unavailable attempts, actual calls/tokens, and native latency sums are reported.
Missing attempts are explicitly unobserved; planned-denominator metrics are lower bounds
when anything is unavailable, not claims that unobserved model behavior failed scientifically.

## Reproduction

Accepted by main at16:26UTC, with `source-024-sufficiency`, the sealed-source
`sufficiency-001/PLAN.json`, and `SUFFICIENCY-DECISION-001.json`. Supervisor6348
waits for authenticated execution-credit release. No model outcomes were
available at acceptance. The earlier CPU-preflight output remains separate.

From a separately sealed copy of the experiment sources:

```bash
TRAINPY sufficiency_probe.py --cases R/sufficiency-inputs-001/cases.jsonl \
  --output R/sufficiency-001 --hours 0.3333333333333333 --prepare-only
# Main may later remove --prepare-only after acceptance and assign the GPU.
```

`TRAINPY` denotes the existing frozen training-environment Python; `R` denotes the active
September21 sidecar directory. This baseline changes neither accepted ALFWorld nor QA jobs.
An informative outcome could motivate a missing-evidence interface experiment, but neither
high accuracy nor a failure alone establishes an advantage for adaptive planning.
