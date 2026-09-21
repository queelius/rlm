# QAMPARI sampling qualification: no baseline rescue

The model-recommended sampling package did not materially repair the direct reader.
Across all 16 frozen DEV questions × 2 repeats, valid JSON fell from 22/32 to 21/32;
official F1 moved from .1863 to .1916, an uncertain +.0054. Native generation time rose
by 19.8%. Retire the current QAMPARI fan-out branch rather than expand to a matched
sampling map run. This is a decision about this model, prompt, evidence pool, and
budget—not a claim that distributed multi-answer QA or decomposition cannot work.

## Matched comparison

Both runs used the same base 4B model without adapters, exact question and ranked 200
retrieved chunks, chat prompt, two seeds, 1,024-token output cap, and 40,960 runtime
context bound without truncation. Only the sampling package changed: temperature/top-p/
top-k .5/1/0 → .7/.8/20. This is not an isolated temperature effect. These are the
already-exposed 16 DEV parents, not fresh confirmation or live global retrieval.

| Measure | Original direct | Recommended sampling |
|---|---:|---:|
| Valid / planned outputs | 22 / 32 | 21 / 32 |
| Official precision | .2882 | .2952 |
| Official recall | .2115 | .2170 |
| Official F1 | .1863 | .1916 |
| Output-cap stops / EOS | 10 / 22 | 11 / 21 |
| Prompt tokens | 1,022,968 | 1,022,968 |
| Generated tokens | 12,962 | 15,646 |
| Total tokens | 1,035,930 | 1,038,614 |
| Native generation seconds | 624.0 | 747.8 |

All 32 new calls and all 32 reused baseline calls are available; no missing outcomes,
transport errors, unlinked calls, or unresolved starts. New physical work is 32 calls,
not 64. The audit independently reconstructed native prompts/token IDs, seeds and
sampling, regraded strict JSON with the official alias scorer, and checked receipts
against the prior immutable report. Empty answer lists score zero; malformed output
is not repaired or partially parsed. Native time is measured call latency, not modeled
attention cost or full wall time.

The paired 20,000-draw parent bootstrap (seed 2026092190) retains both repeats per
parent. Recommended-minus-original F1 is +.00536, 95% CI [−.02190, +.04202]; precision
is +.00697 [−.01429, +.03865], recall +.00551 [−.03569, +.05042]. There is no atomic
independence claim or convincing improvement on this small adaptive panel.

Nineteen paired attempts are valid under both packages: zero F1 improvements, four
declines, fifteen ties. Two formerly invalid attempts become valid, while three
formerly valid attempts become invalid; eight remain invalid. Protocol-involved F1
changes are two wins and one loss—the other two valid-to-invalid transitions had
already scored zero. Thus validity transitions must not be equated with score changes.

## Observed help and harm

For geothermal power stations of at least 10 MW (`ca50668d0ef3ff6c94ca7e9c`), repeat0
changes from malformed output/F1 zero to a valid list/F1 .4667. Repeat1 is unchanged
at F1 .4912. This is a real protocol recovery, not a demonstrated both-valid content
improvement. The other recovery, the reefs question, scores only .0517 and still
contains a long sequence of generic coral-reef categories rather than a clean named
entity list.

For people from the Mataram Sultanate (`b66ec48777b010643b586f03`), repeat0 stays valid
but expands generic Sultan titles: recall stays .2 while precision falls from .0455
to .0164, reducing F1 .0741 → .0303. Repeat1 changes from valid/F1 .1667 to an
unfinished repeated-title list/F1 zero. Both repeats are retained; this is not just
a favorable recovery example.

Inspection of all 11 new capped tails still finds degeneration: Pharaoh novel/module
names repeat, Colin Tilley/Colin Hay repeat, Loyola school names repeat, and country
or Sultan-name sequences cycle. The Orioles answer drifts into a descending sequence
of years rather than contests. The reefs tail also revisits names. These observations
do not support cap-only doubling as the next informative experiment; no outputs were
permissively salvaged. Even a syntactically complete answer can have wrong entity
types or relations.

## Decision and original fan-out caveat

Do not run the remaining map calls merely to match this sampling change: its declared
direct-rescue gate was not met. The original map-union screen had lower F1 (.1661)
than original direct (.1863), with an uncertain difference and lower precision.
Its aggregate native-time saving did not establish an attention-scaling mechanism:
the prior descriptive both-valid/all-EOS subset was 9.6% slower for map, whereas the
full panel included substantial direct-output degeneration. That conditional subset
is not a causal corrected estimate either. Leave the original results intact and
retire further fan-out/decoder tuning here unless a separately motivated training or
task-design question supplies new evidence.

## Immutable provenance

Under `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`:

- New report `analysis-qampari-sampling-control-001.json` and `.md`; JSON SHA256
  `872a1d43cac9e1479a0cfb465b9593b8c6d88e22651bfc51fa4911855b079f50`.
- Run `qampari-sampling-control-001`; terminal `TERMINAL-90b34a8631f4.json`, SHA256
  `eec5c28108a3abcdf46498319689a238be47f347a84f3fc702dfdcd337f28aa5`.
- Qualified analyzer `analysis-source-qampari-sampling-003/analyze_qampari_sampling.py`,
  SHA256 `c5995c3721b0870d4003f86c0777c213147f93c53b02bdd95cb8fade7e9b88d5`.
- Baseline report `analysis-qampari-001.json`, SHA256
  `2e0896c933e8c41817ae6325c0e4a354e3f7bd1b130a87a324e3c95c8b9d15d1`.

The report binds the panel, source, PLAN, grader and every consumed native receipt.
Analysis completed after authenticated terminal on 21 September 2026. Earlier CPU
qualification attempts remain preserved; no live source, raw output, or prior report
was changed and no new GPU work was launched for this analysis.
