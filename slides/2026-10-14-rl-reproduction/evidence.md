# Evidence for the advisor deck

Evidence cutoff: **14 October 2026, 18:10 UTC**. The deck reports our completed
fixed-endpoint SPO learning result, not a selected intermediate checkpoint.

## Claim map

| Visible claim | Primary project evidence |
| --- | --- |
| Our final test is 12,281/21,104 = 58.192760% | [Final independent audit](../../docs/spo-reproduction-2026-10-08/evidence/final690-audit-20261014.json), `iterations.690.test` |
| Matched start is 8,394/21,104 = 39.774450%; gain is 18.418309 points | Same audit, `original_SFT_baseline_comparison`; original reference was independently regraded earlier |
| Fixed endpoint is 690 collections / 11,040 updates | [Execution receipt](../../docs/spo-reproduction-2026-10-08/evidence/final690-execution-20261014.json) and [final report](../../docs/spo-reproduction-2026-10-08/final-result-20261014.md) |
| All 33,024 final responses and 21 metrics checked | Final audit: three splits, seven metrics each; test 21,104, validation 5,968, training subset 5,952 |
| Complete final state preserved with matching hashes | [Preservation receipt](../../docs/spo-reproduction-2026-10-08/evidence/final11040-preservation-20261014.json); 14 files, 17,603,255,667 bytes |
| Whole scheduled test curve, no best-score selection | [CSV](learning-curve.csv) and [point-by-point provenance](curve-provenance.json); all 70 scheduled points plus matched baseline |
| Author released result validates the evaluator | [Protocol audit](../../docs/spo-reproduction-2026-10-08/protocol-audit.md) and final report distinguish 56.69% released-weight evaluation from our own training |
| Transfer to RLM is a proposed research direction | Slides 4–5 are interpretation and proposed comparisons, not additional measured results |

The completed research report retains the full historical record. The advisor
deck focuses on the completed run and interpretation-relevant qualifications.

## Paper and implementation

- [Guo et al., Segment Policy Optimization](https://arxiv.org/html/2505.23564v2):
  the method and its published benchmark comparison.
- [Official source at the pinned commit](https://github.com/AIFrameResearch/SPO/tree/6c5f94723c20e4de68578545b8bb98186e956797).
- [Author short-chain-of-thought comparison figure](https://github.com/AIFrameResearch/SPO/blob/6c5f94723c20e4de68578545b8bb98186e956797/figures/compare-baselines-short-CoT.png):
  paper target 56.7%; starting reference 40.3%.
- [Local protocol audit](../../docs/spo-reproduction-2026-10-08/protocol-audit.md):
  model, dataset splits, source pins and comparison conventions.

## Evaluation and plotting rules

All test scores mean correctness across sampled answers: 1,319 questions times
16 answers, temperature 0.35, effective top-p 1.0. They are not best-of-16,
majority-vote or greedy scores. The 21,104 answers do not constitute independent
training replications.

The curve starts with the separately evaluated original SFT model at zero
updates. Native iteration 0 is already after 16 updates; ordinary scheduled
iteration i maps to 16(i+1) updates. The special final hook at iteration 690
instead verifies exactly 11,040 updates. The CSV uses this distinction.
Lines connect discrete evaluations; they are not unsampled measurements or
uncertainty bands.

Each scheduled point has its independent receipt path and SHA-256 in
`curve-provenance.json`. Paths point into the external research store;
Git carries the derived data and final portable receipts, not all raw outputs
or model weights. The source native root is
`/home/atowell/research-runs/rl-paper-reproduction-20261012/spo-bos-only-from-sft-attempt1/native`.

## Scope of the conclusion

This is a performance-level reproduction from the original supervised starting
model, with one seed and disclosed process-local BOS token alignment and
finite-zero BF16 numerical adapters. Native EOS omission remains. Historical
implementation equivalence and author checkpoint-selection details are
unresolved. The final endpoint for our run was fixed in advance.

The evidence supports a real measured gain and our ability to execute and audit
the learning workflow. It does not establish superiority to the paper,
cross-seed robustness, a causal explanation for each implementation choice, or
transfer to recursive tool use. Future RLM experiments may use another RL method.

## Updating the deck

Update the visible result, plot data, presenter notes, speaker guide, provenance
and cutoff together. Keep the predetermined endpoint and all scheduled points;
do not replace it with a more favorable test score. Rebuild, inspect every
changed page and check the private notes before publication.
