# Fresh003: secondary hop and exact-document-exposure analysis

This is a pre-readout diagnostic, not a new primary panel. Keep all 64 frozen
fresh003 parents and both repeats in every policy's primary denominator. The
all64 panel is deliberately balanced 32 two-hop/32 three-hop: it does not estimate
the natural whole-development benchmark mixture, and no reweighting is introduced.
The two additional partitions are two-hop versus three-hop (32 parents each), and
any exact TRAIN document overlap versus none (14 versus 50). Exposure comes
from the existing `analysis-document-exposure-001.json`, using literal
`(title, text)` equality, including distractors. Title-only overlap is not the
exposure flag. No-overlap is not a claim of unseen facts or semantic independence.

The existing native auditors do not export complete per-attempt EM/F1 tables.
`analyze_fresh_strata.py` is therefore a small postprocessor of their completed
reports. It uses `analysis-fresh-contract-policy-001.json` from the existing
`analyze_planner.py` then `compare_musique_policies.py` sequence, plus the
completed `analyze_direct_adapted.py` report. It checks released, successful
owners before reading episode outcomes. It inherits the full native audits'
checkpoint and coverage checks, verifies only the consumed plan/episode/final
receipt hashes, and regrades those finals with the same official MuSiQue metric.
It does not reload weights, walk their ancestry, or inspect partial evaluations.

Comparisons include RL minus SFT with the fixed helper, helper-adapted direct
minus its simultaneously collected base control, planner versus direct policies,
and agreement between the original and newly collected direct base controls.
The two direct base acquisitions remain separate; matching seeds do not turn
them into additional independent observations or justify pooling.

Bootstrap paired parent repeat means, resampling connected atomic-component
clusters. Build components on all 64 parents and intersect them with each
stratum, preserving transitive links through parents outside that stratum.
Report complete denominators, EM/F1, missing outcomes, protocol-mediated versus
both-valid wins/losses, and cluster counts. Incomplete metric rates are lower
bounds, not scientific errors; differences between lower bounds are not bounds
on a treatment effect. Missing pairs are never counted as observed wins/losses.

All64 remains primary. The hop and exposure views overlap; intervals are
exploratory and unadjusted. Do not infer an interaction from a gain being
significant in one stratum but not another. A small exposed stratum can suggest
a limitation or follow-up, not establish that training caused memorization.
Exposure strata can differ in hop count and difficulty; a hop-by-exposure count
is included, and their outcome differences are associations, not causal effects
of document exposure.
Metadata-only check before evaluation readout: the exposed 14 comprise two
two-hop and twelve three-hop parents; the unexposed 50 comprise thirty two-hop
and twenty three-hop parents. The imbalance is substantial. There are 47 full
panel components, 29/18 in the hop strata, and 8/41 in the exposure strata;
components can span both strata, so these counts need not sum to 47.
The policies differ in architecture, input tokens and calls; a policy advantage
is not a causal decomposition advantage. Full-panel costs/provenance are reused
from native audits; this script does not manufacture subgroup compute matching.

CPU CLI after both complete audits exist (replace the adapted report filename
with its actual immutable path):

```sh
python analyze_fresh_strata.py \
  --policy-report "$R/analysis-fresh-contract-policy-001.json" \
  --adapted-report "$R/analysis-direct-adapted-001.json" \
  --exposure "$R/analysis-document-exposure-001.json" \
  --cases "$R/fresh-dev-inputs-003/cases.jsonl" \
  --report "$R/analysis-fresh-strata-001.json"
```

Outputs are immutable JSON and Markdown. No GPU calls or case reselection.
