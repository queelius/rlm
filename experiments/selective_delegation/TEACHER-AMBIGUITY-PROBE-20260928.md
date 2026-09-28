# Teacher-label ambiguity: useful diagnostic, not a task-failure theorem

September 28, 2026. Recommendation: a small CPU counterfactual audit is useful, provided it
separates exact-label predictability from successful action choice. No new model collection,
training, scheduler, downloads or frozen-source edits were performed.

## Exact evidence

The [eight-pair audit](TEXTCRAFT-OBSERVATIONAL-AMBIGUITY.md) is backed by
[receipt001](/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-observational-ambiguity-001.json),
SHA256 `212410a12a6bb7e6fa772953664e3f6f8b94a9e785abf3a63ecd1eb27cb744d1`.
Today I reran its sealed CPU driver successfully and verified all five source hashes.
It retains eight exposed VAL roots in native worlds42/43, using a common sufficient base
inventory per pair. All eight initial public prompts are byte-identical; all eight privileged
first-query labels differ. Public root-first and privileged teachers each succeed 16/16.

For the explicitly balanced two-world distribution, recomputation gives privileged first-action
label entropy **1 bit per prompt** and maximum expected exact-label accuracy **50%**, versus
0 bits/100% for public root-first labels. These are full-action statistics, not token-mean losses.
An independently randomized predictor cannot exceed the ceiling without extra information.
They constrain imitation of this teacher—not native success. Both initial queries are legal;
the common root-first policy is already a successful alternative.

## Smallest useful extension

Use the eight already frozen official TRAIN
[diagnostic-B goals](/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-fresh-train-20260928-001/diagnostic/MANIFEST.json),
retaining their diagnostic-only role. Never turn the eight VAL cases into training data or
replace the queued diagnostic readout. Proposed additive CPU construction, not yet executed:

1. Construct two recipe completions per goal using cached worlds42/43. Keep the world42 root
   recipe and its public metadata fixed; in the second completion substitute only unqueried,
   same-tier recipes from43. Use the existing conservative componentwise-max base-supply
   construction. Replay the same root-query prefix in both. Require identical complete prompt
   bytes **and** encoded input IDs, including inventory, feedback, history and remaining budgets.
   Keep world IDs host-only and exclude world-correlated sampler seeds or controller side channels.
   Preserve failed construction/feasibility checks without replacement; these are explicitly
   synthetic counterfactual worlds, not untouched official tasks.
2. Recompute deterministic privileged continuations and observable-query schedules from that
   prefix, keeping tie rules fixed. Record literal targets, semantic action identities and name
   visibility separately. Count label conflicts within exact-prompt groups; estimate
   `H_emp(A|X)` and `sum_X p(X) max_A p(A|X)`. This tests whether even visible names conceal
   oracle routing choices. Different prompts never count as a collision. Singleton groups do
   not demonstrate inferability; two completions cannot characterize the full uncertainty.
3. For each pair, branch on the two teacher suggestions and the public-only next action;
   continue with existing [public next_action](prepare_textcraft_public.py), using only native
   replies. At most 48 bounded CPU continuations; retain the original 96-call/token/context
   limits. Report validity, native success, extra queries and calls. Success supplies a witness
   that an alternative works; continuation failure does not prove that action unsalvageable.
   No optimality claim or exhaustive acceptable-action set is warranted.

## Distinguishing explanations

Later fixed-checkpoint scoring could compare full-target-plus-EOS cross-entropy with the
empirical entropy floor, alongside teacher-forced fit on unambiguous controls and native
continuation outcomes. Use absolute sequence probabilities, not renormalization over supplied
teacher candidates, and the same logarithm base. Excess loss indicates a fit/generalization discrepancy, not uniquely
undertraining; reducing it with additional optimization would support that explanation.
No additional epochs can resolve contradictory labels at identical inputs under the balanced
construction. However, the original single-world SFT data need not contain contradictions and
can be memorized: this audit cannot explain its failures by itself.

If conflicting routes all succeed with comparable costs, treat the conflict as harmless teacher
preference and deprioritize it as a failure mechanism. If conflicts disappear after legal evidence
acquisition, that supports an information-gap interpretation, not a unique learning mechanism.
Common-stock overprovisioning, artificial world mixtures, limited completions and correlated roots
remain limitations. Query-name visibility alone is weaker than distinguishability, while exact
label conflict alone is weaker than evidence of harmful supervision. This is an operational
diagnostic of known partial-observability issues, not new theory. The existing
[methods review](TEACHER-OBSERVABILITY-METHODS-REVIEW-20260928.md) suffices; no additional paper checks
were needed.
