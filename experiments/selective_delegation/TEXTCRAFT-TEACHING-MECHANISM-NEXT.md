# Teaching-package mechanism: smallest stronger control

CPU design note, 2026-09-22. No new inputs, training, code changes or GPU job
accepted. Rank **after** accepted062 world transfer,063 TRAIN RL-readiness and
the fresh-root comparison; do not replace RL-readiness priority.

## What the existing evidence identifies

Public-discovery056 versus privileged048 gives 10/16 versus3/16 original-prompt
successes on the exposed eight-root panel. One cheap061 prompt control gives
0/16; that particular instruction did not reproduce the trained behavior. This
does not establish that no prompt intervention could work or identify an
internal reasoning mechanism. Both adapters began from identical step-zero
weights, training seed and hyperparameters.

Target audit004 narrows the supervision difference: initial prompts match on
all32 tasks; query/finish raw-target and unmasked-token multisets match32/32,
craft multisets31/32. Only `textcraft_synth.train.1029` differs: one craft teaches
`raw_t8:6 → output_count12` versus `raw_t8:4 → output_count8` for `t4_i1`.
That is an overproduction difference, not evidence that either full trace fails.

## CPU check of the proposed exclusion

I independently read the two frozen rows.jsonl files and removed **all rows of
train.1029 in memory only**, not from stored artifacts. The task is depth4,
root goal `t6_i4:1`, with15 gold crafts and31 supervised actions.

| Remaining quantity | Privileged | Public discovery |
|---|---:|---:|
| TRAIN parents / rows | 31 /335 | 31 /335 |
| Query / craft / finish targets | 152 /152 /31 | 152 /152 /31 |
| Unmasked target tokens including EOS | 8,055 | 8,055 |
| Prompt tokens | 361,556 | 381,020 |
| One-epoch updates, effective batch16 | 21 | 21 |
| Final update rows | 15 | 15 |

For every remaining parent, canonical actions, raw target-string multisets and
unmasked label-token tuple multisets match exactly. All31 initial prompt strings
match; first target strings nevertheless differ on29/31. Public prompts contain
19,464 more tokens (+5.38%). Thus equal target counts/tokens would **not** mean
equal conditioning, input compute, row order or SGD trajectory.

## Recommended conditional test versus a simple seed replication

If the question is stability of the original result, simply rerun both original
32-task packages with a single new shared seed:366 rows,23 updates,final14.
That is the cleaner direct seed replication, retaining the one known target
difference. Reusing the original seed would only check runtime reproducibility.

For **causal strengthening of the teaching-package contrast**, prefer the
31-task matched-target pair as the next *mechanism* test: remove only this
preidentified TRAIN-audit mismatch from both packages; preserve every other row
and original within-package order. Proposed new shared seed **2026092231**,
fresh identical base/rank8 initialization, existing1e-4 recipe, one epoch,
effective batch16, fixed **checkpoint21**, no checkpoint selection. Freeze both
filtered byte inventories and this stopping rule before any new model outcome.
Do not repeat rows to force23 updates. Cap each training arm30min; assess only
complete21 endpoints. Use the same predeclared original-prompt panel/seeds/caps
for both readouts, keeping any now-exposed fresh panel labeled as such.

This removes the only observed **target-content/multiset** difference and uses
a second training seed at essentially the same small budget. It tests whether
an advantage can survive when different target frequencies/quantities cannot
explain it. It is not an exact replication: removing one task removes31/366
rows (8.47%), changes task coverage and reduces both arms' update dose. A null
or reversed result cannot identify train.1029 as the cause because seed and
dataset also changed; that would require a later crossed seed×exclusion design,
not an immediate expansion now. A positive result supports robustness of the
**history/order-conditioned supervision package**, not pure latent causality,
order-only causality, new-domain independence or a novel learning method.

## Evidence pins and reproducible CPU calculation

All paths are under the selective-delegation-20260921 run store:

- `analysis-textcraft-target-multiset-004.json` SHA256
  `2ec8717a2eb769a1ef42b2471e8acfde7e6c7573ca9bc349a80cc92c7094c84c`.
- Privileged `textcraft-train-inputs-001/rows.jsonl`:
  `caa78390f9d4ac28e600674b26e56375203b72d8cdad1c3f9471da3fb25776a9`.
- Public `textcraft-public-discovery-prototype-001/rows.jsonl`:
  `dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a`.

Calculation: filter exact task_id; group retained rows by task_id; compare
Counters of target strings, sorted-key parsed target JSON, and tuples of labels
with `-100` removed. Sum stored prompt_tokens and unmasked labels; updates are
`ceil(335/16)=21`. No VAL outcomes, native success, token length or model score
enters the removal rule. No other task is eligible for exclusion.
