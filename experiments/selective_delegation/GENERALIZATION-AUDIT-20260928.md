# Generalization audit — 2026-09-28

The six-panel result uses **48 distinct goal roots, not 48 quantity variants**.
However, root-disjoint evaluation does not imply unseen ingredients, recipes,
graph motifs, vocabulary or domains. The defensible claim is transfer to new
goal roots and altered recipe assignments within one synthetic generator.

## Identity and overlap

| Metadata | SFT tasks | Six-panel VAL, world42 |
| --- | ---: | ---: |
| Task IDs | 32 | 48 |
| Distinct root-item strings | 30 | 48 |
| Distinct root-plus-requested-quantity | 32 | 48 |
| Distinct exact initial inventories | 32 | 48 |
| Distinct rooted, name-erased graph shapes | 21 | 39 |
| Shapes retaining ingredient quantities/output yields | 29 | 48 |

The two repeated SFT roots are `m4_i3` and `a2_i4`; quantities and inventories
differ. VAL has no repeated root in any world. TRAIN32 versus VAL48 overlap is
zero for goal roots, root-plus-quantity, and exact initial inventory. Removing
only the root name leaves 48 different named dependency closures in every VAL
world: these are not identical native recipe problems with cosmetic root labels.

Nevertheless, **4/48 VAL roots were supervised prerequisite queries** in both
teachers: `t8_i2_18`, `c9_i3_19`, `m6_i2`, and `c4_i2_24`. Both teachers queried
the same 126 distinct craftable products. In world42, 43/48 VAL tasks share at
least one exact craftable recipe with SFT. Thirteen of 48 have an unweighted
rooted graph shape present in SFT; two retain an SFT shape even when quantities
and yields are included. This is compositional reuse, not recipe-disjoint testing.

Graph comparisons use exact directed-graph isomorphism, preserving shared nodes,
root/base/product roles, and optionally edge quantities/product yields. They
omit item names, requested goal quantity and starting inventory. Native graph
inspection is an explicitly privileged **offline audit**, never actor input.

## Cross-world meaning

All four worlds retain the same 1,452-item namespace and the same 48 goal roots.
Numeric suffixes distinguish generated items; they are not alternative aliases
for one native item. Across all 288 same-root pairs from six world pairs:

- Zero have an identical root recipe, full named closure, or initial inventory.
- Twenty-five retain the same unweighted topology; zero retain the same
  quantity/yield-weighted topology.
- Per-world VAL weighted-shape counts are 48, 47, 48 and 47 for worlds42/50/51/52.

Only 0/48, 2/48 and 0/48 tasks in worlds50/51/52 contain any exact named SFT
recipe, despite substantial name reuse. Thus these worlds provide changed-recipe
evidence, not merely renaming—but not new-vocabulary or new-generator evidence.

## Dependence sensitivity

Task ID and literal root are one-to-one, so reclustering by root alone changes
nothing: discovery binding remains +7.55 percentage points, with the reported
48-task/root depth-stratified interval [+5.08, +10.16].

Shared craftable dependencies connect the roots into five components matching
the generator's material families: sizes 8, 9, 9, 11 and 11. An additional
20,000-draw component-resampling sensitivity, retaining all worlds, fit seeds
and rollout repeats, gives [+4.40, +10.73]. It uses a pooled numerator/denominator
to retain the task-weighted estimand. Unlike the original interval, it does not
fix depth-stratum proportions. With only five constructed families, this is
unstable descriptive sensitivity, **not calibrated confirmatory uncertainty**.
Neither calculation establishes independence across worlds or fitted models.
Original findings and intervals are unchanged.

## Fresh A/B exclusion wording

Official TRAIN contains 2,522 rows/1,021 root strings; official VAL contains
632 rows/**256 root strings**, with zero root overlap. Fresh preparation excludes
the actual root strings, irrespective of requested quantity—not merely IDs.
Its frozen exclusion inventory protects 286 roots including those 256 VAL roots.
Correct wording: “excludes every root represented by the 632 official VAL rows.”

Fresh A/B each have eight unique roots, with no protected-root overlap; none
was among SFT's 126 queried products. But all 16 tasks share exact lower-level
SFT recipes. A/B share 31 craftable products across 12/64 task pairs; no B root
is in A's native closure. B remains a held-away **TRAIN diagnostic**, not a new
official holdout or graph-disjoint benchmark. Historical exposure outside the
frozen inventories is not ruled out.

Evidence and reproduction: [compact JSON](GENERALIZATION-AUDIT-20260928.json) and
[CPU script](analyze_generalization_20260928.py). Run the script with
`--output /tmp/generalization-new.json` in the existing training environment
(networkx3.6.1), with `CUDA_VISIBLE_DEVICES=`. It verifies four native world hashes,
task manifests and the cached breadth-details hash; no full rollout traces or
model ancestry are read. No GPU jobs, live sources, or Git state were changed.
