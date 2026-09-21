# Local compositional-task assets

CPU-only inventory, 2026-09-21. No models launched, third-party code executed, or assets downloaded. This is an asset audit, not a new experiment plan.

## Decision

There is no need to build another exact relational evaluator. The B05 ETL generator is the strongest reusable implementation, and the purchase-record protocols already implement same-facts co-located versus cross-partition joins. However, none of the inspected generators independently controls text length, compositional depth, and linguistic template. A claim about held-out composition depth would require a real generator extension, not merely a new runner.

Do not repeat the bicycle/helmet purchase task or B05 singleton extraction as a new research direction. Both have substantial local evidence. Repartitioning or adding surface variants is scientifically useful only if it tests a specific unresolved mechanism and preserves matched facts, information access, and budgets.

## Reusable implementations

### B05: exact versioned ETL relations and three-stage join

Source directory:

`/project/alex_phd/repos/structured-decomposition-benchmark/data/candidates/sol-generator-tournament-v1/builders/05-program-dataflow-database`

The dependency-free `src/etl_catalog` package exposes:

- `generator.generate(seed, difficulty)` and `extract_child(root, child_index)`.
- Independent child and root solvers in `solver_reference.py` and `solver_independent.py`.
- `combiner.combine_reports(root, displayed_reports)`.
- `render_direct`, `render_child`, and `render_synthesis` in `render.py`.
- Strict child/root response grading, public projections, trusted records, and semantic fingerprints in the adjacent modules.

Each local result is a complete eligible implementation relation. Global composition joins three fixed stages by matching adjacent formats, enforcing cost/latency/feature/capacity constraints, and applying an exact lexicographic ranking. This is genuine cross-relation composition rather than a single-record lookup.

`StructuralConfig` supports candidates per stage 3–20, history depth 1–10, check revisions 1–8, format count, required feature count, and chain/hub/sparse format topology. History depth means change-log depth, not the number of compositional steps. There are always three stages and one renderer family. The generator plants two feasible implementations per stage, with an eight-combination feasible spine; generated roots are guaranteed feasible. Consequently, neither arbitrary join depth nor natural infeasible examples are ready-made controls.

Existing small adapter:

`/project/alex_phd/runs/rlm-research-r4/sidecars/b05-recombination-feasibility-v1/prepare.py`

It already imports the package, compares exact solvers, builds public/trusted projections, and audits tokens. Its small configuration uses three candidates, history/check depth one, three formats, and one required feature. Reuse these functions and the present native collector rather than the historical launch wrapper. Default large instances have roughly 78 KB direct prompts, so default difficulty settings are not appropriate for a cheap 4B pilot.

Tests and recorded evidence: `tests/test_family.py`, `artifacts/fuzz_summary.json`, `FAMILY.md`, `REVIEW.md`. The saved fuzz report records 300 passing instances per named difficulty; this inventory did not rerun it. Tournament reviewers requested revision, including concern about the recognizable planted spine and prompt size. This family is a candidate, not an admitted sealed benchmark.

### Purchase records: exact same-facts partition controls, already tested

Sources:

`/project/alex_phd/runs/rlm-research-r4/sidecars/root-native-partition-join-v1/protocol.py`

`/project/alex_phd/runs/rlm-research-r4/sidecars/root-fresh-join-externalization-order-v1/protocol.py`

The first exposes `worlds`, `oracle`, strict `extraction` and `score`, plus prompt/evidence construction. Four engineered worlds contain 12 customers, 48 purchase records, four products, and a two-product intersection query. Each world has co-located and cross-partition versions of the same facts. The later protocol supplies eight fresh worlds and `ordered_world(world, "random" | "reverse")` for same-facts order changes. Its oracle can intersect more products, but its world/query generator uses two targets; changing the oracle argument alone is not a validated depth generator.

These are the smallest adapters for exact partition sensitivity. Existing scoring rejects malformed, unsorted, duplicate, or out-of-domain outputs and distinguishes missing output from returned invalid output. They do not offer held-out operator grammars or surface templates. `join_taskset.py` is an empty registry, not the generator.

### Six-operator composition: existing independent numerical oracles

`/project/alex_phd/runs/rlm-research-r4/sidecars/root-operator-composition-transfer-v1/ct_protocol.py`

`specs(index)` defines count, distinct users, weight sum, threshold users, maximum weight, and conditional weight. `answer(records, labels, row)` and `enumerated_answer(records, labels, row)` are independently structured oracles. Conditional weight requires summing one category's weights for users who have another category, so it is a real cross-record join. This is a fixed six-operator vocabulary with fixed question templates, not an arbitrary-depth expression generator. Its upstream text/label corpus provenance would need a separate manifest/license check before reuse.

### Route portfolio: exact composable Pareto tables, limited depth axis

`/project/alex_phd/repos/structured-decomposition-benchmark/src/structured_decomposition_benchmark/prototypes/route_portfolio.py`

`generate_problem(seed, decision_depth=2)` returns a problem and trusted receipt. Depth can only be two or three, with three fixed regions. Child complete-enumeration and label-setting solvers independently compute Pareto frontiers; root exact-table and raw-route solvers independently verify the optimum. Public renderers, strict graders, and `combine_displayed_reports` already exist. Tests are in `tests/test_route_portfolio_prototype.py`.

This is a small-adapter option for composing exact trade-off tables. Local binary-decision depth changes search space and record length together; it is not independently controlled recursive composition depth. The family is explicitly a prototype, not an admitted dataset.

## Existing results that constrain a new proposal

All paths below are relative to `/project/alex_phd/runs/rlm-research-r4`.

- `analyses/root-fresh-join-externalization-order-live-2026-09-10/REPORT.md`: file-only 12/16 correct versus inline-plus-file 9/16, but availability was 16/16 versus 10/16. Among the ten fully observed pairs, file-only had zero wins and two losses. No child calls occurred. This does not establish a file-only accuracy advantage.
- `analyses/b05-singleton-replication-independent-2026-09-13/outcome-001/FINDINGS.md`: singleton extraction 22/24 versus vector extraction 10/24, with 12 wins and zero losses. Singleton input cost was substantially higher: 207,004 versus 56,088 tokens. A previous six-context pilot was 10/12 versus 4/12. Granularity works here, but costs were not matched.
- `analyses/b05-public-normalization-fresh12-independent-2026-09-13/outcome-001/FINDINGS.md`: public normalization improved exact accuracy from 1/24 to 9/24; all normalized exact successes were width six, with none at widths 12 or 20. Representation, length, and formatting changed together. This is not evidence of learned recursive composition or an isolated history-depth effect.
- `analyses/b05-flat-selection-rl-dose10-independent-2026-09-13/FINDINGS.md`: held-out balanced accuracy rose about 0.028, but exact correctness stayed 0/18, and recall/F1 declined. An automatic training-dose extension would repeat a weak direction.
- `claims/operator-composition-transfer-limit.md`: SFT24 scored 9/48 strict exact versus SFT6 6/48; composed subsets were 4/24 versus 3/24, but neither checkpoint had a verified task-faithful composed success. Correct numbers sometimes resulted from the wrong operator.
- The structured-decomposition benchmark's rendezvous DEV003 already contrasts advisory and binding exact reports: direct 3/10, advisory 3/10, binding 10/10, while wrong binding reports scored 0/10 and were followed 10/10. Binding can measure obedience to reports, not independently establish better reasoning. Later natural-worker revision had little repair headroom and harmed correct reports.

## Provenance, licensing, and split constraints

Structured-decomposition repository inspected at clean commit `c7ee38127e5f37c327b77f518a21f0bab151ff33`. No LICENSE file or license declaration was found in its package metadata; do not assume a public redistribution license. These are local research assets. Root RLM checkout was at `1025c4da3fa32106f149807ff99431125cb45e2d` when inspected.

Selected source SHA-256 values:

- B05 `src/etl_catalog/generator.py`: `6a99319ee1ce129ef603d1f46d2f9b682ff0290f134c641b497537eaba0171fb`
- Original purchase `protocol.py`: `6d12052bcdf2b930a8d2f39216d24407e61594c5d4cbb53f0b298c9a886e49c2`
- Fresh purchase `protocol.py`: `3293dfc0a4a945b81c51c004b4ed0f06b6fa24cb0cea21c7263ea669d7544989`
- Operator `ct_protocol.py`: `951e6e1c4f516427cebf621d7c43faa8e645dc756358c923b084336c2cbb351d`
- Route portfolio module: `5ed5356c51879b12afcfd182b76a803503533d8bf4043605a8cde31804927ade`

The SDB split policy is `data/specs/SPLIT_POLICY.md`: keep every variant, repeat, decomposition, and partition of a root within one split; distinguish candidate/calibration/development exposure; split semantic templates, motifs, or parameter regimes rather than merely RNG seeds when claiming structural generalization. No inspected SDB family currently supplies an admitted sealed evaluation set. The historical purchase worlds and B05 results are already exposed development data. Fresh seeds alone would establish new instances, not new operators, templates, or depth generalization.

The useful next decision is whether an unresolved question can be answered using these existing exact semantics with a narrowly specified new surface or information-access intervention. If the hypothesis specifically requires independent depth extrapolation, budget and describe the necessary generator extension explicitly rather than calling it a thin adapter.
