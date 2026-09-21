# Completed TRAIN plan-structure diversity

Across all 1,024 saved roots (256 parents × 4 candidates), 111 groups (43.4%)
contain multiple valid dependency structures; 145 have the same valid structure.
Thus exact-list diversity is not merely structural diversity, but genuine changes
in explicit computation structure are also common.

| Valid structures within group | All groups | Mixed reward | Mixed fully-scored rewards | Nonzero-advantage trajectories | Absolute scalar advantage mass |
|---|---:|---:|---:|---:|---:|
| Multiple | 111 | 40 | 38 | 160 | 87.333 (51.6%) |
| Same | 145 | 38 | 35 | 152 | 82.000 (48.4%) |
| None | 0 | 0 | 0 | 0 | 0 |

There are 78 mixed-reward groups. Of these, 73 have both correct and incorrect
protocol-valid final answers: 38 with multiple valid structures and 35 with the
same valid structure. This subset need not have all four candidates valid.
Nine mixed groups contain an invalid candidate (five multiple-structure, four
same-valid-structure); among groups with all four dependency-valid plans, the
mixed split is 35 versus 34. Invalid alternatives are never silently removed
from the original four-candidate reward or advantage calculations.

Step count varies in 86 groups (33.6%). Dependency edges vary between plans of
the same length in 47 groups (18.4%); 22 groups exhibit both forms. Across plans,
1,010 have valid dependency graphs, spanning 23 canonical structures. The most
common is a two-step chain (715 plans). Two roots fail strict plan JSON parsing;
12 parsed plans have invalid references: 11 self-reference occurrences and one
`#0`, with no forward-reference occurrences.

Method: canonical structure is ordered step count plus the sorted unique set of
earlier-step indices referenced at each step. The exact executor syntax
`#(\d+)\b` is checked against `eval_planner.bind_question`; repeated references
collapse to one graph edge. Other hash-like text remains literal. Advantages
reuse `rl_planner.rloo`; “mass” means sum of absolute scalar advantages, not
gradient norm. Token-weighted absolute advantage is also preserved in JSON.
Full-pass audit hashes bind the consumed root/episode receipts; weights and
optimizer states were neither loaded nor rehashed. No fresh readout was inspected.

Interpretation: roughly half the scalar credit is assigned within groups that
have the same valid graph. This does **not** imply equivalent questions or
semantics. Conversely, variable graphs do not show better decomposition; full-
source finals may repair or bypass helpers. These pre-update rewards span
different parent blocks and are not a learning curve or generalization result.
The evidence supports measuring content and execution noise alongside graph
variation, not automatically expanding search because candidate text differs.

Artifacts under `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`:

- Completed provenance audit: `analysis-rl-fullpass-001.json`.
- Immutable detailed report: `analysis-fullpass-plan-structure-001.json` and `.md`.
- Source receipts: `rl-fullpass-001/batch-0001` through `batch-0016`.

Reproduce with `analyze_plan_structure.py --audit <completed-audit.json> --report
<new-report.json>`; existing reports are never overwritten. Two focused fixtures
cover reference classification, graph/text distinctions, invalid alternatives,
and scalar/token-weighted credit arithmetic. Both tests and Ruff checks passed.
