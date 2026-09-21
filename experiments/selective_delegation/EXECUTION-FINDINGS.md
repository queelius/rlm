# Execution diagnostic: no demonstrated isolation advantage

The completed 2×2 probe compared model/reference questions with bundled/isolated
helpers on 26 two-hop parents, two repeats each. Reference questions are privileged
human annotations, not outputs available to a deployed controller. All 208 planned
episodes count; two reference-isolated helper parsing failures score zero.

| Arm | EM | F1 | Correct / 52 | New calls | Input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| Model bundled | 46.15% | 55.90% | 24 | 104 | 300,352 | 11,206 |
| Reference bundled | 53.85% | 63.59% | 28 | 104 | 299,364 | 10,700 |
| Model isolated | 46.15% | 55.00% | 24 | 156 | 430,578 | 2,608 |
| Reference isolated | 50.00% | 60.13% | 26 | 154 | 422,502 | 2,006 |

Paired parent-bootstrap EM differences (percentage points, exploratory 95% CI):

| Contrast | Estimate [95% CI] |
|---|---:|
| Reference − model, bundled | +7.69 [−1.92, +19.23] |
| Reference − model, isolated | +3.85 [−11.54, +19.23] |
| Isolated − bundled, model | 0.00 [−9.62, +7.69] |
| Isolated − bundled, reference | −3.85 [−13.46, +3.85] |
| Reference isolation effect − model isolation effect | −3.85 [−19.23, +13.46] |

The intervals all cross zero. Isolated execution for the next planner experiment
is an explicit-execution diagnostic, not selection of a demonstrated winning arm.

## Cost accounting

Actual acquisition: **518 calls, 1,452,796 input + 26,520 output = 1,479,316 tokens**.
There were no transport failures or unknown-usage calls. The planned count was 520:
two helper parse failures prevented two final calls. Reused historical checkpoints
are 26 unique calls / 73,391 tokens, not new GPU work. Charging them once across
this diagnostic gives 544 calls / 1,552,707 tokens.

Hypothetical deployment charges one checkpoint for each arm/parent/repeat attempt:

| Arm | Calls across 52 attempts | Total tokens | Tokens / attempt |
|---|---:|---:|---:|
| Model bundled | 156 | 458,340 | 8,814.23 |
| Reference bundled | 156 | 456,846 | 8,785.50 |
| Model isolated | 208 | 579,968 | 11,153.23 |
| Reference isolated | 206 | 571,290 | 10,986.35 |

Isolation reduces generated tokens but increases total tokens by repeating source
input. Summing hypothetical deployment costs is not physical acquisition cost.

## Intermediate answers and final synthesis

Against the composed question's gold, isolated last-helper EM/F1 is 21.15%/32.21%
for model plans and 34.62%/46.30% for reference plans. Model finals change zero
correct helpers to incorrect final answers and recover 13 EM matches across seven
parents. Reference finals lose two matches on one parent and recover ten across
five parents. These are diagnostics, not a last-helper policy comparison: the last
model question often asks a different target, and verbose correct entities can
fail exact match. Keeping a full-source final remains justified by these results.

- `01432ceceb7dc7d7f0700f8f`, both repeats: reference helper correctly answers
  Hudson Motor Car Company's end year, 1954; final returns 1932, the car-line end
  year and original checkpoint answer. This suggests an old-answer conflict,
  without proving an anchoring mechanism.
- `799582c27785a18ccd47522f`, both repeats: model helper 1 supplies the requested
  musical elements; helper 2 answers its different album question, “Graduation”.
  Final correctly returns “melody and chord progression”.
- `e22e667dbb3a7eec260d57ed`, both repeats: reference helper cannot provide the
  population of downtown Santa Rosa; final correctly gives 175155. Full source
  and the already-correct old checkpoint remain available.
- `db78bb873cc97a8f109ee344`, repeats 0/1, reference isolated: helper 2 emits a
  plain explanatory sentence containing the answer, not required answer JSON.
  Both episodes stop before final and score zero, without repair. Excluding this
  parent only as sensitivity gives zero EM isolation effect in both conditions;
  the complete planned denominator remains primary.

## Method, limits, and reproducibility

Regrade native final/helper receipts with official alias-aware `probe.grade`.
Average the two repeats within each parent, resample 26 paired parents with
replacement 20,000 times, seed **2026092191**, and use percentile 95% intervals.
The reusable script specifies Python `random.Random.randrange`, PLAN parent order,
and linear percentile interpolation; preserves source/cases/receipt/code hashes;
and refuses to overwrite either its JSON report or markdown sibling.
This fully specified implementation supersedes the preliminary chat bootstrap:
its bundled reference-gain upper endpoint is 19.23 rather than 21.15 percentage
points (finite-bootstrap RNG/order difference); the conclusion is unchanged.

Repeated samples share one fixed checkpoint; they are not independent end-to-end
runs. The 26 parents form 25 connected atomic-component clusters (largest two),
so parent-bootstrap intervals are exploratory, conditional, and not fully
independent generalization guarantees. No multiple-comparison correction is used.

Reference second questions contain literal `#1` for 26/26 parents; model questions
do so for 0/26. Isolation changes question visibility, sequential calls, dependency
binding and output format together. It is not a clean isolation-mechanism estimate.
All original finals retain the shared old checkpoint and provisional answer.
The later title-index-only planner without a provisional answer is an architecture
change; base and SFT must be compared under that same new contract.

Artifacts (external; no raw traces are copied into Git):

- Run: `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/execution-probe-001`
- Cases: `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/inputs-001/cases.jsonl`
- Original checkpoints: same campaign's `pilot-001/checkpoints` and `pilot-001/calls`.
- Sealed execution source: same campaign's `source-004/execution_probe.py`.
- Cases SHA256: `0aebb983cf5c91b38f8bbee93e6fbf1ebe86588fd59048f54c140ae17bf95ef8`.

Run from this source directory using an existing CPU-capable Python environment:

```bash
python analyze_execution.py --output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/execution-probe-001 --cases /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/inputs-001/cases.jsonl --report /absolute/new/report.json
```

Checkpoint-replay overlays are intentionally outside this bounded analyzer.
