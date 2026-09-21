# Giving the final model a plan was competitive with executing it

On this small short-context development panel, we have not established that
running the helper steps improves answer accuracy over supplying the plan alone.
This is not evidence that the two methods are equivalent or that long-context
RLMs are unnecessary. Every final answerer here can read all original documents.

| Root training | Execute helpers, then answer | Plan only, then answer |
|---|---:|---:|
| Supervised examples | 53/128 | 56/128 |
| Supervised examples plus16 RL updates | 56/128 | 53/128 |

The unchanged direct-answer baseline is54/128. All conditions use the same64
questions, two repeats and full-source base final model. Plan-only reuses the
actual root-generated question list, removes the helper report, and generates
a new final answer with the original seed. It is not a direct-answer prompt.

For the supervised root, removing helpers adds2.34 percentage points of exact
match, with a paired component-bootstrap interval of−6.52 to+11.02. For the
RL root, removal loses2.34 points, interval−10.48 to+5.56. These intervals do
not establish an accuracy improvement for either direction.

The effects differ on individual examples: removing helpers produces11both-valid
wins and9losses for the supervised root, plusone recovery from a saved dependency
failure. For the RL root there are9both-valid wins and12losses. The final model
can both benefit from and be misled by the helper report.

## Does RL increase the usefulness of helpers?

The exploratory interaction is+4.69 percentage points in favor of executing
RL plans rather than supervised plans, interval−0.72 to+11.29. This is an actual
paired difference-of-differences over the same parents, not a comparison of two
significance labels. It remains uncertain and does not show faithful reasoning.
Token F1 interaction is+2.72 points, interval−3.48 to+9.86.

Plan-only costs about3,391tokens per attempt, counting its saved root generation
once plus the new final. That is33.1% of the supervised executed policy's tokens
and32.1% of the RL executed policy's tokens. These are hypothetical deployment
token totals from actual calls, not a measured wall-time speedup. Reused training,
root acquisition and additional research controls are not free research compute.

## Checks and decision

All256 plan-only calls and16 preselected factual replay controls completed,
with no runtime errors or missing outcomes. All16 factual controls reproduce
their saved emitted tokens exactly. The collection used812,074 new tokens.
All128 plan-only finals per policy were valid; the source's single supervised
dependency failure remains an explicit source-policy zero, not a missing final.

Combined with actual TRAIN reward improvement and the modest reward-noise
diagnostic, one bounded RL continuation16→24 remains informative. It is now
accepted, with a fixed checkpoint24 readout. But this result raises the bar for
further short-context planner training: small gains must be compared with these
cheap baselines, not presented as evidence that more decomposition is better.

Primary report: `R/plan-only-001/ANALYSIS.json`, SHA256
`c54ced10aa376ba0c93a99d7f5b9c5583ad8192bdd5cea61af9f809293becf11`.
Secondary costs/interaction: `R/analysis-plan-only-secondary-001.json`, with its
formula, inputs,20,000draws and bootstrap seed2026092174. Primary design and
source hashes are in [PLAN-ONLY-DIAGNOSTIC.md](PLAN-ONLY-DIAGNOSTIC.md).
