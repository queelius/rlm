# Code assistance improves fresh-A collection, but reward contrast remains concentrated

Evidence cutoff: **2026-09-29 06:47:06 UTC**. Both collections are complete:
64/64 native-audited attempts, zero unknown outcomes and zero transport failures.
These are the **same warm model using two execution interfaces**, before either
collection's reward-training update. The result is not a training gain.

The unchanged model succeeds on **5/32 raw versus 12/32 assisted attempts**:
eight paired gains, one loss, four shared successes and 19 shared failures.
Assistance uses fewer calls and generated tokens overall. Four of the eight
goals still fail on every attempt with either interface. Three goals per
interface supply within-goal reward contrast, but they are not the same three
goals and do not imply identical or equally useful gradients.

## What is paired?

Both PLANs bind the same eight official TRAIN A tasks and initial inventories,
world42, Qwen3-4B discovery-SFT checkpoint23, FP16 base/FP32 adapter, full
original prompt, sampling and resource caps. Episode seeds are
202609280110–202609280113, four repetitions per target. All 32 actual first
requests have matching task/repeat/seed, full sampling settings, prompt bytes
and input-token IDs; all 32 first response token sequences also match.

The per-call seed is repeat seed + the first eight hexadecimal digits of the
task-ID SHA256 + 1,000 times the global call index. Matching seeds do not keep
subsequent states or outputs identical after execution histories diverge.
There is one realized collection per interface.

“Assisted” means the binder fills ingredient arguments from a recipe already
observed. The model still chooses the item and quantity. It gets no hidden
recipe or stock repair. Original sampled tokens remain the likelihood targets;
the code's replacement ingredient arguments are environment transitions.

## Where the successes and costs change

Each success denominator below is four attempts. Gains/losses compare matched
attempts; the other attempts tie.

| TRAIN target ID | Raw → assisted successes /4 | Gains / losses | Calls, raw → assisted | Output tokens, raw → assisted |
|---|---:|---:|---:|---:|
| 132 | 2 → 4 | 2 / 0 | 136 → 108 | 3,885 → 3,355 |
| 1423 | 0 → 3 | 3 / 0 | 213 → 83 | 5,172 → 2,307 |
| 2281 | 1 → 2 | 1 / 0 | 173 → 177 | 5,332 → 5,778 |
| 1753 | 2 → 3 | 2 / 1 | 227 → 134 | 7,271 → 3,811 |
| 1921 | 0 → 0 | 0 / 0 | 233 → 233 | 8,703 → 7,555 |
| 1051 | 0 → 0 | 0 / 0 | 272 → 216 | 8,180 → 6,158 |
| 2338 | 0 → 0 | 0 / 0 | 278 → 283 | 9,621 → 11,990 |
| 2026 | 0 → 0 | 0 / 0 | 299 → 227 | 9,575 → 6,585 |

Aggregate calls fall **1,831→1,461 (−20.2%)** and output tokens
**57,739→47,539 (−17.7%)**. Prompt tokens fall 6,157,500→4,293,675.
Summed request service time falls 81.6→64.6 minutes (−20.9%); this is not
allocation wall time or training cost. The savings are not uniform across goals.

Native action errors fall **837→479**, while schema errors rise **17→20**.
Missing/extra/wrong ingredient errors fall **551→33**; reported insufficient
inventory errors rise **254→428**. Native validation checks argument correctness
before stock, and later histories differ: this redistribution does not establish
that stock reasoning worsened or isolate a single failure mechanism.

Both interfaces have 28 explicit finishes and four context caps. Unsuccessful
explicit finishes fall 23→16. Calls after first meeting the target quantity rise
23→44, but binder produces more successful episodes and these counts include
ordinary finishing calls; the aggregate rise is not evidence of worse termination.

## What reward training can learn from this batch

The current terminal RLOO rule assigns each attempt its success minus the mean
success of the other three attempts on the same goal. An all-failure group and
an all-success group both have zero direct advantage. All sampled tokens in
a nonzero-advantage trajectory receive its sign, including errors, with a fixed
32-episode denominator. Other goals' updates can still change shared parameters.

| Available contrast | Raw | Assisted |
|---|---:|---:|
| Goals with nonzero advantages | 132, 2281, 1753 | 1423, 2281, 1753 |
| Trajectories with nonzero advantages | 12/32 | 12/32 |
| Calls in those trajectories | 536 | 394 |
| Output tokens in those trajectories | 16,488 | 11,896 |
| Positive / negative trajectories | 5 / 7 | 8 / 4 |
| Calls in all-failure groups | 1,295 | 959 |
| Output tokens in all-failure groups | 41,251 | 32,288 |

Assistance makes 132 all-success, removing that group's terminal reward contrast,
and makes 1423 mixed, introducing contrast there. The success patterns and
advantage signs also change on 2281 and 1753. Assisted 132 consumes a further
108 calls/3,355 tokens with zero advantage because all four attempts succeed.
Thus 20/32 trajectories still have zero direct advantage in either arm, for
different reasons. Fewer credited tokens is not itself a worse or better update;
gradient quality and subsequent task completion remain unmeasured here.

The four common all-failure goals are 1921, 1051, 2338 and 2026: **0/16 per interface**.
Code assistance helps execution without resolving those goals' absent terminal
contrast. This refines the recovery-teaching motivation; it does not prove that
sparse reward alone causes failure or that recovery teaching will work.

## Decision and claim boundary

Keep this as reading-guide and next-decision evidence until the fixed first RL
updates/readouts and extra-SFT control complete. It strengthens existing
execution-assistance evidence and explains why collection accuracy alone is
insufficient to assess learning. It does not change the advisor deck's two main
findings or justify a third learning headline.

These are eight prospectively selected official TRAIN goals in one shared
recipe world, with overlapping component recipes possible. Four repetitions
are not four independent problems. No held-out generalization, planning
mechanism, learned co-adaptation or recursion benefit follows. No p-value or
interval treating 32 attempts as independent is reported. Neither B outcomes
nor post-update readouts enter this report.

[RESULTS.json](RESULTS.json) retains per-root/per-attempt outcomes, advantage
patterns, costs, errors, exact common bindings and source/receipt hashes.
The two primary sources are
[raw collection](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-fresh-rl-20260928-002/raw/collect-0001/SUMMARY.json>)
and [assisted collection](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-fresh-rl-20260928-002/binder/collect-0001/SUMMARY.json>),
with their completed native audits as scoring authority.

Verification checked 29 small source/manifest/summary/audit hashes and 192 native
receipt bindings: 64 episodes, 64 root nodes and 64 first calls. Per-root costs,
scores, terminations, public error categories and RLOO advantages were
recomputed. No broad native replay, model/ancestry rehash, GPU/model execution,
environment mutation, prior-report/slide edit, queue/owner change or Git
operation was performed. Only this README and RESULTS.json were added.
