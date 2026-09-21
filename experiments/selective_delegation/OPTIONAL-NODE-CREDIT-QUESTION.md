---
status: exact_cpu_toy_prospective_rlm_question
created_utc: 2026-09-21T20:11:00Z
gpu_experiment: none
novelty: not_established
---

# Can rarely used helpers lose their learning signal?

A recursive model chooses which helpers to call. Some deeper calls may appear
in only one of four sampled solutions. If a training rule needs two examples at
the same depth to compare their rewards, that helper can receive no update—even
when its action is informative. This is a question for future recursive training,
not a diagnosed defect in the current fixed-group paired-RL run.

## What the primary literature establishes—and assumes

[Learning to Correct](https://arxiv.org/html/2604.17912v1) studies multiple attempts
with verifier feedback. Section3 sets attempt-level advantage to zero when only
one trajectory reaches that attempt. AppendixC's variance/unbiasedness argument
explicitly assumes at least two reached trajectories almost surely, or restricts
to those events. Its practical GRPO uses additional normalization; that is not
our raw-gradient toy. This motivates checking optional-node edge cases before
transferring a depth-grouped estimator to recursive helpers. It does not refute
the paper's theorem under its stated assumption or its experimental results.

## Our independent finite calculation

Consider independent root trajectories. A root finishes before needing a helper
with probability `1-q`; otherwise its helper succeeds with probability `p`.
Hold `q` fixed and differentiate success probability with respect to the helper's
success logit. The true gradient is `q*p*(1-p)`.

Use leave-one-out credit among reached helpers, divide by the original number of
roots `N`, and set the sole helper's advantage to zero. Exact enumeration gives

```text
expected gradient = q*p*(1-p) * [1 - (1-q)^(N-1)].
```

The bracket is an additional suppression beyond the genuinely rarer visits.
For four root trajectories and `p=.5`:

| Chance of needing a helper | True gradient | Singleton-zero gradient | Fraction retained |
|---|---:|---:|---:|
| 20% | .0500 | .0244 | 48.8% |
| 5% | .0125 | .001783 | 14.3% |

In this toy, retaining a singleton's reward with baseline zero restores the exact
mean gradient. Ordinary full-root-group terminal RLOO also retains the exact
mean. Neither result promises lower variance or better learning: the rare
singleton update can be noisy. There is no parameter sharing with the parent,
history-dependent visitation, clipping, Adam or GRPO scaling in this calculation.

## What would make this an experiment rather than a mathematical footnote?

First, measure actual helper-visit counts and reward variation in a functioning
recursive policy. If most relevant nodes have useful comparison groups, retire
this concern. If optional nodes frequently occur alone, compare terminal-root
credit, singleton-dropping local credit and a declared singleton baseline at
fixed root rollouts. Record node frequency, gradient mean/variance and root task
success separately. Do not use inaccessible verifier values in policy inputs.

This could explain a specific failure of recursive training, but a correction
to a toy estimator is not a publishable architecture improvement by itself.
The existing flat-actor and crafting comparisons remain higher-priority GPU work.

Reproduction: `audit_optional_node_credit.py` and its four focused CPU fixtures;
immutable source and `REPORT.json` under `R/analysis-optional-node-credit-001/`.
The calculation enumerates all outcomes; it uses no model weights or randomness.
