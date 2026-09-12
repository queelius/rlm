---
date: 2026-09-12
cutoff_utc: "11:15"
status: exploratory_completed_results
questions:
  - When is another helper call worth its cost?
  - Do larger RL updates fix weak learning?
  - Can a numerical task provide a more informative learning problem?
research_store: /project/alex_phd/runs/rlm-research-r4
---

# Extra checking needs to earn its cost; larger RL updates are not yet helping

We have not yet obtained a meaningful RL improvement. The latest experiments
are useful because they distinguish several reasons that a plausible approach
can fail. They also discourage spending more compute on the same unproductive
change.

## Checking disagreements was cheaper than voting, but not clearly better than one pass

A helper classified records in groups of sixteen. We then regrouped the same
records with different neighbors. If the two answers disagreed, a selective
policy asked about that record alone. The routing decision was saved before
collecting the follow-up answers or consulting the answer key.

The comparison used 64 fresh question-category examples and 64 fresh news
examples. We also measured ordinary three-answer voting and asking about every
record individually. All 152 physical requests returned valid answers.

| Policy | Correct question categories | Correct news categories | Question tokens | News tokens |
|---|---:|---:|---:|---:|
| One grouped pass | 62/64 | 55/64 | 6,135 | 8,383 |
| Vote over three groupings | 61/64 | 54/64 | 18,401 | 25,138 |
| Check disagreements individually | 61/64 | 56/64 | 15,598 | 23,025 |
| Ask about every record individually | 60/64 | 55/64 | 53,233 | 50,435 |

Selective checking passed our predeclared comparison against three-answer
voting: it was at least as accurate on each dataset and used fewer tokens.
But that is a narrow result. Against the cheapest single pass, it lost one
question answer and gained one news answer, with substantially more token use.
We should not present this as a general improvement in the RLM.

The mechanism is revealing. Only twelve records triggered checking, and eight
of those original answers were already correct. Checking corrected three
answers and damaged three. The two initial groupings also agreed on seven
wrong answers, which a disagreement-only rule could never select for repair.
Agreement is not proof that a helper is right.

The next useful test should change the source of evidence or the solver,
not merely add another similar vote. It must compete with keeping the original
answer. We will first measure whether a different solver offers complementary
correct answers before claiming that a learned selector would be worthwhile.

## Larger RL updates exhausted the relative reward signal

Our RL comparison samples four answers to each training question and rewards
correct answers. A question teaches this particular update rule nothing when
all four answers receive the same reward: there is no better attempt to favor.

With a ten-times larger learning rate, three updates completed. In the fourth
collection, thirty questions had four correct answers and two questions had
four wrong answers. All 128 samples therefore had zero relative advantage,
and no fourth update was applied. Some wrong answers differed in wording or
category, so answer variety alone does not guarantee useful reward variety.

The last committed model scored 117/128 question categories and 111/128 news
categories, versus 119/128 and 112/128 before RL. This is an explicitly labeled
readout of a model stopped after three updates, not the originally planned
four-update result. The matched three-update reference scored 120/128 and
112/128, with all 256 answers available in both conditions. Thus the larger
updates were also worse at the same update count: they corrected one answer
and broke five that the reference got right. An independent audit decoded all
raw answers and checked the model bindings. The result does not support
increasing the learning rate as the immediate fix.

We are separately testing a reward comparison that can penalize uniformly wrong
groups, and a broader training-data intervention. A paired reward-baseline run
collected its answers but failed on a software import before either model update.
That failure is preserved separately from scientific results; the repair will
reuse the saved answers only if their probabilities still match the starting model.

## The numerical-data pilot found an execution-budget problem

We also tried ten public numerical-data tasks with a different, unmodified
Qwen3.5-4B model. One condition saw a subsampled text representation. Another
could write Python to inspect the full arrays, with three prescribed inspection
turns followed by an answer.

All ten direct attempts returned answers, but none of the Python attempts
reached a final answer: eight ran out of their 512-token code allowance and
two hit the 45-second episode limit. The official score averages were 0.0916
for direct answers and zero for the failed Python episodes. Those scores do
not measure the benefit of a successfully functioning Python-assisted policy.

The next experiment will change how the fixed output budget is allocated to
inspection turns and allow enough execution time. It will retain the failed
pilot and separate unfinished attempts from wrong final answers. We should
establish that this interface works before using it for RL.

## Evidence and limits

These are small, exploratory comparisons, not publication-ready causal claims.
The fresh classification panel was excluded from verified local helper training
and earlier panels; base-model pretraining exposure is unknown. Its question
subset has five observed categories and no abbreviation examples, so it is not
pooled with the older evaluation panel. Logical policy costs reuse a shared
collection of physical calls and do not establish separate policy wall times.

Resume and audit pointers in the external research store:

- `analyses/helper-adaptive-fresh-independent-2026-09-12/REPORT.{md,json}`
- `analyses/helper-adaptive-routing-decision-2026-09-12/DECISION.{md,yaml}`
- `sidecars/helper-hf-lr10x-stopped-step3-unseen-eval-v1/outputs/attempt-001/`
- `sidecars/helper-hf-reference-step3-unseen-eval-v1/outputs/attempt-001/`
- `analyses/helper-hf-lr10x-stopped-vs-reference-step3-2026-09-12/REPORT.{md,json}`
- `analyses/anomalyxl-native-mini-independent-2026-09-12/REPORT.md`
- `sidecars/anomalyxl-native-mini-v1/outputs/attempt-001/episodes/`

GitHub checkpoints preserve source and synthesized reports, not model weights,
environments, or the complete external run store.
