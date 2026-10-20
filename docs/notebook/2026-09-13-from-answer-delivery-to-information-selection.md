# From delivering answers to choosing useful information

Exploratory update, September 13, 2026, 01:00 UTC. This follows the
[previous checkpoint](2026-09-12-rl-dose-and-local-delegation.md). The research
question is whether a small model can learn to inspect information, select what
matters, and delegate useful work—not simply produce a more polished answer.

The clearest new finding is a harness change, not a training breakthrough:
letting Python organize the available records before the model selects from them
improved selection on a fresh test while roughly halving input length.
The selected reinforcement-learning update (RL: learning from rewards for
attempted answers) improved exact answer delivery on one frozen panel, but our
selection-training experiments have not yet produced reliable recovery of all
the correct records.

## Organizing information helped on fresh cases

Our generated database task gives the model candidate implementations, their
updates and checks, and a policy describing which candidates qualify. The model
must return every qualifying candidate and no others. The local experiments
below test that selection step, not the complete database-planning problem.

The new harness uses Python to combine the public updates and show each
candidate's current values and latest checks together. It retains every
candidate. It does not consult the answer key, decide eligibility, or filter the
list. The model still applies the policy and chooses the answer.

For example, imagine a candidate whose original capacity is 10, with an applied
update adding 2 and a failed security check in another table. Python presents
“capacity: 12; security check: failed.” The model must still decide whether that
candidate meets the policy. This is a simplified illustration, not a training
example or a verbatim experimental record.

| Comparison | Original tables | Organized records |
|---|---:|---:|
| Initial test: correct complete sets, ignoring order; 9 cases × 2 attempts | 0/18 | 4/18 |
| Fresh test: correct complete sets, ignoring order; 12 cases × 2 attempts | 1/24 | 9/24 |
| Fresh test: fraction of qualifying candidates found | 82.3% | 94.9% |
| Fresh test: fraction of selected candidates that qualify | 63.1% | 81.1% |
| Fresh test: input tokens across all attempts | 101,706 | 48,798 |

The fresh test had eight gains and no losses in complete-set accuracy, spread
across five of twelve cases. All calls returned usable answers. The two attempts
per case are repeated measurements, not independent problems. The fresh cases
were fixed before either condition answered them and did not share generated
candidate identities with the earlier panels.

There is an important size limit: all exact successes were on six-candidate
cases. Organized input solved 9/10 attempts at that size, but 0/10 with twelve
candidates and 0/4 with twenty. Some larger answers became much better without
being complete. One twenty-candidate case instead returned every candidate,
including ten with failed checks. Input length and explanatory wording both
changed, so we cannot yet attribute the benefit solely to resolving updates.

These counts evaluate the selected set regardless of list order. The strict
contract additionally requires sorting: on the fresh test, strict successes
were 0 versus 2, not 1 versus 9. We keep that interface problem separate from
the question of whether the right candidates were selected.

The fresh cases also include longer update histories, but the history groups
contain different problems. They do not establish a causal effect of history
length. This is replication within a generated task family, not transfer to a
new dataset. It supports a useful hand-designed division of work between Python
and the model; it does not demonstrate learned delegation or recursion.

## A small RL gain reached new conversations, but its meaning is narrow

In this task, the model must recover a particular earlier reply from a long
conversation—for example, the third reply to a repeated request—and return it
exactly. We started with the 4B controller previously trained on 32 worked
examples of using Python to search the conversation. That earlier training was
supervised fine-tuning (SFT). The new experiment instead learned from rewards
for its own attempted answers, updating a small adapter and assigning direct
credit only to the final-answer tokens, not the Python search code.

The higher-learning-rate update answered 27 of 32 new conversation questions exactly,
compared with 24 before the update. There were five gains and two losses. All
outcomes were available, and an independent audit checked the actual model
requests, returned tokens, scoring, and execution paths.

| What was measured on the same 32 new conversations? | Before RL | After RL |
|---|---:|---:|
| Exact final answer, including required whitespace | 24 | 27 |
| Final answer after ignoring edge whitespace | 29 | 29 |
| Correct target appeared in the Python result | 29 | 29 |
| Generated tokens | 17,583 | 21,302 |

Most gains concerned delivering information already found. Four of the five
new successes followed unchanged retrieved information; one used a better search
path. These retrieval trials used Python but no helper calls. One regression
failed before a usable Python action; another added unwanted
whitespace after the same correct retrieval. This is evidence that reward training
can affect a useful behavior, but not yet that it teaches better decomposition.

The new panel was frozen before either model answered it, with eight questions
for each requested occurrence from first through fourth. These are new contexts
for this project within the same public repeated-request task (MRCR), not a different task or proof
that the base model never encountered the data. We selected the learning rate
using earlier development results. The 32 examples are too few to establish a
robust general improvement; we retain the losses and the earlier failed replication.

The earlier comparison helps explain why we changed direction. On the original
training attempts, one smaller update raised exact answers from 7/32 to 17/32;
a ten-times larger learning rate reached 24/32. Yet the two consistently wrong
search strategies stayed wrong. Their answers had no relative reward variation,
and the loss trained only the final response, not the search action.

## Rewarding selection changed behavior, but did not solve the task

In the generated database task, a helper must select every implementation that
meets a stated policy. Our new training batch contains nine local problems with
two model attempts each. None of the pairs provides a difference under strict
all-or-nothing success: both answers fail. However, four pairs differ in which
items they correctly include or exclude.

We therefore use a graded reward that values finding eligible items and rejecting
ineligible items equally when both classes exist. This discourages the shortcut
of returning every ID. Python checks the selections against the generated answer
key; the key is not shown in the prompt. Training applies credit to the model's
sampled ID-list tokens, not to a supplied ideal answer.

We completed two single-update comparisons from the released 4B model, each
starting from exactly the same newly initialized small adapter. One used a
ten-times larger learning rate. Each training run took about 48 seconds including
ownership and checkpoint handling; the training program took about 35
seconds. Each was followed by 72 model calls: before and after training, on both
the original nine stages and nine separate stages, with two attempts per stage.

The smaller update improved the graded score by 4.1 percentage points on new
attempts at the nine training problems, measured over seventeen pairs with valid
answers from both models. It did not improve complete-set accuracy. On the separate stages, the graded
score declined slightly and exact selection stayed at 0/18. Renaming the candidate
IDs consistently throughout the original problems removed the local advantage
on the seventeen comparisons valid in every condition. Both the base and trained
model changed their selections strongly. This is sensitivity to names and their
tokens, not proof that the model memorized the examples.

The larger update did not simply improve learning further. Its graded training
score fell by 5.6 points on seventeen matched valid attempts. On the separate
stages, that score rose by 2.8 points, but the model found fewer good candidates:
recall fell from 79.1% to 67.6%, and complete-set accuracy remained 0/18. It made
shorter selections, removing eleven false positives but also seventeen true
positives in aggregate. Thus a higher reward can coexist with a worse result
for a user who needs the complete list.

These are observed tradeoffs, not evidence about the model's hidden reasoning.
The larger update reused the original fixed training actions and initialization;
it did not continue training the smaller adapter. Each comparison used fresh
before-training controls. The separate stages were untouched for the initial
evaluation, but were already exposed when the larger dose was chosen. We retain
invalid-answer counts and matched denominators rather than treating unavailable
data as failures or silently changing the population.

This is a new exploratory recipe, not a controlled comparison isolating only
reward type from the conversation experiment: the task, starting weights, and
credited action also change. Its immediate purpose is to find whether selection
is learnable before committing to a larger training campaign.

## Another task family exposed an interface problem

We tested financial-report questions with two answer formats: a direct number,
or a short calculation program interpreted by Python. Both received the same
source material. Initially all 16 programs were invalid. Two small synthetic
worked examples in the prompt made 7 executable, with 3 target matches versus 1
direct answer. This changed the prompt, not the model's weights.

On the next 16 distinct pages, with that prompt fixed, 8 programs were executable
and 2 matched the supplied target; direct answers matched 1. The sole additional
program success computed a requested residential share from the appropriate
values. Scores are still low. Some targets also disagree with the question or
source, so a target mismatch is not always a model mistake. Primary scores were
not repaired or filtered after seeing answers.

The useful lesson is that a model can fail to use an unfamiliar interface even
when the interface would help. Correct execution also does not guarantee correct
evidence selection: an earlier correct ratio resulted from double-counting both
numerator and denominator. We need to inspect the computation, not just its answer.

## What we learned from other harness comparisons

Splitting inputs across more helpers did not reliably improve local selections.
Four helpers selected fewer ineligible records but also missed more eligible ones,
and used 42% more input tokens. Simply increasing the helper count is not our next
step.

This motivated the organized-record experiment above. A small alternative-model
check also tested a cached, post-trained 8B model on four complete problems,
directly and with perfect local selection reports supplied. It solved none of
the eight attempts. Fixing only its reported bookkeeping values, without changing
its chosen implementations, did not recover an optimal answer either. This is
a diagnostic involving a different model and template, not a controlled claim
about model size. Complete-plan selection remains an additional difficulty.

## The next informative experiments

1. **Require a decision for every candidate.** Compare the current selected-ID
   list with one yes/no decision per candidate, using the same organized records.
   This can separate missed decisions from failures to apply the policy, and
   reduce dependence on reproducing long candidate names. Do not change training
   simultaneously with this first interface comparison.
2. **Train on varied decisions, not just a larger update.** Once the interface
   works, sample varied policies, updates and candidate orders. Give credit to
   the decision being made, and track both missed good candidates and included
   bad candidates alongside complete answers. Hold out new cases and structural
   variations before choosing a training recipe.
3. **Learn when to change the view or split the work.** Compare keeping a problem
   together, organizing its records, and delegating smaller groups. Evaluate
   accuracy together with total calls and tokens. Only add another level of
   delegation when there is evidence it can address the remaining error.

These are ranked proposals, not completed experiments or evidence that a learned
chooser already exists. Repeated dose sweeps, more helper calls without a clear
mechanism, and further financial-prompt tuning are lower priority now.

## Publication prospects and limits

The most useful developing story is separating three bottlenecks: selecting the
right information, mechanically combining records, and faithfully returning the
answer. A reward can improve the last one while leaving the first two unresolved.
The fresh organized-record result makes this a more concrete direction: what
information should a model receive, which mechanical steps should the harness
perform, and which decisions should receive learning credit? The experiments
are beginning to distinguish these failures, but we do not yet have a general
RL or recursive-reasoning result suitable for a strong paper claim. Establishing
novelty will require comparison with existing preprocessing and tool-use methods,
broader tests, and a measured contribution beyond this task-specific transform.

Evidence is retained in the external research store under
`/project/alex_phd/runs/rlm-research-r4/`. Key resume pointers are:

- Fresh conversation audit: `analyses/openai-mrcr-fourneedle-balanced32-dose-transfer-2026-09-12/outcome-002/RESULTS.{json,md}`.
- Selection training: `sidecars/b05-flat-selection-rl-v1/outputs/attempt-001/`, including the optimizer, RNG state, fixed checkpoint, and training receipts.
- Selection audits: `analyses/b05-flat-selection-rl-independent-2026-09-13/` and `analyses/b05-flat-selection-rl-dose10-independent-2026-09-13/`.
- Name sensitivity: `analyses/b05-selection-id-renaming-audit-2026-09-13/`.
- Initial organized-record audit: `analyses/b05-public-normalization-independent-2026-09-13/outcome-001/`.
- Fresh organized-record audit: `analyses/b05-public-normalization-fresh12-independent-2026-09-13/outcome-001/`.
- Fresh financial audit: `analyses/finqa-two-example-fresh16-independent-2026-09-13/outcome-002/`.
- Next harness question: `ideas/2026-09-13-normalize-public-state-before-delegation.md`.
- Ranked follow-up question: `questions/public-state-and-decision-accounting.md`.

These research experiments use the pinned Prime/nano-RLM research harness.
They should not be confused with validation of every interface in the separate
Responses-only runtime in this repository. GitHub checkpoints preserve compact
source and reports, not a backup of external model weights or raw traces.
At the evidence cutoff, all accepted jobs and their independent saved-output
audits had completed. The GPU was released and idle, with no training running
in the background. This is a dated checkpoint, not a live utilization display.
At 00:58 UTC the shared Codex account had 19% remaining. Optional new branches
are winding down to preserve the requested reserve, rather than launching filler
or claiming continued utilization. Runtime-start failures and preparation gaps
are retained as operational costs, not scientific measurements.
