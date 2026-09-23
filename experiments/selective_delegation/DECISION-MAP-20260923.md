---
date: 2026-09-23
status: prospective-exploratory-decisions
written_before: complete larger-update evaluation and all memory results
---

# What would change our next research decision?

Our main question is becoming more specific: **what must an agent discover,
remember and receive credit for in order to solve a multi-step task?** This can
lead back to recursive language models, but the current TextCraft actors are
flat. Better flat-agent results are not evidence that recursion helps.

## 1. Larger RL updates: is step size the main obstacle?

Training has verified a five-times-larger parameter change on the same saved
attempts. Wait for the complete task evaluation, including missing outcomes and
cost. A clear improvement would justify a fresh-batch or changed-world check;
it would not establish a general learning curve from one reused batch. A decline
would argue against simply increasing the learning rate again. A small or mixed
change would leave credit assignment and task difficulty as live explanations.

The prepared positive-only comparison tests whether including negative
whole-attempt credit helps at this learning rate. It changes the gradient and
token dose, so it cannot isolate the effect of one useful action. If accepted,
place it before lower-priority extra-world replications, without interrupting
the already accepted memory comparison or altering its four arms.

## 2. Memory: preserve discovered facts, or remove distracting history?

The four conditions independently vary a notebook of observed recipes and
whether the agent sees all prior interactions or only the latest four.

| Complete comparison | What it would suggest—not prove |
|---|---|
| A notebook helps with full history. | Organizing already visible facts may help, even without forgetting old interactions. |
| A notebook helps mainly with short history. | Retaining recipe information may compensate for removing old observations. |
| Short history helps without a notebook. | Old interactions may distract the model, or shorter inputs may avoid context limits. |
| Neither change helps reliably. | Generic memory changes may not address the dominant errors in these tasks. |

Inspect native errors, context limits, calls, input tokens and elapsed time with
success—not just the best arm. These are alternative explanations, not automatic
mechanistic conclusions. Finish the entire fixed comparison before selecting a
winner; then check another trained actor and a changed recipe world if warranted.

If retaining facts helps, a concrete RLM follow-up is to vary what a helper
returns: an answer alone versus an answer and the public facts it discovered.
Hold the model, tasks and total call budget fixed. First test one helper boundary;
deeper recursion should earn its extra complexity through evidence.

## 3. Teaching: does the advantage survive the known confound?

The public-discovery teacher currently wins across two training seeds and a
changed recipe world. A quantity error in the original teacher still prevents
attributing all of that advantage to teachable information gathering. Run the
prepared quantity-corrected control before strengthening that claim. If the
gap shrinks substantially, revise the story rather than discount the control.
If it persists, separate the effects of teacher actions, observations and memory.

## Publication discipline

LEAP already studies whether a student can imitate privileged experts; generic
memory and step-credit methods also have substantial prior work. A defensible
contribution needs a sharper result: for example, which facts must cross a
decomposition boundary, under what budgets, and whether that result transfers.
Same-world goal panels, recipe-world changes and new task families are different
levels of evidence. Keep them separate.

These are adaptive exploratory rules, not preregistered hypotheses or promises
that every branch will run. Local bounded jobs continue without Codex tokens;
scientific reprioritization requires reviewing their completed reports.

## Later evidence, 09:30 UTC: distinguish planning from tool-argument copying

The completed RL evaluation now motivates another bounded harness question.
All 578 rejected crafting actions under the smaller update already had their
target recipe in prior public history;615 of 689 under the larger update did too.
See the [credit analysis](RL-CREDIT-ASSIGNMENT-20260923.md). This addition is
post-result planning; the earlier sections above were written prospectively.

Candidate: compare model-written ingredient arguments with a public-recipe
executor that translates the model's chosen observed recipe and quantity into
exact tool arguments. First estimate how many recorded errors are ingredient
translation errors versus genuine inventory shortages or incompatible quantities.
If most would survive that translation, do not spend a full GPU run on it yet.
If many are addressable, prepare a matched pilot after the current diagnostics.
No hidden-recipe access, automatic goal planning, or silent retry is permitted.
This is a mechanism control and possible RLM-interface improvement, not a novelty
claim for the general idea of structured tool calls.

At 09:35, native one-step checks support 252 potential argument repairs under the
smaller update but only 64 under the larger update. Inventory shortages survive
the repair at 310 and 546 positions respectively. Repeated errors are not independent
cases, and repaired steps are not repaired tasks. Therefore target an eventual
pilot at the smaller-update actor first, with the same public recipe information
available in both arms; do not assume an equally useful effect on the larger one.
Keep this as a CPU-prepared research question until the current memory/credit
comparisons establish the most informative use of remaining GPU time.

If a public-recipe interface helps, the next RL question is whether learning gets
easier when the model is trained to choose an information request, recipe and
quantity, rather than reproduce all low-level arguments. This directly probes
whether we are training the wrong part of the behavior. It needs matched starting
models, demonstrations and compute accounting; a simpler interface alone is not
a novel RL algorithm. Use an explicit tool contract, not silent correction of
the existing action stream.

For decomposition, inventory shortages motivate a separate future question:
should a parent reserve shared resources before delegating subgoals? Two locally
reasonable helper plans may compete for the same ingredients. Test that only
after controlling basic recipe execution, and compare under the same total budget.
This is an unimplemented hypothesis, not a finding from the present flat actors.

### Prior-art check, September 23

[TAPE](https://arxiv.org/html/2602.19633v1) already separates planning mistakes
from deviations during execution. It constructs a graph from proposed plans,
uses a solver to choose a path, constrains action generation, and replans when
observations disagree. Our inference: the general planning/execution separation
is not a novelty claim. A narrower candidate is how an interface restricted to
discovered facts changes RL learning and the benefit of decomposition. Our error
categories also do not identify TAPE's internal planning-versus-sampling causes.

[RunAgent](https://arxiv.org/abs/2605.00798) interprets natural-language plans
with control constructs, step constraints, and choices among reasoning, tools
and code. Its abstract also describes selective history retention. This reinforces
the need for matched controls around the notebook and execution interface;
these general architectural ingredients already exist. Abstract-level screening,
not a replication or a detailed comparison of implementations.

No paper performance numbers are imported into our results. A short search is
not an exhaustive novelty assessment. No new repository was executed or dataset
added during this check; the GPU experiment continued independently.
