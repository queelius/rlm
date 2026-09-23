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
