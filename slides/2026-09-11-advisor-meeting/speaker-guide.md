---
title: Speaker guide for the advisor meeting
meeting_date: 2026-09-11
deck_structure: 8 main pages plus 6 optional backup pages
suggested_talk_time: 10 minutes
status: current
---

# Speaker guide

This is a ten-minute discussion, not a tour through every experiment. Speak through the eight
main pages and stop. Pages 9–14 are optional answers to questions.

The new story, compared with the prior eight-page deck, is more focused: training transferred to
newly selected record sets; the helper-interface result now appears in two model families; and the
next experiment is a concrete change to the RLM handoff, rather than another broad training run.
These are still exploratory studies on familiar task types.

## Suggested timing

| Time | Main page | Job |
|---|---:|---|
| 0:00–0:45 | 1 | State the question and what changed since the last discussion. |
| 0:45–1:45 | 2 | Explain the RLM in one concrete workflow. |
| 1:45–3:00 | 3 | Give the training result and its limits. |
| 3:00–4:00 | 4 | Introduce explicit input–answer matching. |
| 4:00–5:15 | 5 | Show the central helper-level result. |
| 5:15–6:30 | 6 | Explain the measured quality, time, and token tradeoff. |
| 6:30–7:45 | 7 | Explain the proposed RLM change and its decisive test. |
| 7:45–10:00 | 8 | Summarize, ask for advice, and discuss. |

## Main page 1 — Can better handoffs make an RLM more reliable?

Say: “Last time, we showed that examples could teach a small model a basic Python routine. The
open problem was reliable use of helper calls. Today I have a harder transfer result and evidence
that the way helper answers return to their source records matters.”

Understand: the training and handoff studies test different components. They are not points on one
learning curve, and neither establishes a general-purpose recursive solver.

Likely Q&A — **What is genuinely new since the earlier deck?** The training routine was retested
on newly selected record sets, the batching effect was observed in Qwen and Mistral, and the work
now motivates a falsifiable program-level handoff experiment. Recursive calls and identifiers
themselves are not new inventions.

## Main page 2 — A long file can become many small questions whose answers Python can combine.

Say: “A main model can inspect a long file, ask helpers smaller reading questions, and use Python
to combine their replies. The hope is that an unfamiliar whole becomes a set of familiar steps.”

Likely Q&A — **Why is this interesting beyond handling long files?** The main model need not read
the whole input in its prompt. It can inspect the workspace and see only selected pieces or helper
results. More broadly, a difficult unfamiliar problem might become a sequence of smaller decisions
and calculations the model already knows how to handle. That is the larger research motivation;
our experiments test specific parts of it. See [Alex Zhang's explanation of the surrounding program](https://alexzhang13.github.io/blog/2026/harness/)
and the [RLM paper](https://arxiv.org/abs/2512.24601).

For a concrete illustration of that hope, compare counting place questions for each user with
counting faulty-machine reports for each site. The texts and categories differ, but both can use
the same plan: ask a focused reading question about each record, group the replies, and count.
Helpers handle the changing subject matter. The main model may be able to reuse the plan.
This illustrates a possible kind of transfer; our experiments have not established it across
those two domains.

Use the original Ada example if the audience needs something concrete: Ada has two place questions
with weights 4 and 3, so her total is 7. Ben's only place question has weight 2. Only Ada exceeds
5, so the final answer is 1 user. The helper decides which questions concern places; Python does
the addition and thresholding. The displayed weights are artificial test values, not confidence.

Likely Q&A — **Why call this recursive?** A helper could use the same machinery again, but these
experiments mostly use one helper layer. Do not claim deep autonomous planning.

## Main page 3 — Training taught a more reliable routine for asking helpers and calculating answers.

Say: “We trained the main model on worked interactions: ask a helper, retain its actual replies,
then carry out the requested calculation in Python. On the same 72 questions, verified success was
12 before training and 55 or 53 after training. The input records were new to this test, but the
kinds of questions were familiar.”

State the denominator and missing-result bounds exactly: **12–19, 55, and 53–55 out of 72**. The
two trained models used separate training corpora; one was not trained on top of the other. The
helper was unchanged. Success requires both the right final answer and the requested calculation
using observed helper results.

Likely Q&A — **Does this show generalization?** It transfers to newly selected record sets, but the
question types are familiar. It does not demonstrate arbitrary new tasks or autonomous planning.

## Main page 4 — We changed how helper answers are linked to the text they describe.

Say: “If a helper judges two statements about Maya's bike, we can attach an arbitrary identifier to
each statement and require the same identifier on its answer. Without that link, software relies on
list position.”

Understand: software supplies the identifiers and checks the format. The model still chooses the
labels. An identifier is not a hint and cannot guarantee that the model read the right statement.

Likely Q&A — **Could names leak the answer?** They are arbitrary, and a separate control compares
matching with nonmatching arbitrary names. That supports a matching effect, not a claim about the
model's internal mechanism.

## Main page 5 — Matching names improved accuracy when a helper answered many questions at once.

Say: “At batch size 64, moving from no tags to arbitrary matching tags raised Qwen accuracy from
**44% to 83%** and Mistral accuracy from **35% to 52%**.”

Keep this page to the advertised comparison: **no tags versus arbitrary tags**. The same 16 record
sets were used within each model. Three malformed Mistral tagged responses count as wrong, not
missing. These are helper reading answers, not complete RLM solutions. The full comparison that
also includes row numbers is optional backup page 6.

Likely Q&A — **Why does Mistral remain much worse?** Matching helps both models, but it is not
sufficient for correct reading. This cross-model direction is behavioral evidence, not a clean
capacity comparison or a mechanism result.

## Main page 6 — Smaller calls were faster than large named calls in this local test.

Say: “All four methods answered the same 768 reading questions. With 48 questions per unnamed call,
accuracy was 49% and the whole workload took 19 seconds. Adding matching names raised accuracy to 85%, but that workload
took 91 seconds. Splitting into groups of 16 reached 81% in 19 seconds; one record at a time reached
87% in 31 seconds.”

Explain the tradeoff rather than naming a winner. Large named calls sent less repeated input text
than single-record calls, but generated much more structured output and were slower here. Groups of
16 were fastest in this setup but less accurate than the named or singleton conditions. Each
displayed time is the sum of four nonoverlapping workload blocks for that policy; up to four calls
ran simultaneously inside a block. These are not per-request latency comparisons.

Likely Q&A — **Which method is cheapest?** We did not measure one universal notion of cost. The
answer changes if the constraint is elapsed time, input tokens, output tokens, or accuracy. The
local result says that fewer calls did not automatically mean less time, and that smaller unnamed
calls are a serious baseline for a more elaborate interface.

Optional follow-up — **Can the named replies be shorter?** Yes. A paired compact-output check
reduced output tokens and local elapsed time in both models. Qwen retained about 85% accuracy;
Mistral had fewer malformed replies but still failed on some batches. This refines the proposed
interface, not the main claim about complete solutions. See E2 in [supporting findings](later-findings.md).

## Main page 7 — Proposed RLM change: manage record links and make helper group size an explicit choice.

Say: “The surrounding program would keep each record linked to its returned answer and make group
size an explicit choice. The model still chooses what to ask. We would compare complete solutions
against both the current RLM and simpler small-call baselines under the same resource budget.
First we would compare fixed group sizes; learning when to change them comes later.”

Be precise: the existing `ask_batch` facility aligns whole requests and responses, but not the
individual records inside one request. The proposed component would preserve record identifiers
through grouping, requesting, and joining, while flagging missing or duplicate answers. The decisive
outcome is a correct complete answer whose calculation used the observed evidence, together with
availability, elapsed time, and token cost. Learning when to resize groups is a later experiment.

Likely Q&A — **Do the identifiers work only when answers stay in input order?** In an exploratory
control on the same 16 exposed contexts, keyed accuracy was about 83% in input order, 82% in reverse
order, and 81% with interleaved halves. That supports the supplied key-addressed format when output
order changes; it is not a proof of arbitrary-order robustness, an internal binding mechanism, or a
whole-RLM gain.

Likely Q&A — **What result would convince us?** More reliable complete answers without an
availability loss, trace evidence that the requested calculation used correctly joined records,
and a useful quality/time/token tradeoff versus simply shrinking the helper groups. Identifiers are
standard, and even perfect matching cannot guarantee that a helper read the source correctly.

Late pilot — **Have we now shown better complete solutions?** No. The last short run did not
produce enough observed outcomes to compare the two handoffs. It did expose repeated mistakes in
how the main model called the changed interface. The next test needs an explicit working call
example as well as reliable record matching. Details are in [supporting findings](later-findings.md).
This does not change the main slide's status: a complete-system benefit remains unestablished.

## Main page 8 — We have a promising handoff result and a focused next research question.

Say: “Worked examples improved one useful routine. Explicit matching improved helper reading. The
unresolved question is whether the handoff change improves a complete RLM solution at useful cost.”

Ask: “Is that a worthwhile contribution, and what realistic task would make the test convincing?”

Likely Q&A — **What can we claim now?** A reproducible helper-interface effect and a transferred
training routine, with clear failure cases. We cannot yet claim a whole-system gain, a novel
identifier method, guaranteed truthful reading, or general recursive planning.

**Stop the prepared talk here.** Open a backup page only in response to a question.

# Optional backup pages

## Backup page 1 — Example: the helper reads the text, and Python does the counting.

Use for requests for a worked example. Walk through Ada's 4 + 3 = 7, Ben's 2, and the answer of 1.
Emphasize that a correct Python calculation can still be wrong if the helper supplies a wrong label.

## Backup page 2 — A control suggests that matching matters, not merely having names on the page.

Use when asked whether any extra text would help. Both conditions had arbitrary names; only one
reused the same names on inputs and outputs. Later-answer accuracy rose from 32.6% to 78.8% on a
different panel. This supports matching as a factor, not a causal account of internal attention.

## Backup page 3 — Reward training did not improve the final-answer count in this trial.

Use when asked about reinforcement learning. On this **secondary final-answer metric**, the shared
baseline was **57/72**, answer-reward training produced **55–57/72**, and answer plus a calculation
check produced **54/72**. The two runs attempted 576 solutions and made 19 and 21 optimizer updates.
This metric differs from the faithful-calculation success measure on main page 3; the predeclared
calculation review is not complete. Treat this as a bounded negative trial, not proof that reward
learning cannot work.

## Backup page 4 — Training also helped with new combinations of familiar steps.

Use when asked whether the trained model only learned familiar question types. On 72 questions
combining familiar operations in new ways, verified correct answers and calculations increased
from 2 to 29. Missing-result bounds are 2–13 and 29–30. These questions reused eight previously
tested sets of records, so the new part is the combination of operations, not the input text.

Example: first find users who asked a place question, then count those users' questions asking
for a number. These are two selections across a user's records; one record need not belong to
both categories. The instructions explicitly supplied those steps. This tests carrying out a
new combination, not inventing the plan. Forty-eight answers were zero, so merely obtaining the
correct final number could be misleading. We checked the recorded calculations as well.

On the 24 questions with nonzero correct answers, the trained model had 12 verified successes,
versus none observed before training (two earlier outcomes missing). This supports the direction
without relying solely on easy-to-guess zeros. The review was unblinded and agent-authored.
See S4 in the evidence document. The older 0/8-to-1/8 helper-training warning remains in C1 there.

## Backup page 5 — Misleading names can draw an answer toward the wrong input.

Use when asked for failure evidence. Deliberately misleading visible-record names yielded 39%
accuracy, versus 80% for unrelated names that pointed to no visible record. This is a stress test of
output behavior, not evidence that we measured attention or guaranteed which text the model read.

## Backup page 6 — Both row numbers and arbitrary names helped in the larger-batch tests.

Use when asked for the full format comparison behind main page 5. This is the only page where the
row-number condition should be discussed in the prepared deck. Both row numbers and arbitrary
matching names helped. All 480 calls returned; three malformed Mistral arbitrary-name outputs count
as wrong and complicate a direct comparison between the two named formats.

# Supporting material

- [Evidence and methods](evidence-and-methods.md) gives the claim ledger, study definitions, and
  provenance behind the displayed results.
- [Detailed findings guide](detailed-findings-guide.md) preserves the longer background and caveats.
- [Speaker notes](speaker-notes.json) contains private cues in a separate pdfpc sidecar,
  not embedded in the audience PDF.
- [Deck source](research-update.tex) is the authoritative wording and current page order.
