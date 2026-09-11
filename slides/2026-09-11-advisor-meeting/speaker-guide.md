---
title: Speaker guide for the advisor meeting
meeting_date: 2026-09-11
deck_structure: 9 main pages plus 6 optional backup pages
suggested_talk_time: 10 minutes
status: current
---

# Speaker guide

This is a ten-minute discussion, not a tour through every experiment. Speak through the nine
main pages and stop. Pages 10–15 are optional answers to questions.

The new story, compared with the prior eight-page deck, is more focused: training transferred to
newly selected record sets; the helper-interface result now appears in two model families; and the
next experiment is a concrete change to the RLM handoff, rather than another broad training run.
These are still exploratory studies on familiar task types.

## Suggested timing

| Time | Main page | Job |
|---|---:|---|
| 0:00–0:40 | 1 | State the research question. |
| 0:40–1:40 | 2 | Explain the RLM through a concrete task. |
| 1:40–2:40 | 3 | Show what SFT teaches the model to do. |
| 2:40–3:40 | 4 | Explain the training results. |
| 3:40–4:30 | 5 | Introduce matching inputs to answers. |
| 4:30–5:30 | 6 | Show the helper-level result. |
| 5:30–6:30 | 7 | Compare accuracy and total time. |
| 6:30–7:30 | 8 | Explain the proposed RLM change and its test. |
| 7:30–10:00 | 9 | Summarize and invite discussion. |

## Main page 1 — Can better handoffs make an RLM more reliable?

Say: “A handoff is the exchange of work and answers between the main model and a helper model.
Last time, examples taught a small model a basic Python routine. Today we have a better routine
using helpers and a separate finding: the way a helper answer is linked to its input matters.”

Understand: the training and handoff studies test different components. They are not points on one
learning curve, and neither establishes a general-purpose recursive solver.

Likely Q&A — **What is genuinely new since the earlier deck?** The training routine was retested
on newly selected record sets, the batching effect was observed in Qwen and Mistral, and the work
now motivates a falsifiable program-level handoff experiment. Recursive calls and identifiers
themselves are not new inventions.

## Main page 2 — An RLM divides a large task into smaller steps.

Say: “The task is to count each user's questions that ask for a location. Ada's Oslo question
qualifies; her Hamlet question asks for a person, so it does not. Ben's Kyoto question qualifies.
The main model uses Python to divide the file. Helpers decide which questions qualify, and
Python counts their replies: one for Ada and one for Ben.”

**What does ‘asks for a location’ mean?** It means that the expected answer is a location.
‘Where is Oslo?’ is a location question. ‘Who wrote Hamlet?’ is a person question. The helper
classifies the question; it is not being asked to answer Oslo or Hamlet itself. A question can
mention a place without asking for a location: ‘How many people live in Oslo?’ asks for a number.

**What is a helper?** It is another request to a language model, not necessarily a different
model or another GPU. It receives a smaller question and the relevant text. Python is the
programming environment the main model uses to inspect data and calculate. The three rows are
an invented miniature file so the entire example can be checked on screen.

Likely Q&A — **Why is this interesting beyond handling long files?** The main model need not read
the whole input in its prompt. It can inspect the workspace and see only selected pieces or helper
results. More broadly, a difficult unfamiliar problem might become a sequence of smaller decisions
and calculations the model already knows how to handle. That is the larger research motivation;
our experiments test specific parts of it. See [Alex Zhang's explanation of the surrounding program](https://alexzhang13.github.io/blog/2026/harness/)
and the [RLM paper](https://arxiv.org/abs/2512.24601).

For a concrete illustration of that hope, compare counting questions asking for a location with
counting faulty-machine reports for each site. The texts and categories differ, but both can use
the same plan: ask a focused reading question about each record, group the replies, and count.
Helpers handle the changing subject matter. The main model may be able to reuse the plan.
This illustrates a possible kind of transfer; our experiments have not established it across
those two domains.

Backup page 1 extends this to adding assigned points and counting users above a threshold.
Keep that separate from the simpler main-slide example: there the final answer is a count for
each user, not a weighted total. Assigned points are artificial test values, not confidence.

Likely Q&A — **Why call this recursive?** A helper could use the same machinery again, but these
experiments mostly use one helper layer. Do not claim deep autonomous planning.

## Main page 3 — We used supervised fine-tuning (SFT) to teach the main model a routine.

Say: “SFT means learning to imitate demonstrated actions. Read this example from left to right.
The task is to count location questions. The helper has already supplied three question types.
We train the main model to write code that counts the two location replies. The target is that
useful action, not just the final number.”

The code starts a count at zero, visits each reply, adds one for each location, and prints two.
This is a simplified illustration of one step, not a literal training record. Full examples also
show asking the helper and returning the calculated answer. See the
[worked training example](sft-worked-example.md) for the actual recorded action behind the illustration.

**What changes during SFT?** The main model's learned parameters change. This is not simply putting
examples in its prompt at test time. Two copies began with the same earlier-trained model; each
learned from 72 complete worked interactions using a different example set. The helper stayed
unchanged. These 72 training examples are separate from the 72 later test tasks.

Likely Q&A — **What were the training examples?** They were worked interactions, not just lists
of correct final numbers. The main model saw examples of inspecting records, asking a helper for
question types, and running Python on the replies. The underlying texts were public TREC questions,
with artificial users, record names, and numeric weights. The real task distinguishes six broad
answer types; the location/not-location example on page 2 is deliberately simpler. Training
changed a small set of added model parameters (an adapter), not the helper model.

## Main page 4 — Both separately trained copies solved more of the 72 test tasks.

Say: “We tested all three versions on the same tasks. We confirmed 12 successes before this SFT,
55 for one trained copy, and 53 for the other. Success means getting the right answer and doing
the requested calculation using the helper's replies.”

**What happened to the attempts without a verifiable final answer?** Six starting-model attempts
and two second-copy attempts timed out after approximately three minutes. Their folders contain
error reports and intermediate activity, but no completed final-result record. One additional
starting-model attempt has a result record that ends with a tool call rather than a verifiable
final answer. This is not evidence that completed answers were lost.

All 72 attempts remain in each comparison. A timeout is an unsuccessful completion within the
allowed time, even though we cannot judge a final answer that was never received. The original
analysis used “unknown” for unavailable answer correctness; that bookkeeping term obscured the
practical meaning and has been removed from the main slide.

| Version | Confirmed successes | No verifiable final answer | Original missing-answer bounds |
|---|---:|---:|---:|
| Before this SFT | 12 | 7 | 12–19 |
| Copy trained with the first example set | 55 | 0 | 55 |
| Copy trained with a different example set | 53 | 2 | 53–55 |

The original analysis also asked whether assigning success to every unavailable answer could
erase the comparison. Even that generous assumption gives the starting model only 19 successes,
still below either trained copy. These are bookkeeping bounds, not statistical confidence
intervals, actual additional successes, or predictions of what more time would achieve.
The copies were trained separately: the 53 bar is not a later stage of the 55 bar. Their similar
results support the usefulness of the routine, not a claim that one example set is better.

Likely Q&A — **Does this show generalization?** It transfers to newly selected record sets, but the
question types are familiar. It does not demonstrate arbitrary new tasks or autonomous planning.

## Main page 5 — We tested whether names help link each helper answer to the right statement.

Say: “Read across each row. Each input item contains a text and a claim about that text.
A bike is a vehicle, so the first claim is supported. Blue contradicts red. The third row uses
a different text: a train arrived at noon. We cannot tell whether it was late without knowing
its schedule. One helper request contains several such pairs, and asks for one judgment per pair.”

**Are these the same questions applied to many texts?** No. The instruction is always to judge
a claim against its paired text, but both the text and the claim can vary. Some pairs share a
text, as the first two rows do. A claim is a statement to check: “The bike is blue” means
“Does this text support the claim that the bike is blue?” It is not a question about every text.

**What changes between the two formats?** Without added matching names, software uses answer
order: the first judgment belongs to the first pair. With matching names, the reply repeats
the name assigned to its pair, such as `k7ab: Supported`. The same pairs are tested in both
formats. The table shows names for illustration, not because both formats repeat them in replies.

**What does k7ab mean?** Nothing about the content. It is an arbitrary name, like a coat-check
ticket number. Repeating it on the reply identifies which statement the reply describes.
It does not tell the helper whether that statement is supported or contradicted. The three correct
judgments on this slide explain the task; they are not given to the model. The next slide tests whether
the named format helps when the model must judge many statements in one request.

**Is this the same task as the training study?** No. The matching experiments use MultiNLI
text–statement pairs, not the TREC counting tasks. The real task allows three judgments:
supported, contradicted, or not enough information. The table now illustrates all three.
Explanations on the slide teach what the judgments mean; they are not literal model outputs.

Understand: software supplies the identifiers and checks the format. The model still chooses the
labels. An identifier is not a hint and cannot guarantee that the model read the right statement.

Likely Q&A — **Could names leak the answer?** They are arbitrary, and a separate control compares
matching with nonmatching arbitrary names. That supports a matching effect, not a claim about the
model's internal mechanism.

## Main page 6 — Matching names helped helpers judge many statements in one request.

Say: “Qwen and Mistral are two language models. Moving right on the plot means each request
contains more statements to judge. At 64, adding matching names raised Qwen accuracy from
**44% to 83%** and Mistral accuracy from **35% to 52%**.”

Keep this page to the advertised comparison: **without versus with matching names**. The same 16 record
sets were used within each model. Three malformed Mistral tagged responses count as wrong, not
missing. These are helper reading answers, not complete RLM solutions. The full comparison that
also includes row numbers is optional backup page 6.

Likely Q&A — **Why does Mistral remain much worse?** Matching helps both models, but it is not
sufficient for correct reading. This cross-model direction is behavioral evidence, not a clean
capacity comparison or a mechanism result.

## Main page 7 — Smaller requests were faster than large requests with matching names in this local test.

Say: “All four methods judged the same 768 statements. With 48 statements per unnamed request,
accuracy was 49% and the whole workload took 19 seconds. Adding matching names raised accuracy to 85%, but that workload
took 91 seconds. Splitting into groups of 16 reached 81% in 19 seconds; one statement at a time reached
87% in 31 seconds.”

A request is one message to the helper asking it to judge one or more statements. ‘Total time’
covers all 768 statements for that method, not a single request. A statement and its accompanying
text are one input item. Repeated input is a cost because the model processes more text.

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

## Main page 8 — Next test: let the RLM program organize helper requests and match their answers.

Say: “The surrounding program would keep each record linked to its returned answer and make group
size an explicit choice. The model still chooses what to ask. We would compare complete solutions
against both the current RLM and simpler small-call baselines under the same resource budget.
First we would compare fixed group sizes; learning when to change them comes later.”

In the picture, an input item means one question or statement and its relevant text. Group size
means how many such items we send in one helper request. ‘Match’ means connect k7ab's reply to
k7ab's original item, not merely take the next answer in the list. The program handles this
bookkeeping; the language model still decides what to ask and what the text means.

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

## Main page 9 — We have a promising handoff result and a focused next research question.

Say: “Worked examples improved one useful routine. Explicit matching improved helper reading. The
unresolved question is whether the handoff change improves a complete RLM solution at useful cost.”

Ask: “Is that a worthwhile contribution, and what realistic task would make the test convincing?”

Likely Q&A — **What can we claim now?** A reproducible helper-interface effect and a transferred
training routine, with clear failure cases. We cannot yet claim a whole-system gain, a novel
identifier method, guaranteed truthful reading, or general recursive planning.

**Stop the prepared talk here.** Open a backup page only in response to a question.

# Optional backup pages

## Backup page 1 — Example: the helper identifies question types, and Python adds the relevant points.

Use for requests for a worked example. Walk through Ada's 4 + 3 = 7, Ben's 2, and the answer of 1.
Emphasize that a correct Python calculation can still be wrong if the helper supplies a wrong label.
‘Points’ are the assigned numeric weights in the experiments. Ben's six points do not count because
the Hamlet question asks for a person, not a location. Asking for Peru's capital asks for a place
name, so Ada's three points do count. No geographical answers need to be produced.

## Backup page 2 — Using the same name on a statement and its answer helped more than using different names.

Use when asked whether any extra text would help. Both conditions had arbitrary names; only one
reused the same names on inputs and outputs. Later-answer accuracy rose from 32.6% to 78.8% on a
different panel. This supports matching as a factor, not a causal account of internal attention.
In both versions, answer position still determined the intended statement. The different-name
version did not tell the helper to switch to another statement. Only the last 32 judgments from
each 48-statement request are included in this plot; it is not the all-position score on page 6.

## Backup page 3 — Reward training did not improve the final-answer count in this trial.

Use when asked about reinforcement learning. On this **secondary final-answer metric**, the shared
baseline was **57/72**, answer-reward training produced **55–57/72**, and answer plus a calculation
check produced **54/72**. The two runs attempted 576 solutions and made 19 and 21 optimizer updates.
This metric differs from the faithful-calculation success measure on main page 4; the predeclared
calculation review is not complete. Treat this as a bounded negative trial, not proof that reward
learning cannot work.

In plain language, worked-example training says ‘copy these useful actions.’ Reward training says
‘try a solution; use its score to adjust the model.’ The added agreement check compared the final
number with a Python calculation from helper replies. This is not a guarantee of truth: a helper
can misclassify text. The 576 attempts are training practice; the 72 questions are the later test.
The 55–57 range reflects two missing test results, not uncertainty across many repeated runs.

## Backup page 4 — Training also helped with new combinations of familiar steps.

Use when asked whether the trained model only learned familiar question types. On 72 questions
combining familiar operations in new ways, verified correct answers and calculations increased
from 2 to 29. Eleven earlier outcomes and one later outcome could not be verified. Even if all
eleven unknown earlier outcomes succeeded, the earlier total would be 13, still below 29.
These questions reused eight previously
tested sets of records, so the new part is the combination of operations, not the input text.

Example: first find users who asked for a location, then count those users' questions asking
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
The example instruction means ‘answer row 7, but label the reply row8.’ Row 8 is also in the input,
so the label competes with the instruction about which row to answer. The unrelated-name comparison
has no such second input to point toward. These are invented examples of names, not quoted outputs.

## Backup page 6 — Both row numbers and arbitrary names helped when helpers judged many statements together.

Use when asked for the full format comparison behind main page 6. This is the only page where the
row-number condition should be discussed in the prepared deck. Both row numbers and arbitrary
matching names helped. All 480 calls returned; three malformed Mistral arbitrary-name outputs count
as wrong and complicate a direct comparison between the two named formats.
Row numbers look like 1 and 2; arbitrary names look like k7ab and z2pm. Both repeat beside the
statement and its answer. The horizontal axis is the number of statements judged in one request,
and the vertical axis is the share judged correctly. Qwen and Mistral are the two models, not
two training stages.

# Supporting material

- [Evidence and methods](evidence-and-methods.md) gives the claim ledger, study definitions, and
  provenance behind the displayed results.
- [Detailed findings guide](detailed-findings-guide.md) preserves the longer background and caveats.
- [Speaker notes](speaker-notes.json) contains private cues in a separate pdfpc sidecar,
  not embedded in the audience PDF.
- [Deck source](research-update.tex) is the authoritative wording and current page order.
