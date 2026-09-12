---
schema: research-checkpoint-v1
updated_utc: 2026-09-12T21:49:00Z
status: exploratory
questions:
  - Can reward-based training improve a controller that already retrieves the right text?
  - Does an extra helper improve reasoning, or merely change how the answer is written?
claim_level: small_unreplicated_rl_signal_and_mechanism_controls
---

# A small RL signal, and a reason to test simpler helpers

We have a small positive result from reward-based training, but not yet a
reliable general improvement. We also found that an apparently helpful extra
reasoning step improved answer wording rather than solving new reasoning steps.
Both findings suggest concrete next comparisons.

## Reward-based training improved two short answers, but not longer ones

We started from the supervised-trained model that already knows how to search
the conversation through Python. We then made one reward-based weight update,
using 32 previously saved attempts at eight training questions. Correct exact
answers received a positive training signal; incorrect answers received a
negative one. Only the final answer tokens contributed directly to the loss.

This differs from our previous group-relative objective. That objective compares
attempts at the same question, and our familiar questions produced identical
rewards on every attempt. It therefore had no learning signal. The new objective
instead compares each binary reward with a fixed baseline of one half. This is
a standard policy-gradient idea, not a new RL method.

| Same evaluation conditions | Before the reward update | After the reward update |
|---|---:|---:|
| Exact short answers | 23/32 | 25/32 |
| Exact longer-input answers | 10/16 | 10/16 |
| Available outcomes | 48/48 | 48/48 |

The short evaluation contains 16 conversations with two requested decoding
seeds each. The longer evaluation contains 16 different conversations with one
seed each. The updated model gained two short answers and lost none. Those gains
occurred on different conversations. One restored two missing trailing spaces;
the other replaced substantial incorrect answer content. The parsed Python
programs and tool observations were unchanged across all 48 comparisons.

The trained weights are shared across the whole controller. Restricting the
loss to final answers does **not** guarantee that only final-answer behavior
can change. Nor do equal observed programs prove identical internal reasoning.

The update itself took about 44 seconds; its two evaluation runs took about
405 and 248 seconds including model startup and cleanup. Training changed the
weights: we saved and checked the initial weights, gradient, updated adapter,
optimizer state and random state. On the saved training answers, the negative
gradient was overwhelmingly concentrated on erroneous whitespace, not on the
answer body. This explains the local training signal; it does not by itself
explain every changed evaluation answer.

These are research-exposed evaluation panels, one training update and reused
qualified control outputs. We have not established statistical reliability,
new-task transfer, or that the gains persist under new decoding seeds. A new
paired evaluation will run **both** models on all 16 short conversations with
two additional seeds each. It will retain every outcome, not just the two wins.

The exact primary score retains spaces and line breaks. A correct answer with
missing required spaces is wrong under this copying task's contract. This is
important for diagnosing training, but it is not the same as improving semantic
reasoning. We keep those interpretations separate.

Evidence in the external research store:

- `analyses/openai-mrcr-fixed-baseline-rl-paired-2026-09-12/readout-002.json`
  independently reproduces native outputs, scores, pairing and costs.
- `analyses/openai-mrcr-cp32-fixed-baseline-rl-update-audit-2026-09-12/RESULTS.json`
  records the actual update and gradient decomposition.
- `sidecars/openai-mrcr-cp32-fixed-baseline-final-rl-v1/outputs/attempt-001/checkpoint-0001/`
  contains the resumable trained state. Weights are not included in GitHub.

## More varied questions may be more useful than higher temperature

The competent model produced 28 exact answers out of 32 familiar training
attempts, but no question mixed correct and incorrect answers across its four
attempts. Higher temperature produced 27/32 and only one mixed group. That
group included a failure to execute the routine, while the repeated extra-line-
break error stayed wrong in all four attempts.

An earlier supervised checkpoint produced more mixed outcomes, but accuracy
fell to 6 of 31 available answers, with one additional outcome unavailable.
Losing the useful routine is not the clean way to teach accurate delivery.

The next screen used eight previously unused conversations selected in a fixed,
outcome-blind order. It produced7/32 exact answers and three mixed groups. The
model retrieved the correct text in24/32 attempts. Their remaining17 failures
were copying errors: nine added line breaks and eight omitted required spaces.
All three mixed groups had correct retrieval, so they provide useful contrast
for training final delivery. They do not provide a selector-training signal.

The other eight attempts failed on two requests because the model guessed the
source wording. For example, the source says “write a email about style,” but
the generated code searched for “write an email about style.” The requested
reply was present; changing the grammar caused literal matching to fail.
Another request was rewritten as a message or story. Email had no supervised
examples, but neither did song, which was retrieved successfully. New genre
alone therefore does not explain these cases.

This is not a paired28→7 model regression: the questions and decoding seeds
changed. It distinguishes copying contrast from failed literal selection.
We are preparing a generic instruction to inspect actual request strings
before matching them, with no email-specific patch or supplied answer. The
third- and fourth-request transfer evaluation is now running separately.

Evidence: `analyses/openai-mrcr-sft32-fresh8-g4-mechanism-2026-09-12/REPORT.json`.

## The extra helper changed wording, not the facts it could answer

On 12 exploratory multi-document questions, the final model received four
selected original paragraphs. We added a helper that extracted short relations
with exact supporting quotes, then gave its report to the final model alongside
the same four paragraphs. No new source material was added.

The exact-answer score rose from 3/12 to 5/12, with two gains and no losses.
However, inspecting every example showed that **both gains already contained
the correct answer content before the helper was added**:

- “48.4” became “48.4 square miles.”
- “Fewer than 20 tornadoes per year” became “fewer than 20.”

These changes match the benchmark's expected answer phrases. They do not show
that the helper recovered a missing fact or solved a previously unsuccessful
chain of reasoning. The helper increased the complete policy from 48 to 60
model calls; total completion tokens rose from 1,078 to 5,597.

Four helper reports failed exact quote checks. Even a valid quote was not
enough to establish a true relation: one report attached a father's death date
to his daughter. Quoting text faithfully and assigning it to the correct person
are different requirements.

We tested a cheaper control: keep the same four paragraphs, model, answer
schema and requested seeds, but directly ask for a concise answer phrase with
needed measurement units. This comparison asks whether a clearer answer
instruction can account for the apparent helper benefit without another call.
It was deliberately proposed after seeing these 12 cases, so any success will
need testing on new cases before we claim it generalizes.

The control also scored5/12, with two gains and no losses, but it corrected
**different questions**: one answer became the requested year2011, and another
added the square-mile units. Only the units gain overlapped the extra helper's
gains. Equal totals do not mean equivalent behavior or equally faithful answers.
Both interventions reduced exact supporting-citation accuracy from2/12 to1/12.
The cheaper instruction used880 completion tokens for the complete policy,
versus5,597 with the helper; it did not fix the wrong-person or missing-source
problems. There is no demonstrated advantage from adding generic extraction.

Control evidence:
`analyses/musique-direct-answer-contract-independent-2026-09-12/REPORT.json`.

Evidence: `analyses/musique-source-quoted-relations-independent-2026-09-12/outcome/REPORT.json`
and `MECHANISM_REVIEW.md`. Independent decoding found no scoring or integrity
discrepancies. The review preserves all 12 cases, including unresolved source
selection, entity attribution, citation and dataset-label issues.

## What now looks promising

The most useful direction is to identify **which stage prevents success**:
finding the evidence, using it correctly, or returning the answer in the
required form. Improvements in one stage need not improve the whole system.
Our next experiments separate those stages instead of treating every low exact
score as a failure of decomposition.

A publishable result would need a repeatable intervention that addresses a
specific failure and works on new material, with its cost and limits measured.
We do not yet have evidence for learned recursive decomposition or a broadly
successful RL method. The stronger established result remains supervised
retrieval transfer to new, longer conversations; the reward update is now a
small, testable extension to that baseline.

The already-presented advisor deck remains a historical snapshot. This report
is a post-meeting research checkpoint, not a silent revision of that deck.
