---
schema: research-checkpoint-v1
updated_utc: 2026-09-12T22:55:00Z
status: exploratory
questions:
  - Can reward-based training improve a controller that already retrieves the right text?
  - Does an extra helper improve reasoning, or merely change how the answer is written?
claim_level: rl_gain_did_not_replicate_and_same_family_sft_transfer
---

# Retrieval transfers, but the small RL gain did not replicate

The supervised-trained search routine works on longer conversations and on
third/fourth occurrences that were not in its demonstrations. Reward-based
training is less convincing: an initial small gain reversed when we repeated
the comparison with new decoding seeds. An apparently helpful extra reasoning
step also improved answer wording rather than solving new reasoning steps.
These results help us choose what to try next; they are not evidence of a
generally successful RL method or learned recursive decomposition.

## The first RL gain reversed under new decoding seeds

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
| Short answers, fresh paired seed block | 25/32 | 22/32 |

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

We then ran **both** fixed models on all 16 short conversations with two new
decoding seeds each, retaining every outcome. All 32 pairs were available.
The starting model scored 25/32 and the updated model 22/32: three losses and
no gains. All three losses added a final line break after otherwise correct
retrieval. They occurred on two conversations; repeated seeds are not
independent conversations. The complete token paths changed in five attempts,
including two that remained wrong. Returned calls rose from 64 to 66 and
completion tokens from 20,739 to 23,170.

The two short blocks therefore give inconsistent signs, not a replicated
improvement. These are research-exposed panels and one trained checkpoint;
we will not select the favorable seed block and call RL successful. The
longer-input result remains unchanged. This does not show that RL cannot work;
it retires the positive claim for this particular one-step recipe.

The exact primary score retains spaces and line breaks. A correct answer with
missing required spaces is wrong under this copying task's contract. This is
important for diagnosing training, but it is not the same as improving semantic
reasoning. We keep those interpretations separate.

Evidence in the external research store:

- `analyses/openai-mrcr-fixed-baseline-rl-paired-2026-09-12/readout-002.json`
  independently reproduces native outputs, scores, pairing and costs.
- `analyses/openai-mrcr-fixed-rl-decode-replica-2026-09-12/RESULTS.json`
  reproduces both newly generated seed-block arms and the reversal.
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
We tested a generic instruction to inspect actual request strings before
matching them, with no email-specific patch or supplied answer. It did not
produce that behavior in any of the 32 attempts. Exact answers fell from 7 to
4, correct retrieval fell from 24 to 18, and generated tokens rose by 29%.
We are retiring this instruction, not concluding that learning to inspect
the input is impossible. A useful next intervention must actually teach or
elicit inspection before testing its effect.

A separate group-relative reward update used the three mixed groups. It
credited 12 actual final answers and retained the other 20 zero-signal
attempts in the 32-attempt denominator. Training took about 33 seconds.
The completed readout did not improve the main panels:

| Fixed starting model versus new reward update | Before | After |
|---|---:|---:|
| Short answers, same fresh seed block | 25/32 | 25/32 |
| Longer-input answers | 10/16 | 10/16 |
| Third/fourth-occurrence answers | 10/16 | 11/16 |

All 64 outcomes were available. The single additional exact answer restored
two required spaces after otherwise identical retrieval. Another attempt
recovered a working search program but still copied the final answer
incorrectly; a different attempt lost usable program generation. Overall,
59 of 64 complete model-call paths were unchanged. This is not an established
retrieval improvement or a replicated RL gain.

This recipe changes both training questions and reward baseline relative to
the earlier update, so it does not isolate the choice of RL objective. We
are now checking its original training attempts to distinguish local learning
without transfer from an update too small to change sampled behavior. A CPU
audit also found that reduced-precision serving perturbs the small updates,
but does not erase them. It is a possible noise source, not an explanation
that rescues the accuracy claim.

Evidence: `analyses/openai-mrcr-sft32-fresh8-g4-mechanism-2026-09-12/REPORT.json`.
Completed follow-ups:

- `analyses/openai-mrcr-fresh8-rloo-paired-2026-09-12/readout-002.json` and
  `CHANGED_PATHS.md` document every paired outcome and changed model path.
- `analyses/openai-mrcr-fresh8-literal-inspection-independent-2026-09-12/outcome-002/`
  documents the unsuccessful instruction and its actual behavior.
- `analyses/mrcr-serving-precision-audit-2026-09-12/REPORT_V2.md` separates
  saved-weight and implementation evidence from unobserved live GPU tensors.

## The supervised routine extends to third and fourth occurrences

The demonstrations taught the model to retrieve the first or second response
to a repeated request. We froze 16 new conversations asking for the third or
fourth response, eight of each, before running either model. Exact conversation
and target overlaps with prior research panels were excluded.

The base model scored 0/16; the supervised-trained model scored 10/16. All
outcomes were available, with five gains in each requested position. The
trained model selected the correct text in 13/16 attempts. Three then omitted
only the required two trailing spaces. Of the other three failures, two
searched for altered request wording and one exhausted its output allowance
before completing an action. The native audit accounted for all 83 model
calls and replayed all 32 episode records without integrity discrepancies.

Together with the separate longer-input result, this supports a narrow but
useful conclusion: the learned Python search routine is not limited to the
exact lengths and occurrence positions in its demonstrations. Both tests use
the same public repeated-request task family. Neither shows new-task transfer,
recursive delegation, or freedom from unknown pretraining exposure.

Evidence: `analyses/openai-mrcr-fourneedle-ordinal-transfer-independent-2026-09-12/`
contains the paired outcome, native-validation-v2 and residual mechanism reports.

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
successful RL method. The stronger result remains supervised retrieval
transfer to longer conversations and new occurrence positions. We are testing
whether more informative reward contrasts improve final delivery, while
preparing a different task with verifiable intermediate results to investigate
decomposition itself rather than continuing to optimize copying alone.

The already-presented advisor deck remains a historical snapshot. This report
is a post-meeting research checkpoint, not a silent revision of that deck.
