---
schema: research-checkpoint-v1
updated_utc: 2026-09-12T21:00:00Z
status: exploratory
questions:
  - Does the learned retrieval routine work on new, longer conversations?
  - Does a competent controller supply enough variation for reinforcement learning?
---

# Retrieval transfers, but familiar questions give RL no contrast

Two new experiments answer different questions. The supervised-trained model
did substantially better on new, longer conversations. However, repeated
attempts on familiar training questions were so consistent that our
group-relative RL method would receive no learning signal.

## The learned routine worked on new, longer inputs

We selected 16 additional public conversations before observing either
model's answers. Their conversation content ranges from about 16,000 to 30,000
tokens. Each task asks for a particular earlier reply, copied exactly with a
supplied prefix. The conversation lives in a Python-accessible file; it is
not inserted wholesale into the model's neural input.

| Outcome on the same 16 new conversations | Starting model | Supervised checkpoint |
|---|---:|---:|
| Exact final answers | 0/16 | 10/16 |
| Exact target text visible in a clean tool observation | 0/16 | 15/16 |
| Available outcomes | 16/16 | 16/16 |

The trained model made ten corrections and no regressions. In five other
cases it obtained the correct text but omitted two required trailing spaces
when returning it. The saved model tokens contain these omissions; they are
not another runtime-trimming defect. In the sixth failure, the question asked
for a second email, but the generated program searched for a second program.
It found no matching request and failed to complete a repair within its limit.
These are different failures: five in copying, one in selecting the request.

Both models used the same output-preserving runtime, task prompts, requested
seeds, six-action total limit, and grader. The checkpoint was fixed at the
previously declared 32-update supervised endpoint; no new training or
checkpoint selection occurred for this comparison. No helper was called.
All 83 model calls returned, with no missing outcomes or unknown call costs.

The starting model made 51 calls, consuming 100,388 input and 14,560 output
tokens. The trained model made 32 calls, consuming 34,352 input and 11,431
output tokens. These are observed model-token costs, not a claim of matching
compute or a large wall-clock speedup. The respective owners took about 268
and 252 seconds, including startup and cleanup.

This is meaningful length transfer within the same retrieval task family.
The 16 new conversations exclude exact core/target overlaps with the 48 earlier
short records and with each other. Shared introductory examples, semantic
similarities, and unknown pretraining exposure remain possible. The initial
neural input was only 768–775 tokens, so this is not evidence of a larger
neural context window. It is also not learned recursive decomposition or
generalization to a new task type.

The source-selection length band was 16,384–32,768 tokens including the answer;
the reported content range above excludes that answer. This distinction keeps
the input-size claim separate from the acquisition filter.

Evidence: `analyses/openai-mrcr-long-transfer-independent-2026-09-12/outcome/RESULTS.json`,
SHA `7fc003a651b82cb361c0af766c587e90a6fa90a5a5163031bbc4f0238db79a20`.
The independent analysis reproduced scores, pairing, physical prefixes and
costs with no integrity discrepancies. Earlier service-startup failures are
retained separately and contain no model-quality measurements.

Failure evidence: `analyses/openai-mrcr-long-transfer-failure-mechanism-2026-09-12/RESULTS.json`,
SHA `948dd57b5bb841ea6fd7efae6befdcbeb953caefc541556b9e382403491948d5`.

## A high success rate can still give group-relative RL zero signal

Separately, we sampled four attempts at each of the first eight original
training questions from the same supervised checkpoint. It answered 28 of 32
attempts exactly. But every question produced the same answer in all four
attempts: seven questions were always correct; one always added two extra
line breaks after the otherwise correct answer.

All 32 attempts used teacher-equivalent first programs, printed the correct
target, and finished with a text answer. The extra line breaks were generated
by the model, not introduced by the runtime. The unchanged continuous
similarity score was also identical within each question.

Our group-relative learning signal compares attempts at the same question.
If all four receive the same reward, none is better or worse than its peers.
Every relative advantage in this batch is exactly zero. Changing the learning
rate, importance weights, or which action receives credit cannot turn those
zero advantages into a useful update. We did not take an optimizer step.

This does not rule out other RL objectives or other training examples. It
identifies why this particular competent-but-repetitive batch cannot support
the planned group-relative update. The 32 attempts are eight training-context
units with four samples each, not a heldout accuracy estimate.

## Earlier training restored variation but lost accuracy

We next evaluated the saved 16-update checkpoint at the original sampling
temperature. It answered 6 of 31 available attempts exactly; one additional
attempt was unavailable and is not counted as wrong. Three of seven complete
four-attempt groups mixed successes and failures, unlike the later checkpoint.
However, it exposed clean target text in only 7/32 recorded attempts, and none
of its first programs matched the teacher program's syntax tree. These are
separate observations: different code is not automatically incorrect code.

Among 31 available paired attempts, the earlier checkpoint lost 21 exact
answers and gained none. We chose this midpoint after seeing the later
checkpoint's repeated answers, but before seeing any midpoint outcomes.
It is an exploratory training-dose diagnostic, not the originally declared
training endpoint. Simply stopping SFT earlier did not preserve the useful
routine while restoring variety. There is reward contrast for a future
whole-controller RL test, but not the clean copying-only test we wanted.

Evidence: `analyses/openai-mrcr-sft16-g4-mechanism-2026-09-12/MAIN_REPORT_001.json`.
All 81 actual native model calls returned; the one unavailable episode requires
separate classification rather than a silent score substitution.

We are separately testing higher sampling temperature at the later checkpoint.
Both comparisons keep the eight questions and requested seeds fixed; together
they are not a complete temperature-by-training-dose experiment.

The first higher-temperature attempt failed at the connection layer: 238
client attempts returned connection errors and none returned a model answer.
We stopped its collector and preserved the failure records. Its roughly
631 seconds are an operational loss, not evidence about temperature or model
quality. A second attempt passed a native-request fixture but failed again:
that fixture omitted an audit hook used in the real experiment. We then found
the hook's hardcoded requirement that temperature equal 0.5. It rejects the
temperature-1.0 request before network transmission, and the client reports
this as a connection error. We stopped the second collector after 72 failed
requests; its owner took 290 seconds including cleanup. The next repair
targets that specific check and includes it in the fixture. Both failed
attempts remain separate from model-quality results.

If genuine reward variation appears, we can take an informative RL step and
evaluate fixed checkpoints on separate problems. If first actions are
identical across a group, their relative gradients may cancel; an all-action
versus final-only comparison would then add little information. We will check
that before spending compute on two effectively identical updates.

We are also preparing a different RL objective that compares exact success
with a fixed reward baseline, rather than with the other attempts at that
question. That objective can have a nonzero gradient on this batch. A single
small update focused on final-answer tokens will test whether it improves
delivery while preserving retrieval. Preparation is not a training result,
and changing the objective is not proof that the new update will help.

## Better source coverage did not yet mean more exact answers

A separate harness experiment gave the final answer four source passages
either split evenly between two helpers or selected flexibly across their
eight candidates. We used the same 12 previously examined MuSiQue questions.
The candidate pool contained all annotated supporting sources in 9/12 cases.
Flexible allocation increased complete selected-source coverage from 3/12 to
6/12, but both policies answered 3/12 questions exactly: one gain, one loss.

All 60 physical calls were available and their actual inputs were verified.
The natural policies would use 36 calls for fixed allocation and 48 for flexible
allocation; shared selectors reduced the actual experimental total. Both
valid final inputs contained four paragraphs, but not equal numbers of tokens.
Coverage alone does not establish that the required facts were combined
correctly. We are reviewing the source relations and answers, not treating
this coverage gain as an end-to-end accuracy gain or a learned routing method.

The case review makes the tie more informative. The new exact success recovered
a genuinely missing source linking San Francisco to the Transamerica Pyramid.
The lost exact success had previously copied a Copenhagen performance venue
without evidence for the requested death location; the new answer instead
used the death city of an unrelated person. Another answer recovered the right
baseball year but remained exact-match wrong because it was a full sentence.
These distinctions do not change the official 3/12 scores. Also, some coverage
gains kept the two-plus-two allocation, so global reranking and the extra call
cannot be separated from quota flexibility in this experiment.

Evidence: `analyses/musique-flexible-four-source-independent-2026-09-12/outcome/REPORT.json`,
SHA `3ff7230537b4ed3f17b35d9dcbaa896ead0cf22267785fa5aae07339a3f2c2b2`.

## Next decision

The next harness comparison holds all 12 selected source sets fixed and adds
a short, source-quoted relation-extraction step before answering. It asks
whether making entity links explicit helps the model use facts it already has.
The extra model call is counted; this is not a compute-matched test or a new
retrieval result. It is a small exploratory step toward adaptive decomposition.

The most promising direction is now concrete: preserve the learned access
and selection skill, improve delivery of the answer, and obtain informative
training feedback without confusing repeated answers with successful RL.
These experiments still do not establish a general RL improvement.
