---
title: Evidence and methods for the advisor meeting
meeting_date: 2026-09-11
status: exploratory_evidence_synthesis
evidence_cutoff_utc: 2026-09-11T03:45:00Z
---

# Evidence and methods for the September 11 advisor meeting

For a first reading, start with [the speaker guide](speaker-guide.md), which
explains every slide and defines the terminology. Here, “root” means the main
model and “child” means a helper call. “Native-authenticated” means the reported
outcome was checked against the recorded model request and response. A
“performed” calculation is one the recorded program actually carried out.

This document explains the current evidence in plain language. It is meant both as a source for the
advisor presentation and as a guide to what the experiments actually establish. The central lesson
is that recursive language-model systems have at least three separable problems:

1. The root model must understand which operation the question asks for.
2. The child model must return semantically correct evidence tied to the right source records.
3. The root must retain, combine, and use that evidence correctly.

The strongest positive results concern the first problem and a specific part of the second. The
strongest negative results show that improvements at either level do not automatically solve the
third.

## Finding 1: varied demonstrations taught the root to perform the requested operation

### Question

Can a small root model learn to choose and execute the calculation requested by the current
question, instead of repeating one familiar calculation regardless of the question?

### Method

The root was a Qwen 3 4B model with a LoRA adapter. It received 72 complete tool-using
demonstrations and six fixed supervised updates. Each demonstration contained three root actions:
request labels from a real child-model call, calculate over the returned label map in Python, and
return the printed scalar. The questions varied category choices, user scopes, thresholds, and
primitive versus composed operators. Only root-action tokens received training loss; child answers,
tool observations, and earlier history were masked. Wrong child labels were retained rather than
replaced with gold.

The protected evaluation contained eight inputs held out from this root-training
process and 72 questions per policy. The starting root already had earlier training;
the comparison is with and without this further training, using the same fixed
helper. Analysts read all 149 available sampled programs across the protected
and development evaluations, together with their linked tool observations. They
judged separately whether the final number
was correct and whether the program actually performed the requested operation. Generated programs
were not re-executed during analysis.

### Result

The clearest endpoint is “correct and performed”: the answer was correct and the sampled program
faithfully executed the requested calculation.

| Protected endpoint | Before training | After training |
|---|---:|---:|
| Native answer available | 64/72 | 71/72 |
| Correct answer | 28/72 | 57/72 |
| Requested calculation performed | 18/72 | 62/72 |
| Correct and performed | 15/72 | 52/72 |
| Correct and performed on composed questions | 1/48 | 33/48 |

The slide deliberately shows a related check with changed names and numbers,
not the table above. We kept the same trained checkpoint and eight source
inputs, but replaced user names, weights, and thresholds. On those 72 questions,
correct-and-performed answers rose from **12 to 50**. The requested calculation
was performed in 17 before-training paths versus 62 after-training paths.
The baseline had 55 available outcomes; the trained model had all 72.
Even granting success to every one of the 17 missing baseline outcomes gives
only 29, below 50. This supports transfer across that limited change; it is
not a new training run or a new source-text evaluation.

The paired correct-and-performed gain was positive in all eight context clusters. On the 63 pairs
where both policies returned an available answer, the trained policy had 31 wins and no losses on
that endpoint. Missing-outcome bounds still favored training: the gain in correct-and-performed
answers was between 29 and 38 out of 72.

The result was not merely a preference for returning zero. On the original nonzero-gold stratum,
correct-and-performed outcomes rose from 10/46 to 32/46. Two related robustness readouts also showed
positive nonzero gains: 9/46 to 27/46 under fresh sampling seeds and 7/44 to 25/44 after changing
metadata. Composed nonzero gains were 1/29 to 20/29, 2/29 to 16/29, and 0/27 to 14/27.

### Worked example

One protected question asked for the total weight of description-category records belonging to
users who also had an entity-category record. The before- and after-training policies received the
same 16 child labels. The earlier policy summed the entity records and returned 14. The trained
policy first found the users with entity records, then summed description records for those users,
and returned the correct answer, 21. This example holds child evidence fixed and isolates a better
root calculation.

### Limits

The table above comes from one training run. The original, fresh-seed, and changed-metadata numbers
are three related readouts of the same trained checkpoint on the same source contexts. Fresh seeds
test sampling sensitivity; changed metadata tests a limited parameter change. A separate-corpus
training replication is described below, but it still uses the same research-exposed evaluation
contexts.

All operator families appeared in training, so this does not demonstrate a new unseen operator.
The underlying public question records had been exposed to child training and earlier research,
although root-training and protected context groups were disjoint. There were 26 zero-gold protected
questions, and some zero answers were coincidentally protected by empty selections. The explicit
nonzero analysis reduces, but does not eliminate, that concern. Finally, ten trained-policy errors
used the correct calculation but wrong child labels; root learning cannot repair evidence it never
received correctly.

### Completed follow-up: a separately captured training corpus gave a similar SFT result [S2]

**Question.** Does the supervised result depend entirely on the first set of 72 training
trajectories?

**Method.** A second set of 72 demonstrations was captured from newly selected source groups. The
new run began from the same already-trained fixed24 root adapter—not an untouched released base—and
used fresh Adam state, six full updates, the same helper, and the same 72-question metadata
evaluation. The two training corpora were captured separately, but the eight evaluation contexts
were shared and already research-exposed.

**Result.** Correct-and-performed answers were 12/72 at the shared fixed24 start, 50/72 after the
original-corpus SFT, and 55/72 after the new-corpus SFT. The new run had 70 authenticated answers;
two physically attempted endpoints were unavailable and remained NULL, so its planned-denominator
bounds are 55–57/72. Among the 70 available answers, all acquired genuine child outputs, 64
performed the requested computation, and 55 were both performed and correct. The corresponding
strict-answer counts were 17/72 at fixed24, 53/72 after the original training corpus, and 57/72
after the new corpus, with new-run bounds of 57–59/72.

**Interpretation and limits.** This is bounded evidence that the earlier gain was not unique to one
captured training corpus. It does not establish corpus superiority or performance on new inputs:
there are two training corpora but only one shared eight-context evaluation panel. Six available
new-corpus paths did not perform the requested computation, including two that nevertheless reached
the correct number by coincidence. This motivated the newly selected evaluation source
contexts reported next; additional training realizations remain useful.

### Completed follow-up: both trained roots transferred to newly selected inputs [S3]

**Question.** Do the gains appear on source groups outside the earlier root-evaluation inventory?

**Method.** Eight source groups were deterministically selected from public IDs before gold was
computed and before model outputs. The same 72 questions were asked of the fixed24 starting adapter,
the original-corpus SFT6 policy, and the new-corpus SFT6 policy. A success required both the correct
number and an actual path that used genuine child evidence to perform the requested operator, scope,
categories, threshold, and final reduction. Analysts read all 207 available paths; sampled programs
were not re-executed.

**Result.** Faithful-and-correct totals were **12/72** for fixed24 (seven NULLs, bounds 12–19),
**55/72** for original-corpus SFT6, and **53/72** for new-corpus SFT6 (two NULLs, bounds 53–55).
The corresponding exact-answer-only totals were 23, 57, and 54. Fixed24 had no faithful-and-correct
composed result; the trained policies had 35/48 and 34/48. Each trained policy had more observed
faithful-and-correct answers than fixed24 in all eight context clusters.

**Interpretation and limits.** This strengthens the bounded transfer evidence: the result is not
only a changed-number readout on the original source groups. It remains the same nine studied task
families, fixed child, and metadata transformation. “New” means absent from named root experiment
inventories, not unseen in model pretraining, child training, or every research catalog. Manual
semantic judgments were outcome-visible; MAIN checked the full arithmetic and selected traces, not
all 207 judgments independently.

## Finding 2: matching input and output rows repairs late-batch correspondence

### Question

When 48 natural-language inference records share one prompt, can explicit row numbers keep each
output label attached to the intended input record, especially late in the batch?

### Method

Sixteen newly selected MultiNLI contexts were used as the clustered units. Each contained 48
premise–hypothesis pairs. The study crossed three visible-reference conditions—misleading IDs,
unrelated IDs, and aligned IDs—with two input formats and two output formats:

- The input either omitted row numbers or displayed rows 0 through 47.
- The output either returned labels alone or returned each label with its row number.

All conditions used the same released Qwen 3 4B model, semantic task, record order, sampling seed
within a context, and three-way label choices. Structured decoding enforced each condition's output
grammar. Every one of the 192 calls returned a native-authenticated, contract-valid answer.

The primary population was positions 17 through 48, where the original long-batch degradation was
largest. Each reference-specific late cell therefore contains 16 contexts × 32 late positions =
512 labels. Pooling the three reference conditions gives 1,536 labels per input/output bar.

### Result

The four pooled late-position bars are the most transparent summary:

| Input format | Output format | Correct late labels | Accuracy |
|---|---|---:|---:|
| No input rows | Labels only | 577/1,536 | 37.57% |
| No input rows | Row-keyed output | 634/1,536 | 41.28% |
| Input rows shown | Labels only | 626/1,536 | 40.76% |
| Input rows shown | Matching row-keyed output | 1,340/1,536 | 87.24% |

The output-row gain without input rows was `634 − 577 = 57` labels, or 3.71 percentage points. The
gain with matching input rows was `1,340 − 626 = 714` labels, or 46.48 points. The prospectively
specified interaction was therefore

`(1,340 − 626) − (634 − 577) = 657 / 1,536 = 42.77 percentage points`.

All 16 context-level interactions were positive. The matched-row gain was similar for misleading,
unrelated, and aligned reference conditions: +235/512, +242/512, and +237/512 late labels. This
matters because it shows that the effect is not only relief from one specially misleading ID.

### Worked example

Suppose row 7 says “A dog is running” and asks whether “An animal is moving”; its correct label is
entailment. In a long batch, a bare list of 48 labels gives the model no explicit output-side handle
for keeping the seventh decision attached to row 7. If the input record says `row: 7` and the output
must begin `{"row": 7, "label": ...}`, the association remains visible across the prompt and the
answer. The experiment shows a large improvement for exactly that matched package.

### Limits

The result establishes an encoding-package effect, not an internal attention or binding mechanism.
The output grammar forced row tokens, so exact row emission is not evidence that the model freely
learned to copy them. In this first study, sequential numbers might help because they are stable keys, because the model
can count in order, or both. A frozen follow-up now compares sequential, permuted numeric, and opaque
keys on new contexts. That follow-up is now complete, as described below.

The 1,536 labels in a bar are not independent trials. The inferential unit is the context, giving 16
paired clusters. The source groups were excluded from named experimental inventories, not guaranteed
unseen in pretraining. One seed per context limits sampling-generalization claims.

### Completed follow-up: the keys did not need to count in order [H3]

**Question.** Does the benefit require ordinary row numbers, or can each record
and answer share an arbitrary tag? This separates a practical matching method
from an aid that depends on following the sequence 0, 1, 2, and so on.

**Method.** Sixteen more MultiNLI batches each contained 48 statements. Each
batch was tested with no matching keys, sequential numbers, shuffled numbers,
and arbitrary text tags. Each version was crossed with misleading, unrelated,
and aligned public record identifiers. The software required the keys and answer
order; the model still chose the labels. All 192 calls returned valid outputs
whose recorded requests, model identity, and generated tokens were checked.

**Result.** Accuracy on the later 32 records was:

| Matching method | Correct later labels | Accuracy |
|---|---:|---:|
| No added matching keys | 525 / 1,536 | 34.2% |
| Sequential row numbers | 1,312 / 1,536 | 85.4% |
| Shuffled numbers | 1,323 / 1,536 | 86.1% |
| Arbitrary text tags | 1,304 / 1,536 | 84.9% |

Arbitrary tags gained 50.72 percentage points over no matching keys and were
0.52 point below sequential numbering. Shuffled numbers gained 51.95 points
and were 0.72 point above sequential numbering. Both improved every one of the
sixteen paired batches and passed their separately specified advance criteria.
There were no missing calls. These are new inputs, so the bar heights should not
be interpreted as changes from the preceding four-condition experiment.

**What this changes.** Ordinary counting order is not necessary for the large
benefit in this setup. Matching tags are therefore worth testing inside a full
task. This does not establish that the model learned to manage tags itself or
identify a unique cause inside the model. The tags were supplied by the output
rules, answer order stayed fixed, and arbitrary tags used more tokens than numbers.

**Limits and cost.** The sixteen batches, not the repeated label slots, are the
paired units. There is one model and one random generation seed per batch. Source
selection required complete three-label groups and excluded named earlier research
inventories; it does not establish that the model had never seen the public text.
The 192 new calls used 787,869 input and 100,581 output tokens, with no cache reuse.
The complete GPU job took about 10.8 minutes. The reviewer independently parsed
and counted the outputs but also authored two pre-run scoring corrections; that
overlap is disclosed in the audit.

### Completed follow-up: matching keys prevented the large-batch collapse [H8]

**Question.** At what batch size does the labels-only interface break down, and do matching keys
retain their advantage on newly selected inputs?

**Method.** Sixteen newly selected MultiNLI context clusters supplied nested prefixes of 8, 16, 32,
48, and 64 records. Each prefix used the same gold labels under three paired output formats: labels
only, matching sequential row numbers, and matching opaque tags. All 240 calls were available,
contract-valid, and native-authenticated. Prefix sizes are nested measurements, not independent
replications.

| Records per call | Labels only | Sequential keys | Opaque keys |
|---:|---:|---:|---:|
| 8 | 82.8% | 87.5% | 84.4% |
| 16 | 82.0% | 86.7% | 87.1% |
| 32 | 56.4% | 86.9% | 85.7% |
| 48 | 49.2% | 85.4% | 85.5% |
| 64 | 43.8% | 84.6% | 83.4% |

The prospectively defined onset rule first passed at 32 records. At 32, 48, and 64 records both keyed
formats improved all 16 paired context clusters over labels only. At 64 records the gains were
+40.72 and +39.55 percentage points.

**Interpretation and limits.** Matching keys preserved label accuracy as batches grew in this
Qwen3-4B encoding package. This is not a universal 32-record limit, an internal-mechanism result,
or evidence that a complete RLM improves. There are 16 context clusters—not thousands of independent
label trials—and keyed formats change prompt/output tokens as part of the intervention.

### Completed follow-up: the matching-key pattern also appeared in Qwen3-8B [H4]

**Question.** Does the late-batch matching-key effect appear in another released model size from the
same family?

**Method.** Qwen3-8B answered the same already exposed 16 contexts under labels-only, sequential-key,
and opaque-key conditions. The 144 calls were newly sampled; the 4B values below are reused, not
rerun. All 144 8B calls were native-authenticated, available, and contract-valid.

**Result.** On the later 32 positions, 8B scored 484/1,536 with labels only, 1,313/1,536 with
sequential keys, and 1,245/1,536 with opaque keys. The corresponding 4B counts on this panel were
525/1,536, 1,312/1,536, and 1,304/1,536. Thus both matching-key formats were far above labels-only
for 8B, while the difference between sequential and opaque keys was larger than it had been for 4B.

**Interpretation and limits.** This is one 8B realization on the same exposed panel and within the
same model family. Model weights, tokenizer, and rendered tokenization all changed, so this is not a
pure capacity comparison. It supports testing the matching interface on fresh evaluation contexts;
this particular check does not establish behavior on another family or in a complete
main-model task. The subsequent H5 check below extends the direction to Mistral;
the complete-task question remains open.

## Finding 3: output identifiers can redirect the model toward the wrong record

### Question

Are record IDs merely harmless formatting, or can an ID make the model answer for a different
record than the one occupying the output position?

### Method

In one exact-contract comparison, the requested output IDs either matched the displayed records or
were cyclically shifted by 17 positions. The two arms used the same ID multiset, record texts, order,
model, and paired seeds. Only the mapping between output position and requested ID changed. A fresh
16-context study added unrelated, token-length-matched IDs. The label choice remained free even
though the decoder forced the requested output ID.

### Result

Matching IDs scored 622/768 displayed labels (81.0%); shifted IDs scored 280/768 (36.5%). On 522
positions where the displayed and ID-named records had different gold labels, 71.5% of outputs
followed the ID-named record, while 15.3% followed the displayed record.

The fresh-context comparison reproduced the behavior. Matching IDs scored 79.56%, unrelated IDs
80.47%, and shifted visible-record IDs 38.80%. Unrelated minus shifted was +41.67 points and positive
in all 16 context clusters. Because unrelated IDs perform like matching IDs, the shifted penalty is
not well explained by unusual identifier tokens alone.

### Worked example

Imagine that the displayed row is an entailment example, but its output slot is stamped with the ID
of another visible row whose correct label is contradiction. The model frequently writes
“contradiction.” That label is locally sensible for the record named by the stamp, but globally
wrong for the displayed row. Replacing the stamp with an unrelated code largely restores the
displayed row's label.

### Limits

This is behavioral evidence of semantic redirection under constrained decoding. It does not prove
how attention moves internally, and it does not show unconstrained ID generation. The fresh panel is
a source-context replication, but it was adaptively motivated by the earlier result and used the
same model and task family.

## Finding 4: child interface training improved labels, but gains depended on the contract

### Question

Can supervised child training improve a compact label interface without sacrificing the original
six-class interface?

### Method

The child was trained on two answer contracts. The full contract returned one of six semantic
classes. The compact contract projected those classes into A, B, or other for the downstream task.
Pure-contract continuations and a deterministic mixed-contract continuation were evaluated on the
same research-exposed panel. The primary scientific scores were independently decoded from native
responses and checked against frozen host labels.

### Result

Pure compact-contract training gained 90/768 compact labels relative to the starting child, but lost
18/768 full-contract labels. That 2.34-point retention loss exceeded the frozen 2-point limit, so the
arm was not promoted.

Mixed-contract training improved compact labels by 75/768 and full-contract labels by 9/768 relative
to the starting child. It passed the narrow local gate: +9.77 points on the compact interface with no
full-interface loss. This is evidence that balanced interface exposure can teach a useful compact
contract without the observed catastrophic tradeoff of pure compact training.

### Limits

The interface intervention bundles prompt wording, schema, and target representation. The evaluation
panel is research- and optimizer-exposed, so this is not outcome-pristine transfer. A larger compact
score is not automatically a better downstream answer, as the next finding demonstrates.

## Finding 5: better child labels did not reliably produce better aggregate answers

### Question

If the experimenter supplies the acquisition plan and an exact deterministic reducer, are child
label errors still enough to break the final composed answer?

### Method

Eight episodes were nested within four source clusters: four 64-record and four 256-record episodes.
The experimenter fixed complete 32-record chunks and the public reducer. The same reducer was applied
first to child-predicted labels and then, diagnostically, to private oracle labels. The oracle was
used only for scoring, never placed in a model prompt. A later comparison replaced the original
child with the mixed-contract child while preserving the supplied plan.

### Result

The original child classified 1,129/1,280 records correctly, or 88.20%, but achieved 0/8 exact final
answers. Oracle labels produced 8/8, demonstrating that the acquisition and reducer were sufficient
when their evidence was correct.

The mixed-contract child improved six-class correctness by 19 labels and projected A/B/other
correctness by 20. Downstream exactness moved only from 0/8 to 1/8. Total absolute scalar error
worsened from 117 to 130, despite improvement in three of four source clusters. The downstream gate
failed.

### Limits

Eight episodes from four clusters are a small, correlated diagnostic, not a general estimate. The
supplied plan is an experimenter-authored ceiling, not learned root planning. The result does not say
that the mixed child is worse overall; it says that its changed error pattern was not reliably better
for this exact reducer.

## Finding 6: selective repair found real errors but rarely fixed exact composition

### Question

Can native child confidence identify the labels worth rechecking, and can an equal-budget second pass
repair enough of them to recover exact final answers?

### Method

For each episode, the bottom-confidence 25% of labels were selected without using gold. A
hash-uniform 25% sample was the equal-budget control. Selected records were repacked and classified
again; valid second-pass labels unconditionally replaced the selected first-pass labels. A later
study compared confidence selection with a task-sensitive score and tested agreement between two
second-pass samples. Missing batches would have made the episode NULL; none were missing.

### Result

Confidence selection captured 107/151 initial errors, versus 34/151 for uniform selection. It made
71 repairs and 31 regressions, a net gain of 40 correct labels; uniform rechecking had a net change
of −9. This is a useful ranking result.

The downstream gain was much smaller: confidence rechecking repaired one of eight answers to exact,
while uniform repaired none. In the two-sample extension, confidence-single reached 2/8 exact and
task-aware-single 1/8. Agreement-based abstention did not act as a reliable verifier. The two samples
agreed on 42 wrong labels under confidence selection and 48 wrong labels under task-aware selection,
showing that repeated samples can share errors.

### Limits

The confidence score ranks errors within this model and structured-output setting; it is not a
calibrated probability. Repacking changes context, and the same child generates both passes. The
result motivates a genuinely different verifier or evidence source, not more same-child samples.

## Finding 7: asking directly for sufficient statistics made the composed answer worse

### Question

Can the child skip the full label map and directly return, per user, whether category A exists and
the sum of category-B weights—the exact statistics needed by the reducer?

### Method

The same eight episodes and exact 40 chunks were used. For each eligible user, the child returned a
Boolean `has_target_a` and a nonnegative integer `target_b_weight_sum`. Across chunks, code ORed the
Booleans, added the sums, and counted B weights only for globally A-qualified users. Any missing
chunk would make an episode NULL, and any authenticated malformed chunk would score as an observed
invalid result. Gold entered only after the 40 native results and eight merges were fixed.

### Result

All 40 calls were HTTP 200, native-authenticated, and schema-valid; all eight episodes were complete.
The direct-statistics bundle nevertheless scored 0/8 exact and reduced absolute error in 0/8 paired
episodes. Total episode absolute error was 1,805, compared with 117 for the historical full-label
control. Mean error was 86.0 versus 7.25 at size 64 and 365.25 versus 22.0 at size 256.

The Boolean A-existence flags were correct for 28/32 episode-user pairs. The dominant failure was the
B-weight sum, whose per-user absolute errors totaled 1,706. A compact answer is only useful when the
model can compute it reliably.

### Limits

This retires the exact bundle of task-aware prompt, statistics schema, and model-performed summation.
It does not prove that sufficient statistics are intrinsically bad or isolate compression alone.
There was no fresh human semantic review of the 40 outputs; the audit used native token evidence,
strict schema parsing, and trusted host arithmetic. The same small, research-exposed four-cluster
panel limits generalization.

## How the findings fit together

The evidence supports a staged account rather than one headline accuracy number:

- Diverse trajectory supervision can teach the root to choose and execute the requested calculation.
- Long batched child outputs suffer a large correspondence failure that matched input/output rows can
  repair.
- Child training and confidence-based repair can improve local labels.
- Exact aggregation is brittle: a few consequential label errors, correlated recheck errors, or bad
  model-computed sums can erase those local gains.

The most defensible research question is therefore: **which learned or engineered interfaces let
useful local predictions remain attached to their sources and survive recursive composition?** This
is narrower than claiming that decomposition is solved, but it is also more diagnostic.

The larger-update reinforcement-learning continuation is final. Its fixed checkpoint 6
scored 47/72 planned answers (69 available; bounds 47--50), versus 55/72 for the unchanged start
(71 available; bounds 55--56); exact paired missing-data bounds are -9 to -5. On composed tasks,
42/45 available endpoints acquired a complete child map, but only 32 performed the requested
operator/scope/threshold and only 26 were both faithful and strict-correct. Four further strict
answers were zero coincidences from wrong computations. End-answer reward does not distinguish
those paths from genuine successes; whether that caused the decline is untested. The result argues
against repeating this exact high-learning-rate terminal-only recipe, not against reinforcement
learning in general. The separate-corpus supervised study is now complete: correct-and-performed
answers were 55/72, with two attempted endpoints unavailable and bounds of 55–57. Because both
training runs used the same research-exposed eight-context evaluation panel, newly selected
evaluation inputs remain the main missing test.

### Smaller reward updates: no clear improvement (R2)

The follow-up restarted from the same trained main model at learning rate 1e-5,
five times smaller. It completed all eight collection windows and made six actual
updates; two windows had no useful contrast between their rewards and made no
update. All 192 attempted training questions were retained, with 188 available
outcomes. Thirteen mixed-reward groups supplied 50 eligible training attempts.

The final checkpoint gave 52 correct answers out of 72, with four missing outcomes
and possible totals 52–56. The reused starting model gave 55, with one missing
outcome and possible totals 55–56. On the primary 48 questions combining operations,
both models gave 36 correct answers. Among 46 known pairs, there were three wins
and three losses. The missing-outcome effect range is −4.17 to +4.17 percentage
points; this range describes missing evidence, not statistical uncertainty.
The independently prepared, agent-authored complete path review also found no
combined-operation gain: 33/48 faithful-and-correct in each model. It was not a
human or blinded review.

This small-dose result gives no reason to keep repeating learning-rate changes
alone. A useful follow-up would first check whether rewarding intermediate
calculations supplies a better learning signal. That is a hypothesis, not a
demonstrated treatment. The smaller-rate implementation used a memory-saving
calculation of the same objective; it was not bitwise identical to the first
dense update in the earlier run. Neither comparison isolates every numerical detail.

The full workflow took 3,443.90 seconds: 1,690.67 generating training attempts,
401.14 updating weights, and 601.10 evaluating, plus service/orchestration overhead.
The native record contains 1,307 physical requests, 1,303 returns, 2,523,653 known
input tokens, 173,202 output tokens, and four requests with unknown usage. The
producer's obsolete-layout cost summary said zero; the native audit corrects it.

### Cross-family matching-tag check (H5)

Mistral-7B-Instruct-v0.3 used the same 16 previously selected batches and three
identifier conditions as the Qwen comparison. All 144 calls were available,
native-authenticated, and valid. Accuracy on the later 32 answers was 490/1,536
(31.9%) without added tags, 857/1,536 (55.8%) with row numbers, and 807/1,536
(52.5%) with arbitrary tags. Row numbers helped 15/16 input groups; arbitrary
tags helped 16/16. Both met the prospectively specified directional checks.

This extends the direction of the effect to a second family, but Mistral's final
accuracy remains substantially below the Qwen models. It is not a clean model-size
or family comparison: tokenization, weights, chat templates, context limits, and
decoding also differ. No internal matching mechanism is established. The earlier
sealed report's interpretation about semantic label names is not needed for this
claim; the experiment did not separately manipulate those names.
An additive interpretation erratum preserves the original report while narrowing
those phrases; no scores or costs changed.

The audit checked exact request bodies, native token IDs, output keys, model
revision, and Mistral's own end token. Its minimal Python environment lacked
Transformers, so a disclosed wrapper used tokenizers 0.23.1 with the exact pinned
tokenizer file; all 144 outputs decoded identically. MAIN checked ten sealed
artifact/terminal hashes and independently recounted all 144 literal responses.
Parent time was 517.79 seconds; known usage was 666,729 input and 92,062 output
tokens. No missing usage or cache tokens were reported; billing was not measured.

### Balanced literal-tag reuse: matching mattered even with both sides tagged [H6]

**Question.** Does repeating the same arbitrary string beside an input and its
answer help, beyond having an arbitrary tag on each side?

**Method.** The released Qwen3-4B model answered the same exposed 16 batches of
48 MultiNLI records used in the stable-tag studies. Two disjoint, equal-character-
length tag sets, A and B, gave four input/output pairings: AA, AB, BA and BB.
Thus A and B each appeared equally often on each side. We crossed these pairings
with the three earlier visible-reference conditions: misleading, unrelated and
aligned names. There were 192 planned responses. Record order, reading task,
label options and intended positional target were unchanged. Each output position
always labeled the input at that position, even when the arbitrary tags differed.
The prompt explicitly instructed this and did not ask for input-tag copying.
The output rules fixed tags and order, leaving the reading labels to the model.

**Result.** All 192 calls were received, authenticated and format-valid. On the
prospectively selected later 32 positions, the matched AA/BB conditions scored
2,420/3,072 (78.78%); disjoint AB/BA scored 1,000/3,072 (32.55%). The paired
difference was 46.22 percentage points and positive in all 16 input batches.
The improvement was similar across the three reference conditions: 46.39 points
with misleading names, 45.70 with unrelated names, and 46.58 with aligned names.
There were no missing outcomes, invalid outputs or producer/auditor score
disagreements. The planned exploratory gate passed; this is not a fresh-input
confirmation. Repeated labels are clustered within 16 inputs, not thousands of
independent trials.

**Interpretation.** Reuse improved intended-position reading accuracy within
this interface. The comparison is more specific than adding tags versus none,
and its scores are not explained by malformed output. It does not identify an
internal binding or attention mechanism, establish freely learned tag copying,
or prove an improvement to complete-task answers. Equal character length does
not mean every paired prompt had exactly the same tokenizer length.

**Why this does not contradict H1.** H1's unrelated names avoided pointing to
another visible record; H6 changes an additional input/output bookkeeping tag
while preserving the positional instruction. H6's gain appears in all three
reference conditions. H1's 80% and H6's 33% are not interchangeable baselines:
the prompts, interfaces, panels and pooled measures differ.

**Checks and cost.** MAIN verified all 14 original seal entries, rerendered all
192 frozen requests, checked 9,216 requested tags, and recounted the 192 literal
response label arrays. Counts and audit summary agreed; the effect differed by
only about 7e-15 percentage points from floating-point arithmetic. The completed
attempt used 875,070 input and 237,078 output tokens, no cached tokens or unknown
usage, and 1,449.67 seconds of parent wall time. The earlier zero-science startup
failure remains separate. Next: test complete-task value and compare the same
prompt with output-format constraints turned on and off.

## Relation to prior work

We should not claim that numbered batch items, constrained JSON, semantic operators, or recursive
model calls are new.

- Cheng et al., [“Batch Prompting: Efficient Inference with Large Language Model APIs”](https://aclanthology.org/2023.emnlp-industry.74/), EMNLP Industry 2023, explicitly use indexed input/output correspondence and parsing.
- Lin et al., [“BatchPrompt: Accomplish more with less”](https://arxiv.org/abs/2309.00384), study batch order and position sensitivity and permutation-based mitigation.
- Park et al., [“Grammar-Aligned Decoding”](https://arxiv.org/abs/2405.21047), show that grammar masking can change model generation distributions; our schema arms are therefore different action spaces, not mere cosmetic formatting.
- Patel et al., [“Semantic Operators: A Declarative Model for Rich, AI-based Data Processing”](https://arxiv.org/abs/2407.11418v3), implement LOTUS and separate logical semantic operations from execution strategies. Earlier versions used the LOTUS title.
- Zhang et al., [“Recursive Language Models”](https://arxiv.org/abs/2512.24601), frame long-context reasoning around external state, recursive calls, and subsequent observations.
- The official [Lambda-RLM implementation](https://github.com/lambda-calculus-LLM/lambda-RLM/tree/3874d393483dc4299101918cf8e9af670194bd88) uses a fixed task menu and task-specific merge routines. It is a useful supplied-plan comparator, not evidence that a model learned adaptive planning.

The possible contribution is the empirical decomposition: source-ID redirection, replicated
late-batch correspondence repair, question-sensitive root learning, and the measured failure of
local improvements to compose. The second captured SFT corpus and Qwen3-8B component check reduce
two uncertainties; the subsequent Mistral check extends the direction to another family.
The fresh-root and nested-batch studies now add newly selected evaluation inputs.
A stronger publication claim still needs a different task, further replications,
and a downstream use test of the improved correspondence interface.

## Technical appendix: definitions, provenance, and exact sources

### Metric definitions

`Correct` means the final scalar equals the frozen host oracle. `Performed` means manual review found
that the actual sampled program, linked child observations, and final state faithfully executed the
requested operator, scope, and threshold. `Correct and performed` requires both. An authenticated
empty final is an observed policy failure, not a missing value. `NULL` is reserved for unavailable or
unauthenticated evidence under the frozen study rule.

For row correspondence, “late” means output positions 17–48. One reference-specific late cell has
16 × 32 = 512 labels; one pooled four-bar condition has 3 reference conditions × 16 contexts × 32 =
1,536. The four pooled counts are 577, 634, 626, and 1,340. The interaction is 657/1,536, not a
comparison over 384 labels. Context, not label, is the clustered unit.

### Root-training details

The root continuation used rank-8 LoRA, learning rate 0.0001, fresh Adam state, and six fixed updates.
Action weights were 0.45 acquisition, 0.50 reduction, and 0.05 final. The 72 trajectories contained
216 root actions and 16,904 target tokens per pass; six passes exposed 101,424 target tokens. The
base root model and child checkpoint were fixed. Training loss fell from 0.1893 to 0.04385, but the
behavioral and manual-program endpoints—not loss—support the result.

The S2 replication used the same fixed24 starting adapter and optimization recipe with a separately
captured 72-trajectory corpus and fresh Adam state. Its two missing endpoints were physically
attempted and retained as NULL rather than scored as failures or retried.

### Exact local evidence pins

All paths below are under `/project/alex_phd/runs/rlm-research-r4/`.

| Evidence | Path | SHA-256 |
|---|---|---|
| Question-sensitive root report | `analyses/root-question-sensitive-sft-live-2026-09-10/REPORT_MAIN.md` | `6245e9559d0038d6284433f2ecd9ccb2c17e72d0dabd4be5d3b358b65da8ad4c` |
| Question-sensitive final seal | `analyses/root-question-sensitive-sft-live-2026-09-10/FINAL_MAIN.json` | `442e0cf9112ab647964ae03f239943654e1fd9648608fb13b27c8ed0c09a13c3` |
| S1: changed-names-and-numbers readout | `analyses/root-question-sensitive-readout-followups-2026-09-10/metadata-semantics/REPORT.md` | `cc77b86e0f0be15b848138cf62576ff5a6e41848c7cebb4292548fe6e54b786d` |
| S2: separate-corpus SFT report | `analyses/root-question-sensitive-sft-new-corpus-live-2026-09-10/REPORT_RECOVERY_V2.md` | `c7e41fadecdb6a46c71ef7b40878058de2cd4f2f2e999babf518c79baefe5a43` |
| S2: starting-policy and claim erratum | `analyses/root-question-sensitive-sft-new-corpus-live-2026-09-10/ERRATUM_START_AND_CLAIM.md` | `7974e7c0ff50ae3ad4dc357b05765445404f9144ff8112e335d8721cb91ba27d` |
| S2: comparison figure data | `analyses/root-question-sensitive-sft-new-corpus-live-2026-09-10/FIGURE_DATA_FAITHFUL_CORPUS_REPLICATION.json` | `9d040c85d29a2d5f4886275def234baf9c3cbf4f5ad49eeafdead8869f426058` |
| S2: erratum and figure seal | `analyses/root-question-sensitive-sft-new-corpus-live-2026-09-10/ERRATUM_FIGURE_SEAL.json` | `8dd41c3188b67c74a11d475c0e02ddf0e0aa1b9368c75b5e8d12bd9fb411f557` |
| S3: fresh-input three-policy report | `analyses/root-question-sensitive-fresh-input-three-policy-live-2026-09-11/REPORT.md` | `df5db6c9fe798bebe9e4606a3d6ee7ba273d622078e7b8f1351febeb501547e8` |
| S3: complete semantic audit | `analyses/root-question-sensitive-fresh-input-three-policy-live-2026-09-11/SEMANTICS_ALL.json` | `8d7f0466d423b23a780585f673709af4682dc8da7708e7c485f1f7593aec78ea` |
| S3: final audit seal | `analyses/root-question-sensitive-fresh-input-three-policy-live-2026-09-11/FINAL_SEAL.json` | `9512db664dbe6fdf578ab477aebf8f33ab1808f8bfec1692d8704ffe71c2ae78` |
| R1: reward-training continuation | `analyses/root-composed-rl-continuation-live-2026-09-10/REPORT.md` | `c7317d3f14a7fd6b5c32e735548fc72da85a63257c95aa22c703e9961df8b4d1` |
| R1: final audit seal | `analyses/root-composed-rl-continuation-live-2026-09-10/FINAL_SEAL.json` | `ef8b82b4fc4d1df2865398987673325e82e2cbab572101d05da55ab9c8dafe12` |
| R2: smaller-update report | `analyses/root-question-sensitive-terminal-rlvr-lr1e5-live-2026-09-10/REPORT.md` | `d6901670d6889b9469ba17ab51f39de19166bf013e96a620415a0e0ba9a17df4` |
| R2: final audit seal | `analyses/root-question-sensitive-terminal-rlvr-lr1e5-live-2026-09-10/FINAL_SEAL.json` | `fd1750f6db9c488b5808bda3e395b69d81b23514bfa240785e68c780d6f350d9` |
| H5: Mistral report | `analyses/leaf-mnli-stable-anchor-mistral7b-live-2026-09-11/REPORT.md` | `d5c48a217eb3f292397fa2d337fb2cc97f005d0243b0109dec0ccbcf8f437470` |
| H5: final audit seal | `analyses/leaf-mnli-stable-anchor-mistral7b-live-2026-09-11/FINAL_SEAL.json` | `72bdbde3d6657fe59aff0691c1f5def4c6347d783d116661e1ecd27e6b495c8e` |
| H5: interpretation qualification | `analyses/leaf-mnli-stable-anchor-mistral7b-live-2026-09-11/INTERPRETATION_ERRATUM.md` | `ff598a168efa1f88f85d1b5e606ddbb91562afe600dcd00c739aef090b96d8df` |
| H6: balanced tag-reuse report | `analyses/leaf-mnli-balanced-tag-match-live-2026-09-11/REPORT_ATTEMPT002.md` | `c245e4f0f14b194889837c3cc8d5ccfd313d40d2d6e48f94362c635858cbdabf` |
| H6: final audit seal | `analyses/leaf-mnli-balanced-tag-match-live-2026-09-11/FINAL_SEAL_ATTEMPT002.json` | `f7e8954d73dbc00f8e647bbd05968107546df965ecb002b6bbad230b013bb3bf` |
| H6: plotted counts | `analyses/leaf-mnli-balanced-tag-match-live-2026-09-11/FIGURE_DATA_ATTEMPT002.json` | `ae408560e168193da30f0dde7ebe5cdfde16f79dbc6b42d4a31dce674760c712` |
| H8: nested batch-size report | `analyses/leaf-mnli-nested-batch-matching-live-2026-09-11/REPORT.md` | `300f66b7ceac1ed3e89a32c2f44008eb201ca958f5b8b2ba00bd9927068c64ef` |
| H8: native audit | `analyses/leaf-mnli-nested-batch-matching-live-2026-09-11/NATIVE_AUDIT.json` | `0d14d2abb7631641d403a7cbfdfbe790914d702dfc0651d3f4a55885dcdb6155` |
| H8: final audit seal | `analyses/leaf-mnli-nested-batch-matching-live-2026-09-11/FINAL.json` | `87e7552988bb16ddb6588b6a814e1f0b3c8c84061f39db50ef12f2f9d7c8ccbb` |
| Zero/nonzero strata | `analyses/controller-zero-support-strata-2026-09-10/REPORT.md` | `d550dd2df5fa13189e14168969fecec4e2851e6bf4ffe8b781a3dd826b7e6507` |
| Related-readout erratum | `analyses/controller-zero-support-strata-2026-09-10/ERRATUM.md` | `99fa227e2a0d8bdd686353e76811ac39ccf272047dd75c0d0979fd235a817241` |
| Fresh row-correspondence report | `analyses/leaf-mnli-positional-anchor-new-context-live-2026-09-10/REPORT.md` | `2e0599eb47f2e8bbb73e8ddf7c504763798d089b5e2f507a35696ad1bf9684e5` |
| Late-denominator correction | `analyses/leaf-mnli-positional-anchor-new-context-live-2026-09-10/ERRATUM.md` | `a10f76116f7d4a2787a4607d85d99ecc34074cd7a74dc3f30d936d035f51b45b` |
| H3: arbitrary matching-key report | `analyses/leaf-mnli-stable-anchor-vs-sequence-counting-live-2026-09-10/REPORT.md` | `fa473b7670731a121579d25563f025e684cd7efc5ba332f9221e403b33dea96e` |
| H3: plotted counts | `analyses/leaf-mnli-stable-anchor-vs-sequence-counting-live-2026-09-10/FIGURE_DATA.json` | `47ce60e9ae81f5fc3d81830aaad096ed64213edb07ad55423c453f02811425da` |
| H3: final audit seal | `analyses/leaf-mnli-stable-anchor-vs-sequence-counting-live-2026-09-10/FINAL_SEAL.json` | `f3923f17e472b4ca1051e2448c7175de21f844832c5a822a323b980730c99430` |
| H4: Qwen3-8B matching-key report | `analyses/leaf-mnli-stable-anchor-qwen8b-live-2026-09-10/REPORT.md` | `e7ba139e12ef0aa0f490380085f1d31a2488172e6599a78536dbd0ccec5caa68` |
| H4: Qwen3-8B native audit | `analyses/leaf-mnli-stable-anchor-qwen8b-live-2026-09-10/AUDIT.json` | `7871f8b4ea44dc207d39da607d05804ccbe037450380415d901fbceb5628535d` |
| H4: final audit seal | `analyses/leaf-mnli-stable-anchor-qwen8b-live-2026-09-10/FINAL_SEAL.json` | `b619cb1d4aa79bfa7aa0d292a9e909d640c1efa421b0c0708bd2386bfed0278e` |
| Full/compact child continuation | `analyses/trec-child-interface-sft-followup-live-2026-09-10/REPORT.md` | `04a0a39698565e48d9d4dcc6c69958a745328d679f78d4f0e8c2bad200448a86` |
| Mixed child continuation | `analyses/trec-child-interface-mixed-sft-followup-live-2026-09-10/REPORT.md` | `c66201d836918caab6e726ef6976e08e59d5dce5e160bf3e95e0d7dfbe5eca6d` |
| Supplied-plan child ceiling | `analyses/root-lambda-supplied-plan-ceiling-live-2026-09-10/REPORT.md` | `4ccb7a60082d0bc76e8de88f5500514efca7ba61cf4afd42931a6c7ec04f892e` |
| Mixed-child downstream test | `analyses/root-lambda-supplied-plan-mixed-child-live-2026-09-10/REPORT.md` | `fd757fb7f889cc20a4b8c5521cc91f02585c293cd7da7d477829983edd8662eb` |
| Confidence recheck | `analyses/root-supplied-plan-selective-recheck-live-2026-09-10/REPORT.md` | `8e7d00d18f2953f8cb8a5d7af24d3fe1469d8e1c4e2767461873082c569758e3` |
| Task-aware/agreement recheck | `analyses/root-task-aware-selective-recheck-live-2026-09-10/REPORT.md` | `f9a8dec737268348ec299ed449e4c30fd2d48c537c4370052bc7687951d8e50b` |
| Direct-statistics report | `analyses/root-j1-sufficient-statistics-live-2026-09-10/REPORT.md` | `2f4066b9d1e8e3c753c206b8e42b0c0cb3aa4216c2b24af84aa44e47ea2801b2` |
| Direct-statistics native audit | `analyses/root-j1-sufficient-statistics-live-2026-09-10/AUDIT.json` | `a9a4bf556dd7ba5c8a8da8a8495122ba99798d0d04d1876f8ed60479a76e8fc8` |
| Prior mechanism review | `ideas/2026-09-08-batch-prompting-mechanism-controls.md` | `584a300a2b744f45e2ce694c87417dc9242454566ee2a50f21006b2e8d31a5c9` |
| Runtime/prior-art comparison | `ideas/2026-09-09-official-runtime-comparison.md` | `903b2e4923c7f28e84a16fb5f4119c8c8d7142c73c561d2d5d2882fee34f81a5` |

### Reproducibility boundaries

Native audits checked request identity, model binding, token IDs, decoded content, finish reasons,
usage, and released-owner/parent receipts. Manual root semantics were judged from the actual sampled
program plus authenticated parent and child observations, without executing generated code. Oracle
labels were used only for diagnostics and final scoring. Missing endpoints remained NULL and were not
silently converted to wrong answers or repaired from gold.

These safeguards support the reported local effects. They do not turn related seeds into independent
replications, make exposed source data novel, or make structured-decoder results representative of
free-form generation.
