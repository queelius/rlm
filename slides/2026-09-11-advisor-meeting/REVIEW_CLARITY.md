# Clarity and evidence review

## Current review — incomplete attempts and an explicit format comparison

Slide 4 replaces “unknown” with the operational facts: eight attempts timed out and one
additional attempt lacked a verifiable final answer. All attempts remain in the denominator;
none of these cases counts as a confirmed success. Slide 5 now shows three text–claim pairs,
with each pair on its own table row. Two share a text and the third uses a different text and
claim. The slide explains matching by order versus name beneath the table. This replaces a
single-text illustration that incorrectly suggested applying fixed questions to every text.
Speaker notes and the guide explain the same comparison without changing the results.

## Previous review — explaining SFT before presenting its results

The presenter needs to learn from the slides, not reconstruct unexplained shorthand.
The current deck therefore has nine main pages: page 3 shows what the model sees and the
code it learns to produce, and page 4 explains what improved. The miniature example is
explicitly illustrative, with its connection to a real record in `sft-worked-example.md`.
The remaining pages and private notes clarify comparisons, define reward training, and
explain unknown outcomes with full sentences. Results and proposals remain distinct;
the numerical evidence has not changed. The extra page replaces compression, not scope.

## Previous review — 11 September, after presenter feedback

The presenter could not tell what “a place” meant on page 2. That exposed a
larger problem: the earlier review assumed background knowledge the audience
does not have. This pass reviewed **all eight main pages and six backups** for
the task being asked, the meaning of terms, the comparison shown, and the claim
the evidence supports. Separate read-only reviews covered the main results
and backups; MAIN integrated the revisions and inspected every rendered page.

| Page | Clarification now visible on the slide |
|---|---|
| 1 | RLM is expanded; a handoff is the exchange of work and answers. Training and matching are separate studies. |
| 2 | The task is to count questions asking for a location. Three invented rows show Yes/No decisions and the final Ada/Ben counts. The diagram identifies the main model, helpers, and Python's roles. |
| 3 | Worked examples teach classification followed by calculation. Success requires both the answer and the requested calculation; the two trained copies use different example sets. |
| 4 | The helper study is a separate task: judging statements about a short text. The bike example explains both supported and contradicted. |
| 5 | Two language models judge statements. The horizontal axis counts statements per request; the vertical axis measures correct judgments. |
| 6 | Every method judges the same 768 statements. Time covers the entire workload, not one request. Repeated input text is the competing cost. |
| 7 | A proposed program component groups inputs, sends requests, and matches replies. The test concerns the final calculation, not merely helper judgments. |
| 8 | Observed component results are separated from the proposed complete-solution test. The audience is invited to help choose a convincing task. |
| 9 / B1 | Assigned test points replace unexplained weights; questions ask for locations. The slide shows exactly which points count and why the answer is one user. |
| 10 / B2 | Both versions supply names and preserve answer order. The comparison is whether statement and answer use the same name. The later-answer metric is identified. |
| 11 / B3 | Reward training is explained. The extra rule checks calculation agreement; practice attempts, test questions, and missing results are distinguished. |
| 12 / B4 | Location and number questions have concrete examples. The model selects users using one question and counts their other questions. The plan was supplied, not invented. |
| 13 / B5 | The confusing “answer box” metaphor is replaced by an explicit conflict: answer row 7, but label the reply row8. |
| 14 / B6 | Row numbers and arbitrary names have examples. Larger chart labels use the same request/judgment language as the main plot. |

The public pages use **matching names** consistently, rather than cycling through
tags, keys, IDs, and labels. Technical equivalents remain available in the guide.
The guide also distinguishes classifying a question from answering it: “How many
people live in Oslo?” mentions a place but asks for a number. This is the kind of
explanation a presenter should not have to reconstruct from shorthand.

No experimental counts, scores, source pins, or evidence cutoff changed. The two
changed fields in the portable data contain display labels only. Initial builds
exposed overflow introduced by the added explanations. Shorter text, less excess
spacing, and removal of repetition fixed it without reducing the body font.
The main talk remains eight pages. See [VERIFICATION.md](VERIFICATION.md) for the
final PDF and actual pdfpc checks; clarity remains subject to presenter feedback,
not something a successful compiler can establish.

## Historical review of the first twelve-slide draft (superseded)

Historical review of the first twelve-slide draft. The issues below were recorded
before revision; they are not the status of the current eight-main-page package.
See [the verification record](VERIFICATION.md) for the changes and completed checks.

Reviewed files: `research-update.tex`, `figures.py`, and `data/claims.json`, checked against
`evidence-and-methods.md` and its pinned source reports. This review is intentionally about what a
first-time presenter or a somewhat lay advisor could misunderstand. It does not recommend cosmetic
polish for its own sake.

## Bottom line

The deck has a good discussion-first arc: explain the system, show that root behavior can be taught,
show a concrete correspondence failure and repair, then show that local label gains do not reliably
compose. The three plotted datasets are numerically consistent with their pinned sources.

Before presentation, two issues are blockers: eight frames report vertical overflow that can hide
content, and the reward-training slide treats a result that is currently outside the adopted
evidence cutoff as completed evidence.
The next most important revision is to make the root-training chart's status unmistakable: it is a
metadata robustness readout of one trained checkpoint, not the primary protected result or an
independent training replication.

## Priority 0: must fix before using the deck

### 1. The deck compiles, but eight frames overflow vertically

The refreshed TeX correctly uses `\\` line breaks and produces a 12-page PDF. The log nevertheless
reports `Overfull \\vbox` on frames ending at source lines 69, 91, 116, 126, 160, 170, 207, and 239,
with overflow as large as 41.14 points. These are mainly content-heavy frames with bottom notes, so
evidence qualifiers or citations may be clipped below the visible page. Inspect all 12 rendered
pages and shorten or move notes rather than shrinking already-small evidence text. A successful PDF
exit is not enough when the log says frame content exceeds the page.

### 2. The reward-training slide is not supported by the current portable evidence

Slide 9 gives concrete 55/72 and 47/72 results under the headline “This reward-training recipe
did not improve the trained model.”
Those values do not appear in `data/claims.json`, have no source path or hash there, and the requested
evidence policy leaves the latest six-update RL result pending unless a final audit is explicitly
adopted. The placeholder `[Evidence R1]` has no key in the deck or portable claims file.

Until the final native and semantic audit is adopted, remove the numeric result or label the entire
slide “pending audit” without presenting the values as findings. The refreshed deck correctly says
**“this reward-training recipe,”** not “our first reward-training recipe”; retain that wording because
there is earlier reward-training history. After adoption, an even safer headline would be: “This
reward-training recipe did not improve this checkpoint on this panel.”

An evidence-strong alternative is to use this slot for the already adopted negative result: repeated
same-child checks repaired many individual labels but rarely repaired exact aggregate answers, and
direct sufficient-statistic requests were substantially worse. That negative finding advances the
main argument without relying on a pending run.

## Priority 1: likely to cause a substantive misunderstanding

### 3. The training slide shows the robustness readout, not the headline protected result

The chart on slide 5 displays 12/72 before and 50/72 after. Those are correct-and-performed counts
from the **changed-metadata readout**: the same trained checkpoint and same eight source contexts,
with names, weights, and thresholds changed. The strongest original protected result is 15/72 to
52/72. The slide title accurately mentions changed names and numbers, but the surrounding narrative
can easily make 12→50 sound like the primary training evaluation or a new-context replication. It
is not an independent training replication.

Choose one of two clear presentations:

- Lead with the primary result, 15→52, and state that a related changed-metadata readout was 12→50.
- Keep the 12→50 chart, but title or subtitle it “Same trained checkpoint, changed metadata” and say
  aloud that the original protected panel was 15→52.

In either case, explicitly state: **one training run, one checkpoint, related robustness readouts—not
independent training replications.** A true replication would repeat training with a new training
seed and newly held-out source contexts. The current small note says “one training run” and “related”
but asks the audience to infer too much from small text.

### 4. The invented example is labeled, but the actual training source remains too implicit

Slide 3 correctly says its Ada/Ben table is a simplified illustration and not a quoted training
example. However, slides 4 and 5 immediately say “we trained the model on examples of actions”
without telling the audience what the real records were. A presenter who has not internalized the
research may accidentally imply that the model was trained on the displayed place-question table.

Add one plain sentence to slide 4 or its spoken script:

> “The actual trajectories used public TREC question texts with six answer-type labels, plus
> synthetic record IDs, users, and weights; this table only illustrates the calculation.”

Also clarify the actual three-action trace: the root loads records and requests real child labels,
Python computes over the returned—possibly wrong—map, and the root returns the observed scalar. The
important method detail is that training preserved child mistakes rather than showing oracle labels.
That fact makes the root-learning result more credible and connects naturally to the later child
fragility slide.

### 5. “Before” and “after” refer to different training interventions on slides 5 and 8

Slide 5 is root-adapter training. Slide 8 is extra **child** mixed-interface training under an
experimenter-supplied plan. Both figures label bars simply “Before” and “After.” A lay viewer can
reasonably assume slide 8 shows the downstream effect of the root training from slide 5. It does not.

Rename the slide 8 labels or state prominently:

- “Original child” and “mixed-interface child,” and
- “The root plan and Python reducer were supplied and fixed.”

The current footnote contains the second point, but it needs to be in the main explanation. The
critical result is not merely “labels improved but answers did not.” It is that a separately trained
child gained 19/1,280 labels while exact downstream answers moved only 0/8→1/8 and total absolute
error worsened 117→130.

### 6. The deck omits the adopted repair and direct-statistics failures that explain the next step

The deck moves from downstream fragility to a pending RL result. That leaves an advisor without the
strongest evidence for why “just check the labels again” and “ask for a shorter answer” are not the
current solution:

- Confidence selection found 107/151 first-pass errors and produced a net +40 repaired labels,
  versus −9 for uniform selection, but repaired only 1/8 exact aggregate answers in that study.
- Two same-child samples agreed on many wrong labels; agreement abstention was not a reliable
  verifier.
- Direct per-user sufficient statistics were available and schema-valid in all 40 calls but scored
  0/8 exact, with total absolute error 1,805 versus 117 for the full-label control.

These negatives should not displace the positive findings. One compact slide or a spoken bridge
after slide 8 is enough: “We tried selective rechecking and direct summaries; both improved or
simplified local evidence without reliably fixing the final answer.” This makes the proposed move to
a different verifier or source-aware downstream interface intelligible rather than arbitrary.

### 7. The row-matching figure is correct, but the causal contrast is not yet teachable from the slide

The four bars are numerically right:

| Condition | Correct late labels |
|---|---:|
| Neither side numbered | 577/1,536 |
| Answers numbered only | 634/1,536 |
| Inputs numbered only | 626/1,536 |
| Both sides matched | 1,340/1,536 |

The frozen interaction is `(1,340−626)−(634−577)=657/1,536=42.77` percentage points. Each pooled bar
contains three reference conditions × 16 contexts × 32 late positions. Each single-reference late
cell is 512 labels, not 384.

The slide currently says only that “the same numbering on inputs and answers helped much more.” A
presenter may describe 87.2%−37.6%=49.6 points, which is not the prospectively primary interaction.
Put the 42.77-point interaction in the spoken note or main annotation and explain it as: “the gain
from numbering answers became 42.77 points larger when matching numbers were also visible on the
inputs.” Preserve the context as the experimental unit; do not call 1,536 independent examples.

### 8. The wrong-record illustration should show two complete records, not one shared passage

Slide 6 uses one passage and two statements. The actual intervention associates an output ID with a
different complete record, where each record contains its own premise and hypothesis. The current
example can make the advisor think the mechanism is only confusion among two hypotheses about one
passage.

Use two compact record cards instead:

- Record 7: premise “A dog is running”; hypothesis “An animal is moving”; label entailment.
- Record 8: premise “The ground is wet”; hypothesis “The ground is dry”; label contradiction.

Then say that record 7's answer slot is stamped with record 8's ID. “Contradiction” is locally right
for record 8 but globally wrong for record 7. The existing 298/768 versus 618/768 numbers are correct;
phrase the arm as “ID points to another visible record,” not merely “misleading ID.”

### 9. Evidence tags are not resolvable by the presenter or audience

The deck uses `[Evidence S1]`, `[Evidence H1]`, `[Evidence H2]`, `[Evidence C1]`, and `[Evidence R1]`,
but neither the TeX nor `claims.json` defines this key. The companion methods document gives full
paths but does not map these short labels. Either add a one-page evidence key to the speaker material
or remove the bracket tags from the visible deck. Most importantly, do not retain R1 until its audit
and source hash exist.

### 10. The sequence of the two proposed correspondence tests sounds contradictory

Slide 10 says to test arbitrary keys and then downstream use. Slide 12 says the current recommendation
is to connect row matching directly to final-task improvement. Those are both reasonable, but the
order is unclear.

State the decision tree explicitly: first run the already prepared stable-key discriminator because
it tells us whether the reusable interface is an arbitrary key or only an ordinal row; then put the
winning representation into one fixed downstream root pipeline. If opaque keys fail, the useful
claim narrows to sequence position rather than general correspondence keys.

## Priority 2: method and wording safeguards

### 11. The portable validation has two avoidable provenance gaps

`figures.py` checks a source hash only when `path.exists()` is true. A missing evidence file is
silently skipped, so a successful figure build does not prove that every declared source was checked.
It should fail if any declared source is absent and assert the expected number of verified sources.

The training figure also hard-codes its bar labels as `12 / 72` and `50 / 72` instead of deriving
them from `s["correct"]` and `s["planned"]`. The bar heights come from JSON, so later data changes
could produce a visually inconsistent label without failing validation. This is a reproducibility
risk, not a visual preference.

### 12. Manual semantic review should be explained once in ordinary language

“Verified success” on the training slide is not just automatic exact match. Correct-and-performed
requires manual review of the actual sampled code, parent-linked child observations, successful
execution state, and final use of the computed scalar. No analyst re-executed generated code. Say
this once, briefly, because it is a substantive strength and prevents the audience from assuming
that “performed” was inferred from a keyword or from the final number.

### 13. Keep the recursive capability claim conditional

Slide 2 says a helper “can divide its own task again,” while the note says current tests mostly use
one helper layer. The note is honest, but in speech distinguish framework capability from measured
behavior: “The architecture permits recursion; today's evidence is mostly one-level delegation.”
Likewise, the final learning-loop slide is properly labeled a research direction and should remain
so.

### 14. Replace transient queue language with evidence status

“New-example training is running” and “the matching-key comparison is being prepared” can become
false before the meeting. The current evidence policy is more durable: both remain pending until an
adopted final audit exists. Say “pending audited result” unless the deck is regenerated from a newer
cutoff. This applies especially to the new-corpus SFT and latest six-update RL continuation.

### 15. The novelty boundary deserves one direct prior-art sentence in the spoken deck

The deck correctly says numbered batch prompts already exist, but the advisor cannot see what is
already known. Cite at least Batch Prompting for indexed input/output correspondence and BatchPrompt
for order/permutation sensitivity. If discussing constrained schemas, mention that Grammar-Aligned
Decoding shows grammar masking can alter distributions. The proposed contribution is the controlled
behavioral decomposition and downstream boundary—not inventing indexes, JSON, semantic operators,
or recursive calls.

## Numerical cross-check receipt

- Metadata robustness chart: 12/72→50/72 correct-and-performed, with 17→0 NULLs. Correct for the
  pinned metadata report; it is not the original protected 15/72→52/72 result.
- Wrong-visible-ID statement: 298/768 versus unrelated 618/768, approximately 38.80% versus 80.47%.
  Correct.
- Row-matching bars: 577, 634, 626, and 1,340 of 1,536. Correct. Interaction 657/1,536 = 42.77
  points. Correct. Late denominator per reference cell is 512.
- Composition chart: 1,129/1,280→1,148/1,280 child labels; exact answers 0/8→1/8; oracle-label
  diagnostic 8/8; total absolute error 117→130. Correct.
- Reward-training slide: 55/72→47/72 is not represented or pinned in the current claims file and must
  remain pending under the requested evidence cutoff.

## Recommended minimal revision order

1. Inspect and resolve all eight overflowing frames in the compiled PDF.
2. Remove or clearly quarantine the pending reward result; retain “this recipe,” not “first recipe.”
3. Make slide 5 explicitly distinguish the 15→52 primary result from the related 12→50 metadata
   readout and say “one checkpoint, not a training replication.”
4. Make slide 8 unmistakably about child training under a supplied plan.
5. Add one compact adopted-negative bridge covering repair and direct statistics.
6. Clarify the actual TREC-plus-synthetic-metadata training source and the complete-record ID example.
7. Rebuild, inspect every slide, and make evidence tags resolvable.
