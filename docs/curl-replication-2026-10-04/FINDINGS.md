---
date: 2026-10-04
cutoff_utc: 2026-10-05T17:18:00Z
original_100k_cohort_cutoff_utc: 2026-10-04T18:37:00Z
execution_update_utc: 2026-10-05T17:18:00Z
stage: exploratory_compatibility_reproduction
question: Does contrastive learning improve pixel-based control beyond random-crop augmentation?
primary_endpoint: 100000_training_simulator_steps
continuation_endpoint: 500000_retained_training_simulator_steps
external_runs: /project/alex_phd/runs/curl-replication-20261004
---

# What we have learned so far

All three cartpole pairs are complete at both budgets. CURL finished higher in every
100k pair. **The mean advantage shrank from +184.64 at 100k to +5.22 at 500k**,
with different late winners. This small study shows an early benefit but no
consistent late winner. It does not establish equivalence or a universal rule.

## First seed: smaller updates do not remove wrong-target harm, October 5 at 17:18 UTC

Both new models for the first training seed are complete. Each learned to
control the cartpole from pictures for the same 100k simulator steps. Scores
are total test rewards averaged over ten fixed starting states, not percentages.

| Image-matching exercise, seed 123 | Original update | Smaller update |
|---|---:|---:|
| Match two views of the same picture | 678.02 | 795.84 |
| Deliberately match views from different records | 0.11 | 138.08 |

Without the image-matching exercise, this seed scored **454.47**. All conditions
still use random image crops. We reduce only the matching exercise's encoder
update, not the reward-learning optimizer's update.

The smaller update raised the correct-matching score by **117.81** and the
wrong-matching score by **137.96**. Wrong matching nevertheless remained
**316.40 points below doing no matching**. The correct-minus-wrong gap changed
from 677.91 to 657.76, a reduction of only 20.15 points in this seed.

**Interpretation:** this first pair does not support the simple explanation
that the original wrong-target result was only caused by its stronger update.
The weaker wrong-target model still learned much less than the no-matching
model. Nor is this evidence of a reliable interaction: both new conditions
improved by similar amounts, and we have only one completed seed. Between-run
variation and different experiences collected during training remain plausible
explanations. The wrong-target curve is also unstable: it scored 166.94 at 80k,
0.95 at 88k and 138.08 at the fixed endpoint. We do not select a favorable
checkpoint or call that curve a stable recovery.

The new wrong-target run completed normally, without a failure or resume.
All 26 means were recomputed from 260 raw test episodes, with the declared
starts 10000--10009. All 920 logged learning values were finite; the run had
100 natural training episodes, 11,500 updates, and a verified 2,370,118,883-byte
checkpoint. Its 115 sampled pairing records had no fixed permutation positions;
duplicate replay sampling produced two residual same-record matches. These are
sampled diagnostics, not a complete count of all updates. Settings differ from
the original wrong-target run only in the declared arm and encoder-update rule.
Elapsed time was 14.47 minutes, including 6.16 minutes testing and 6.48 seconds
saving state. The [native evidence](encoder-strength-data/single_encoder_shuffled_curl-seed123)
is available unchanged, alongside the correct-matching counterpart.

**Decision:** finish the four remaining models for seeds 456 and 789. The
correct-matching seed-456 model is already training, and three others follow.
Keep this partial two-by-two comparison in supporting documents; do not replace
the PDF's completed-cohort findings or report an unequal-seed group average.

## First smaller-update result is encouraging, October 5 at 17:09 UTC

The first model with a smaller image-matching update has finished. The task
is still cartpole swing-up: learn from pictures how to move a cart so that
its pole swings up and stays upright. The score is total test reward, averaged
over ten fixed starting states, after 100k training simulator steps.

| Condition, training seed 123 | Fixed-endpoint score |
|---|---:|
| Correct image matches, original update | 678.02 |
| Correct image matches, smaller update | 795.84 |
| No image-matching exercise | 454.47 |

The new model finished **117.81 points above the original CURL model**. We
changed how strongly the extra matching exercise updates the image encoder,
not its targets or reward-learning optimizer. The new score comes from the
declared endpoint, not selection of a favorable point on its learning curve.

**What this does and does not tell us:** one smaller-update model learned well.
It does not yet show that this change reliably improves CURL. Training varies
between seeds, and matched seeds do not force models to have identical
experiences once their actions diverge. Nor does this tell us whether a smaller
update reduces the harm from incorrect picture matches. That counterpart is
still running. Finish all six declared new models before assessing the pattern;
do not average one completed new seed against all three original seeds.

Root recomputed all 26 means from 260 native test episodes, checked the ten
declared starts, all 920 logged learning values for finiteness, 100 natural
training episodes, 11,500 updates, and the completed 2,370,118,499-byte checkpoint.
No failure or restart occurred. Rechecking the three original seed-123
comparators also reproduced their reported curves and endpoints. The only
configuration differences against original CURL are the declared arm/update
rule and a shorter time cap that did not bind. Native elapsed time was
14.22 minutes, including 6.03 minutes of testing and 7.02 seconds saving state.

The four [native evidence files](encoder-strength-data/single_encoder_curl-seed123)
are published unchanged. The [protocol](ENCODER_UPDATE_CONTROL.md) explains the
intervention. The wrong-target counterpart is making actual learning updates,
and four later models are queued under the existing owner. No source, condition,
seed or endpoint was changed in response to this first result.

**Presentation decision:** keep this single-run observation in the supporting
documents. The PDF remains the complete-cohort account at its explicit 16:52 UTC
cutoff. Update its conclusions and figures when the strength comparison is ready
to interpret, rather than adding a headline from one favorable training run.

## Wrong matching hurts across all three training seeds, October 5 at 16:52 UTC

| Training seed | Correct matching | No matching exercise | Wrong matching |
|---|---:|---:|---:|
| 123 | 678.02 | 454.47 | 0.11 |
| 456 | 446.17 | 240.54 | 68.10 |
| 789 | 587.99 | 463.26 | 107.79 |
| Mean across three trained models | 570.73 | 386.09 | 58.67 |

These are all the declared fixed-100k endpoints, not the best points on each
curve. The score is total reward averaged over ten fixed test starts; it is
not a percentage. Every wrong-matching model scored below both corresponding
original models. Its mean difference is **-512.06 versus correct matching**
and **-327.42 versus no matching**. The sample standard deviations across
training seeds are 116.89, 126.13 and 54.45 for the three columns. These describe
variation between trained models, not confidence intervals. Testing one model
ten times does not create ten independent training runs.

The [complete curves and machine-readable report](figures/correspondence-three-seeds)
show all nine models. These are six original comparators and three later
wrong-target models, not nine new runs. Every method keeps random image crops.
The extra matching update is retained in the wrong-target condition, with the
candidate pictures reassigned before computing the original matching loss.

**What we learned:** incorrect teaching targets can harm learning in this
setting. The third model fluctuated from 85.24 initially through 187.13 at 96k
to 107.79 at the endpoint. It did not collapse to zero. These results do not
establish representation collapse, a universal effect on other tasks, or the
reason correct matching helped. A damaging update is not a neutral substitute
for one without useful information. Our no-matching comparison remains essential.

Root and an independent CPU audit verified native completion and settings.
The independent audit recomputed all 234 means from 2,340 raw test episodes,
checked the exact starts 10000--10009, finite learning values, natural episode
boundaries and checkpoint sizes for all nine models. Each trained for 100k
simulator steps and 11,500 updates, with 260k separate evaluation interactions.
No new failures or resumes. Original CURL seed 123 has a native completion
and a later queue skip-complete record, rather than a standalone result receipt.
All 12 [public wrong-matching evidence files](correspondence-data) are byte-exact
copies of their native records. Earlier partial reports below remain dated history.

The three wrong-target runs have 345 sampled pairing diagnostics. All have
zero fixed permutation positions. Duplicate replay sampling nevertheless
produced eight residual same-record matches among 44,160 logged row pairs;
the logs are not an exhaustive count of all training updates. Distinct records
may also show similar physical states.

Their total native time was **43.68 minutes**, including **18.29 minutes of
testing** and **19.19 seconds saving checkpoints**. Original three-run totals
were 43.43 minutes with correct matching and 39.93 without matching. These
comparisons match training interactions, not wall time or optimizer work.

**Next decision:** keep the six [smaller-encoder-update runs](ENCODER_UPDATE_CONTROL.md)
fixed and finish them. The first launched 0.875 seconds after the wrong-target
queue ended and returned real simulator rewards after 19.01 seconds. Actual
learning values are finite; no new fixed endpoint exists yet. The design was
chosen before the last two wrong-target endpoints, and these outcomes do not
change its seeds, conditions or stopping points. The updated thirteen-page
[learning guide](learning-guide.pdf) explains this result and the next question.

## Two completed wrong-matching runs are worse, October 5 at 16:36 UTC

| Training seed | Correct matching | No matching exercise | Wrong matching |
|---|---:|---:|---:|
| 123 | 678.02 | 454.47 | 0.11 |
| 456 | 446.17 | 240.54 | 68.10 |

These are fixed-100k test rewards, averaged over ten declared starting states
per model. Only two of the three new runs are complete. Both are worse than
their original comparators, but their outcomes differ: the second did not
finish near zero. Do not describe every wrong-matching model as collapsing,
or report a completed group average before the last model finishes.

Seed 456 completed normally with 11,500 updates and a successful 2.39 GB
checkpoint. Root recomputed all 26 means from 260 raw episodes and checked
all 920 logged learning values for finiteness. All 115 sampled permutations
had no fixed positions; three residual same-record matches reflect replay
duplicates. Its curve fluctuated substantially, reaching 171.68 at 48k and
ending at 68.10. The [native records](correspondence-data/shuffled_curl-seed456)
preserve those fluctuations. No failures or restarts occurred.

The final seed 789 is running and has returned actual rewards and learning
updates. The [six-run update-strength follow-up](ENCODER_UPDATE_CONTROL.md)
passed focused CPU tests and independent review. Its owner is now waiting
behind this final run, with a separate source snapshot and fixed resource
caps. Its design was fixed before seed 456 finished. No new follow-up model
has trained yet; a waiting process is not evidence of learning.

## One wrong-matching model fails to learn a useful test policy, October 5 at 16:28 UTC

For training seed 123, the fixed-100k scores are **678.02 with correct image
matching, 454.47 without the matching exercise, and 0.11 with wrong matches**.
The score is the mean total reward over the same ten declared test starts.
This is one completed new model, not a completed three-seed group. The other
two wrong-matching runs remain scheduled without changes or early stopping.

The low score is a scientific outcome, not a crashed run. Root and an independent
audit recomputed all 26 test means from 260 raw episodes. The run completed
100k training steps and 11,500 updates; all 920 logged learning values were
finite. Its successful receipt and 2.39 GB final checkpoint agree. There were
no failures or resumes. Total native duration was 14.57 minutes, including
6.12 minutes of evaluation and 6.20 seconds writing the final checkpoint.

The learning curve was poor and unstable, not a monotonic decline: it started
at 8.44, reached 106.10 at 60k and ended at 0.11. All ten final test rewards were
near zero. Stochastic training episodes sometimes scored higher than deterministic
test episodes. Different action selection and initial states can explain that
discrepancy; it does not by itself establish an evaluation bug. The late matching
loss was close to chance-level classification, but this does not prove that the
learned image representation collapsed.

All 115 sampled pairing diagnostics had no unchanged permutation positions.
Two residual same-record matches occurred because experience is sampled with
replacement. Do not claim that every training pair contained different records,
or that all different records were semantically unrelated.

**Interpretation:** wrong teaching targets can seriously harm learning in this
run. That is not yet an explanation of the benefit from correct matches. Finish
the three-seed comparison and retain the no-matching control. We are preparing
a [small encoder-update-strength comparison](ENCODER_UPDATE_CONTROL.md) to
separate the pairing question from sensitivity to update magnitude.

The [public native records](correspondence-data/shuffled_curl-seed123) are
byte-exact copies from `curl-cartpole-correspondence-20261005/shuffled_curl-seed123-100k-v1`
under the external run store; the [protocol](CORRESPONDENCE_CONTROL.md) records
the exact settings. The learning PDF remains at its explicit 16:04 UTC cutoff
until the three-seed comparison is complete; this dated note records the newer
single-model result without replacing it with a group claim.

## The additional walking cohort is complete, October 5 at 16:04 UTC

| New training seed | CURL | Same-crops control | CURL minus control |
|---|---:|---:|---:|
| 234 | 502.95151 | 200.42850 | +302.52301 |
| 567 | 254.63460 | 335.19614 | -80.56155 |
| 890 | 436.58392 | 156.21813 | +280.36579 |

CURL won two pairs and lost one. The new group's averages are **398.06 versus
230.61**, a **+167.44** mean difference favoring CURL. This is a more positive
group than the original, whose mean difference was +1.20. Keep both groups
visible. The follow-up adds evidence that the extra update can help under these
settings, but the pair-to-pair variation remains substantial. It does not
establish a universal benefit, equivalence, or the cause of the differences.

The six-pair descriptive mean difference would be +84.32, with four CURL wins
and two control wins. That is not a new confirmatory estimate: we added the
second cohort after inspecting the first. Our main presentation therefore
reports the original and additional groups separately, with every pair shown.
The additional paired differences have sample standard deviation 215.06,
which describes variation across three training pairs, not standard error or
variation across the 30 final test episodes.

The final control learned from 17.31 initially to 156.22. Its intermediate score
reached 177.29 at 88k and then fluctuated. We retain the declared 100k endpoint.
This is limited learning at this budget, not a software failure or proof that
it could never catch up with more training.

Root and an independent CPU audit verified all 156 means from 1,560 exact-start
episodes, successful natural endpoints, finite learning values and checkpoint
sizes. Every run used 100k training steps, 49k updates, 100 natural training
episodes and 260k separate test steps. No failures or restarts. Within-pair
settings differ only by arm; the original cohort differs only by training seed.
All 24 [public native records](walker-additional-data) match their originals.
The [complete-group curves and summary](figures/walker-additional-three-pairs)
preserve all models; the earlier two-pair figure remains a historical snapshot.

Total native start-to-end time was 435.33 minutes (7.26 hours), including
211.72 minutes (3.53 hours) of testing and 7.43 minutes of checkpoint writes.
The next cartpole condition launched 0.081 seconds after the walking queue
ended. Real test returns arrived after 19.01 seconds; learning updates and
zero-fixed-position pairing diagnostics followed. Two further seeds are queued.
This is a running experiment, not evidence yet that wrong matching helps or hurts.

**Next decision:** finish the admitted [three-condition cartpole comparison](CORRESPONDENCE_CONTROL.md),
where the original early benefit was more consistent. Compare correct matching,
no matching, and wrong matching. Incorrect targets can actively harm learning;
their failure alone would not establish why correct matching helps. Update the
learning guide to show both walking cohorts and this narrower mechanism question.

## One more trained walker, but not yet another comparison, October 5 at 14:58 UTC

The third additional CURL model, seed890, completed its fixed 100k training
budget with an average test reward of **436.58392**. Its matched control is
still training. We therefore have five completed models in this six-model
follow-up, but still only two completed pairs. There is no third winner yet.

This model improved from 17.31 before training. Its last three recorded test
means were 404.20, 402.92 and 436.58. Improvement was uneven, and the final score
falls within the earlier CURL walkers' range. The trajectory supports actual
learning; it does not establish a benefit over the unfinished control.

Root and an independent CPU audit recomputed all 26 means from 260 raw test
episodes, each with the declared starts 10000--10009. The run completed 50k
decisions, 100k training steps, 49k learning updates and 100 natural training
episodes. All 3,920 logged learning values were finite. There were no failures
or resumes, and the final checkpoint size matched its saved receipt. Training
settings differ from its active control only by the matching update. The
[native records](walker-additional-data/curl-seed890) preserve the full curve.

Start-to-end wall time was 75.98 minutes, including 36.08 minutes of testing and
1.23 minutes of checkpoint writes. The 260k test interactions are not training
interactions or additional independently trained models.

The next [three-run cartpole comparison](CORRESPONDENCE_CONTROL.md) is admitted
and waiting behind the final walking control. It tests correct versus wrong
image matches, retaining the original no-matching control. The eleven-page
learning PDF remains at its clearly marked earlier cutoff: this unmatched
endpoint does not yet change its scientific conclusions. Revisit that edition
when the cohort or the new comparison is complete.

## Two additional walking pairs have opposite winners, October 5 at 13:39 UTC

| New training seed | CURL | Same-crops control | CURL minus control |
|---|---:|---:|---:|
| 234 | 502.95151 | 200.42850 | +302.52301 |
| 567 | 254.63460 | 335.19614 | -80.56155 |

This is a separately reported, unfinished follow-up cohort: two of its three
pairs are complete. The first favors CURL strongly; the second favors the
control. The original three pairs were also mixed. We cannot claim a dependable
benefit from this partial follow-up, and we will not stop after a favorable
subset. The last reference and its control retain the same settings and endpoint.

The curves matter as well as the endpoints. The first control dropped from
324.88 at 92k steps to 263.33 at 96k and 200.43 at 100k. We keep the declared
100k score rather than replace it with a better earlier checkpoint. That late
decline is a reason to interpret the size of this pair's difference cautiously,
not evidence that every control deteriorates: the second control finished
above its CURL counterpart and improved over its last three evaluations.

All four runs completed naturally without failures or restarts. Recomputed
104 means from 1,040 exact-start test episodes. Every run used 100k training
steps, 49k updates, 100 natural training episodes, and 260k separate evaluation
steps. Paired configurations differ only by learning arm; final checkpoint
sizes match receipts. Keep test episodes separate from training replicates.

[All four native records](walker-additional-data), the
[two-pair curves](figures/walker-additional-two-pairs/learning-curves.pdf), and
[machine-readable summary](figures/walker-additional-two-pairs/summary.json)
support this snapshot. The figure labels this as two completed pairs, not a
complete cohort. The third pair is running and is not assigned a missing-as-zero
score. This strengthens the motivation to study training variation, but does
not identify its cause or establish equivalence.

## First additional walking model complete, October 5 at 10:04 UTC

The first new CURL model (training seed 234) scored **502.95151** at the fixed
100,000-step endpoint, versus 23.14947 before training. Its early progress was
slow: it scored 80.42 at 48k steps before improving to 328.28 at 76k. This is a
completed learning run, but **not yet a comparison with its matched control**.
The control has started. Do not interpret this score as an extra CURL win.

All 26 means were recomputed from 260 test episodes on the declared ten starts.
The run completed naturally with 50,000 decisions, 100,000 training simulator
steps, 49,000 updates and 100 training episodes. All 3,920 logged learning values
were finite, with no failure or restart. The final checkpoint's 9,137,791,883-byte
size matches its receipt. Native start-to-end time was 76.26 minutes, including
36.07 minutes of evaluation and 73.90 seconds writing checkpoints. The separate
260k evaluation steps were not training experience.

[Native records](walker-additional-data/curl-seed234) preserve the endpoint,
full curve, configuration and successful owner receipt. Continue the matched
control and remaining two pairs unchanged. The PDF's completed-comparison
story does not change; this dated entry records new evidence while the first
pair is unfinished.

## The walking pattern persists on new test starts, October 5 at 08:48 UTC

We retested every original final walking model on 50 new starts, with no
learning or checkpoint selection. CURL-minus-control differences are
**+287.28, +23.01 and -240.92**, compared with **+241.55, +24.27 and -262.23**
originally. The same two CURL wins and one large control win remain. The
third pair scores 428.59 versus 669.51 on this larger panel.

The mixed result is therefore not explained away by replacing the original
ten starts. Variability between trained models remains an important question;
this panel does not identify its cause. More episodes do not create new
independent training runs, and this is not transfer to another task. The
original primary scores remain unchanged.

All 300 finite returns, exact starts, zero training/updates and successful
completion receipts were checked. Evaluation used 300k simulator steps and
44.44 minutes of evaluator elapsed time, including validation and setup.
Total batch wall time was 44.57 minutes; neither is pure rollout time. See the
[complete supplemental report](WALKER_FRESH_STARTS.md) and
[all six native records](walker-fresh-starts-data).

The [three additional training pairs](WALKER_ADDITIONAL_SEEDS.md), admitted
before any supplemental comparison was complete, have begun automatically.
The first new run returned its initial ten test episodes, then proceeds to
learning from fresh weights. It is not yet a completed training result.
Keep the cohort and its settings unchanged; do not tune from supplemental scores.

## The complete walking cohort changes the story, October 5 at 08:03 UTC

The third control finished at **698.06230**, compared with **435.82973** for
CURL. Its advantage was visible well before the final checkpoint. This is a
completed fixed-endpoint result, not a selected high point on its curve.

| Training pair | CURL | Same-crops control | CURL minus control |
|---|---:|---:|---:|
| Seed 123 | 482.83 | 241.28 | +241.55 |
| Seed 456 | 385.23 | 360.96 | +24.27 |
| Seed 789 | 435.83 | 698.06 | -262.23 |
| Mean across the three pairs | 434.63 | 433.44 | +1.20 |

The first two wins did not establish a dependable walking advantage. The third
pair nearly cancels their pooled benefit. The near-tied averages also do not
establish equivalence: outcomes vary greatly across just three training seeds.
The control's across-seed standard deviation is 236.86, versus 48.81 for CURL,
but three seeds are insufficient to claim that one method is generally more
reliable. This is a reason to measure more training runs, not defend the earlier
two-pair impression. Keep the historical snapshots below as the record of how
the evidence changed our view.

All six runs completed naturally with no failure or restart. The analysis and
an independent CPU audit recomputed 156 means from 1,560 raw test episodes,
checked exact start seeds and paired configurations, and confirmed 50,000
decisions, 100,000 training steps and 49,000 updates per run. Each model also
used 260,000 separate evaluation steps. Checkpoint sizes match their receipts.
The complete cohort used 438.85 minutes of native start-to-end time, including
213.57 minutes of evaluation and 7.54 minutes writing checkpoints. Equal
training experience is not equal wall time or optimization work.

The [six full curves](figures/walker-three-pairs/learning-curves.pdf),
[machine-readable summary](figures/walker-three-pairs/summary.json), and
[final control's records](walker-data/third-control) support this update.

**Next decision:** the already-declared [new-start panel](WALKER_FRESH_STARTS.md)
is running on all six unchanged policies. It checks sensitivity to test starts.
We also admitted [three new training-seed pairs](WALKER_ADDITIONAL_SEEDS.md)
with unchanged settings, queued after that panel. They check variability from
training new models. This choice follows the mixed original cohort, not
selection of favorable policies or schedules using the supplemental panel.
Report the new cohort separately; any pooled six-pair summary remains exploratory.

## Third walking CURL model complete, October 5 at 06:54 UTC

The third CURL model finished at **435.82973** after the fixed 100k training
steps, compared with 18.16 before training. Its score falls between the other
two CURL models, 482.83 and 385.23. Its matched control has started, so this is
not yet a third completed comparison or evidence of a third CURL advantage.

Root and an independent CPU audit recomputed all 26 test averages from the
260 recorded episodes. The run completed naturally at 50,000 decisions,
100,000 training simulator steps and 49,000 learning updates. All 3,920 logged
learning values were finite, with no failure or recovery. The final checkpoint
is 9,137,791,883 bytes, matching its receipt. Native start-to-end duration was
74.69 minutes, including 35.06 minutes of evaluation and 72.51 seconds of
checkpoint writes. The 260,000 evaluation steps were separate from training.

The control started 0.444 seconds after this process exited. It has returned
real episode rewards and finite learning updates; its initial score and all
configuration fields except the learning arm match the reference. Finish it
before interpreting the third difference. The early two-pair conclusion is
unchanged, and the PDF retains its explicit 05:43 completed-comparison cutoff.
The [third reference records](walker-data/third-reference) are now available.

The next check is separately declared: test all six final policies on
[50 fresh starting states](WALKER_FRESH_STARTS.md), without further training
or replacing the primary scores. Its evaluation tool has passed focused CPU
checks; a serial launcher is being prepared while the control trains. This
tests sensitivity to test starts, not new training seeds or why CURL works.

## Two walking pairs complete, October 5 at 05:43 UTC

At the fixed 100k endpoint, the second pair scored **385.23473 for CURL versus
360.96295 for the control**, a difference of **+24.27177**. The first difference
was +241.55020. Both signs favor CURL, but the benefit's size varies greatly.
The two-pair means are 434.03 versus 301.12, with a mean difference of +132.91;
the individual pairs are more informative than this average alone.

The second control nearly matched CURL at several late checkpoints, but its
curve fluctuates. This does not establish steady catch-up, uniform CURL
dominance, a persistent advantage or a mechanism. These are two training
pairs, not twenty independent models per arm or 1,040 training replicates.
The third pair must be completed regardless of these first two results.

Both root and an independent CPU audit recomputed all 104 curve means from
1,040 individual episode records, checked ten distinct declared starts per
point, and confirmed that configurations differ only in arm within each pair.
All four runs completed naturally at 50k decisions, 100k training steps and
49k ordinary updates, with no failures or recovery. The second control has
3,430 finite logged learning values and a final checkpoint of 9,105,904,477
bytes matching its receipt. It took 71.54 minutes, including 36.12 minutes of
evaluation and 1.27 minutes of checkpoint writes. Each run used another 260k
evaluation simulator steps; experience is matched, not compute or wall time.

The owner launched CURL seed 789, child 1704067, 0.277 seconds after the control
exited. First real returns arrived after 89.46 seconds and finite updates
after 106.25 seconds. The third control remains queued. Continue unchanged,
then decide between longer training and more seeds. CPU-only feasibility work
on a future continuation is preparation, not a changed or launched protocol.
The small second gap also motivates testing all six final models on a new
fixed panel of starts before expensive new training. This would supplement,
not replace, the original primary result or create new training replicates.
See [ranked follow-ups and feasibility](NEXT_COMPARISONS.md).

The guide now shows [both complete curves](figures/walker-two-pairs/learning-curves.pdf)
and explicitly emphasizes the unequal gaps. See the [paired summary](figures/walker-two-pairs/summary.json),
[second reference records](walker-data/second-reference) and
[second control records](walker-data/second-control). The first-pair figure is
preserved as a historical snapshot. Full weights remain external.

## Second walking CURL run complete, October 5 at 04:27 UTC

The second fresh CURL model scored **385.23473** at the declared 100k endpoint,
up from 20.83 before training. Its matched control has started, so this is not
yet another completed comparison. Do not compare it against the first seed's
control. The first pair remains 482.83 versus 241.28.

All 26 evaluation means were recomputed from their ten distinct declared
starts. The final mean is 385.23472867873534; no favorable checkpoint selection.
The run completed naturally: 50k decisions, 100k training simulator steps,
100 training episodes and 49k ordinary update calls. All 3,920 logged learning
values were finite, with no failure or recovery. Its final 9,137,791,883-byte
checkpoint matches its receipt. Native start-to-end time was 76.82 minutes,
including 36.32 minutes of evaluation and 1.27 minutes of checkpoint writes.
Evaluation used another 260k simulator steps, separately from training.

**Lesson:** an earlier lead does not guarantee a higher final score. At 56k,
the second CURL seed scored 285.04 versus 140.34 for the first. At 100k their
ordering reversed: 385.23 versus 482.83. This is variation between two runs of
the same method, not evidence for or against the image-matching update. Keep
the full curves and complete every declared pair.

The owner launched the matched control 0.408 seconds after the reference
process exited. Real returns arrived after 88.15 seconds and finite updates
after 105.03 seconds. Configurations differ only in the arm; both have the same
initial mean of 20.83289. Both third-seed jobs remain queued. No live source or
protocol change. Decide on longer training versus additional seeds only after
the cohort. The PDF still reports the first completed pair at its explicit
03:13 cutoff; this unpaired endpoint does not change its main conclusion.

Native evidence is in the external campaign
`/project/alex_phd/runs/curl-walker-replication-20261004/`:
`curl-seed456-100k-v1/result.json`, `curl-seed456-100k-v1/run/metrics.jsonl`,
the matched control's native records and `queue.jsonl`. The dated review is
`/project/alex_phd/runs/curl-replication-20261004/reviews/REVIEW-1791174299-850971b1.md`.
These external records and model checkpoints are not backed up by this Git push.

## First walking pair complete, October 5 at 03:13 UTC

At the fixed 100k endpoint, **CURL scored 482.83347 versus 241.28327 for the
same-crops control**, a difference of **+241.55020**. This is one fresh training
pair on walker/walk, not a three-seed result, a transfer of cartpole weights or
a novelty claim. Both configurations match except for the arm. Each score is
the mean of ten declared evaluation starts. All 52 points on the two complete
curves were recomputed from the individual episode records.

Both methods learned from their initial mean of 17.99. CURL learned earlier
in this pair. The control's last six means were 163.19, 200.75, 212.38, 229.18,
170.84 and 241.28. Its improvement makes a learning delay plausible, but neither
catch-up with more training nor a persistent gap is established. Seed variation
could also explain part of the difference. Ten evaluations of one policy are
not ten independent trained models.

The control completed 50k decisions, 100k training simulator steps, 100 natural
training episodes and 49k ordinary update calls, with exit 0 and complete=true.
All 3,430 logged learning values were finite; no recovery. The final checkpoint
is 9,105,904,477 bytes, matching its receipt, saved in 24.03 seconds.
Both arms used 100k training and 260k evaluation simulator steps. Native
start-to-end durations were 74.83 minutes for CURL and 69.40 for the control.
Control evaluation took 2,091.56 seconds and checkpoint writes 78.44 seconds.
The comparison is matched on experience, not wall time or optimizer work.

**Decision:** continue all three declared pairs, regardless of this favorable
first result. The owner started CURL seed 456 immediately after the control
exited. Its real simulator returns arrived after 87.32 seconds and finite
updates after 104.27 seconds. Three later jobs remain queued. If the completed
cohort supports the gap, a longer-horizon comparison can ask whether the
control catches up; if seed outcomes vary, more seeds may be more informative.
No live protocol or training source was changed.

See [paired curves](figures/walker-first-pair/learning-curves.pdf),
[machine-readable summary](figures/walker-first-pair/summary.json),
[reference records](walker-data/first-reference) and
[control records](walker-data/first-control). The guide adds a page with the
task, score definition, complete curves, limitations and next decision.

## First walking reference complete, October 5 at 02:01 UTC

The first fresh walker/walk CURL model finished at **482.83347 average episode
reward** after the declared 100,000 training simulator steps. Its initial mean
was 17.98903. This establishes learning in one run on the second task, not an
advantage over the control or a reproduction of the paper's average.
Its matched control is running; all three pairs remain planned.

The last six evaluation means were 489.84, 541.42, 526.97, 491.15, 543.51 and
482.83. We report the fixed final value, not the more flattering 96k peak.
The final ten declared evaluation starts give returns from 202.33 to 612.46.
These ten episodes describe one trained model, not ten training replicates.

Native records confirm a natural 50,000-decision endpoint, 100 training
episodes, 49,000 ordinary update calls, exit 0 and complete=true. All 3,920
logged learning values are finite. The final checkpoint is 9,137,791,883 bytes,
matching its receipt; saving took 23.77 seconds. No recovery or repeated
training was needed. It used 100k training plus 260k evaluation simulator steps.
Start-to-end wall time was 4,489.66 seconds (74.83 minutes), including 2,106.02
seconds of evaluation and 74.50 seconds of checkpoint writes. The remainder is
not a separately measured optimizer-only duration.

The existing owner launched the control 0.018 seconds after the reference
process exited. Initial simulator returns arrived after 87.87 seconds, and
finite learning updates after 105.28 seconds following warmup. Its configuration
differs only in the arm, and its initial evaluation mean matches the reference.
This checks the matched starting setup, not equality of later training data.
Four more jobs are queued after the active control.

**Decision:** finish all three declared pairs without changing the endpoint.
Use the matched outcomes to decide whether the cartpole early benefit extends
to walking. A single reference score cannot answer that question, even if it
looks favorable next to a published mean. No new mechanism claim is warranted.
The guide records this single-run milestone; comparative walking plots will
be added when a matched pair exists.

See [native records and configuration](walker-data/first-reference) and
[the predeclared walking protocol](WALKER_PROTOCOL.md). Earlier entries below
retain their original evidence cutoffs.

## Completed cartpole cohort, October 5 at 00:46 UTC

| Seed | CURL at 100k | Control at 100k | Early difference | CURL at 500k | Control at 500k | Late difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 123 | 678.02 | 454.47 | +223.55 | 842.74 | 866.94 | -24.20 |
| 456 | 446.17 | 240.54 | +205.63 | 866.14 | 810.96 | +55.18 |
| 789 | 587.99 | 463.26 | +124.74 | 855.22 | 870.53 | -15.31 |
| Mean | 570.73 | 386.09 | +184.64 | 854.70 | 849.48 | +5.22 |

The early gap shrank in every pair. Each score is the mean reward of ten fixed
evaluation episodes for the planned endpoint, not the best checkpoint. A row
follows one training seed through both budgets, not two independent replicates.
An independent CPU audit recomputed the 12 endpoint means from 120 evaluation
episode records, checked
the ten declared starts, terminal counters and matched configurations, and
confirmed these values. Later sample SD across seeds is 11.70 for CURL and
33.41 for the control; three seeds do not establish a precise population effect.

**What we learned:** more training changed the comparison substantially. The
extra image-matching exercise helped at the early endpoint, but did not produce
a consistent later advantage in this task and protocol. The control also
learned strong behavior. We should not call the extra exercise necessary for
learning, or infer a universal benefit or harm from one horizon.

**What remains uncertain:** close averages do not prove equivalence. We have
one task, three training seeds and ten fixed evaluation starts. The software
stack differs from the historical paper. Removing the matching update also
removes extra optimizer work, so the ablation does not isolate image correspondence
information from additional optimization. Shared-checkpoint interventions could
test continued value, but this scratch comparison does not identify a mechanism.

**New endpoint evidence:** final control mean 870.5324417389436; natural
62,500 decisions, 500k training steps and 61,500 update calls, exit 0 and
complete=true. All 3,500 logged learning values are finite; no failure. Last
five means: 868.39, 866.53, 869.62, 845.11, 870.53. The 11,354,767,581-byte
checkpoint matches its receipt and took 29.47 seconds to write. It used 500k
physical training steps and 1.26m evaluation steps. All six final chains pass
the same endpoint and ancestry checks.

**Cost caveat:** first-pair CURL used 602k physical training steps after repeating
102k lost-state work. The other five models used 500k each. All retain a 500k
history; the abandoned branch is not plotted or counted as another seed. These
are not equal-physical-cost claims. Evaluation work is separate from training.

**Decision and launch:** the prepared walker/walk comparison started at
00:43:32 UTC, after cartpole queue completion and owner exit. It asks whether
the early benefit appears on another task with fresh models, not whether
cartpole-trained weights transfer. Run all three declared pairs regardless of
the first result. First real returns were recorded after 89.59 seconds, following
an 85-second initial evaluation; finite learning updates followed after 107.46
seconds, after the random-action warmup. No walking endpoint exists yet.
The 107-second gap between cartpole queue exit and walker child launch included
review/admission checks and a corrected read-only command typo; it was not
training. No document polishing delayed the launch.

See [final control records](extension-data/no-curl-seed789-500k),
[complete six-run summary](extension-data/three-500k-pairs-summary.json),
[all three longer curves](figures/three-completed-pairs/completed-500k-pairs.pdf)
and [walker protocol](WALKER_PROTOCOL.md). Older dated entries below preserve
what was known at each earlier decision point.

## Third longer CURL result at 23:46 UTC: all references complete

CURL seed 789 finished at **855.2208340094433**, up from 587.9943 at 100k.
The three longer CURL scores are now 842.74, 866.14 and 855.22: mean 854.70,
sample SD 11.70 across three training seeds. This is not a third paired result:
the final control has only just started. Do not compare a three-seed CURL mean
against a two-seed control mean as though the comparison were matched.

The new model ended naturally at 62,500 decisions, 500,000 training simulator
steps and 61,500 update calls, with exit 0 and complete=true. Its final mean was
recomputed from the ten declared evaluation seeds. All 4,000 logged learning
values are finite; no failure occurred. The last five means were 836.21, 854.12,
871.24, 866.71 and 855.22. We report the final value, not the highest one.
Its 11,386,654,987-byte checkpoint matches the receipt and took 29.88 seconds
to write. This chain used 500k physical training interactions and 1.26m
evaluation interactions, with no abandoned training branch.

All three reference runs learned substantially more with continued training.
That supports a working learning reproduction, not an exact reproduction of
the paper's ten-seed mean or proof that the extra matching update helps later.
The two completed paired differences remain -24.20 and +55.18. The observed
early benefit and uncertain later benefit remain the main local pattern.

The owner started the final control 3.78 seconds after CURL exited. Actual
learning records arrived within 12 seconds; finite learning and natural
episodes were checked through 107k, with mean 528.19 at 104k. Keep that run's
fixed endpoint unchanged. Six walker runs remain prepared, not launched.
Next: finish this pair, test scope on walker, then reconsider the conditional
shared-checkpoint matching-on/off study. No source or GPU owner changed.

Evidence: [third completed CURL records](extension-data/curl-seed789-500k) and
[dated continuation snapshot](extension-data/third-curl-500k-summary.json).
The guide's paired figure retains its earlier 22:48 cutoff and only the two
completed pairs; the text and reference table now include the new endpoint.

## Two longer pairs at 22:48 UTC: the first reversal is not universal

| Training seed | CURL at 100k | Control at 100k | CURL at 500k | Control at 500k | Final paired difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 123 | 678.02 | 454.47 | 842.74 | 866.94 | -24.20 |
| 456 | 446.17 | 240.54 | 866.14 | 810.96 | +55.18 |

All are the declared endpoint scores, not best checkpoints. Each score averages
ten fixed evaluation starts; each row is one training seed. The corresponding
early differences were +223.55 and +205.63. Both late differences are smaller,
but their signs differ. The two-pair average difference is +15.49, a descriptive
number that must not hide the opposing outcomes or be treated as precise evidence.

**What changed:** the second pair contradicts a general account in which the
control necessarily catches up and wins. Extra image matching may help early
learning more reliably than final performance, and the later result can depend
on training randomness. This is a hypothesis supported by a small local pattern,
not a general causal mechanism or evidence that the methods are equivalent.

**New endpoint checks:** control seed 456 ended naturally at 62,500 decisions /
500,000 training steps / 61,500 update calls, exit 0 and complete=true. The
mean 810.9574102627266 was independently recalculated from the ten expected
evaluation seeds. All 3,500 logged training values in this invocation are finite;
no failure occurred. Its last five means were 801.17, 781.36, 804.81, 815.77 and
810.96. The final checkpoint is 11,354,767,581 bytes, matching the receipt;
write time was 29.44 seconds. Both second-pair chains pass configuration,
ancestry and fixed-endpoint checks. Each used 500k physical training interactions
and 1.26m evaluation interactions. CURL still performs extra optimizer work,
so matching interactions does not match wall-clock cost. First-pair CURL's extra
102k recovery interactions remain separately reported.

**Decision:** finish the third pair, then prioritize testing scope over explaining
the first seed's reversal as general late harm. The owner already started CURL
seed 789; real scientific updates arrived within 11 seconds, followed by finite
learning and a checked evaluation of 565.99 at 116k. Its matched control is
queued. A CPU protocol check is preparing a second-task comparison, not changing
the live cartpole runs. The shared-checkpoint matching-on/off proposal remains
available to test continued value, but no mechanism study was launched.

The guide now shows both completed pairs side by side and keeps the three early
results. Evidence: [second control records](extension-data/no-curl-seed456-500k),
[two-pair snapshot](extension-data/two-500k-pairs-summary.json), and
[completed-pair curves](figures/completed-500k-pairs.pdf). Earlier sections below
preserve interpretations at their original cutoffs.

## Second longer CURL result at 21:53 UTC: more learning, comparison still pending

CURL seed 456 finished at **866.1353**, up from **446.1711** at 100k. It used
500k physical training interactions, without an abandoned branch. This is the
second completed longer CURL model, but its control is still training. Do not
compare the mean of two completed CURL seeds with one completed control seed as
though that were a matched comparison. The only completed 500k pair remains
seed 123, where the control finished higher.

The new endpoint is the planned final policy, averaged over the ten expected
evaluation starts. The last five measured means were 862.47, 863.91, 860.56,
860.65 and 866.14. The final mean was independently recalculated from episode
returns. Native completion is 62,500 decisions / 500,000 training steps / 61,500
update calls; exit 0, complete=true, natural final episode, no failure, and all
4,000 logged metric values in this invocation are finite. The final checkpoint
is 11,386,654,987 bytes, matching its receipt; the write took 29.82 seconds.
Evaluation used 1.26m interactions across the original run and continuation,
separate from training. Continuing a saved model is not a new independent seed.

**Interpretation:** both completed CURL models learned substantially more with
longer training. That does not establish that continuing the extra objective
helps compared with omitting it. The fresh seed's successful large saves also
show that the repaired save path works beyond the original recovery case.
This is operational confirmation, not an extra scientific replicate.

**Decision:** keep the fixed-endpoint queue unchanged. The owner started control
seed 456 about three seconds after CURL exited. Real learning returned within
11 seconds; the control subsequently reached 127k with finite updates and a
checked mean of 455.1383 at 124k. Both seed-789 extensions remain queued.
Conditional mechanism and second-task proposals in NEXT_COMPARISONS.md remain
unlaunched. The guide records the second CURL endpoint without adding an
unmatched curve or claiming a second paired result.

Evidence: [completed second CURL records](extension-data/curl-seed456-500k),
[dated continuation snapshot](extension-data/second-curl-500k-summary.json).
Original records are under the repaired external campaign root. The older
sections below preserve what was known at their stated cutoffs.

## First longer pair at 20:50 UTC: an early lead does not guarantee a later lead

| Retained training steps | CURL | Same crops, no image matching | CURL minus control |
| --- | ---: | ---: | ---: |
| 100,000 | 678.02 | 454.47 | +223.55 |
| 500,000 | 842.74 | 866.94 | -24.20 |

Both rows follow the same training seed, not independent repetitions. Each score
averages ten fixed evaluation starts. The 500k scores are the planned terminal
policies, not the best points along the curves. Neither failed attempts nor
the diagnostic pilot contribute scores. See the [paired summary](extension-data/first-500k-pair-summary.json),
[completed control records](extension-data/recovered-control-seed123), and
[paired curve](figures/first-500k-pair.pdf).

**What changed:** the early difference in this pair did not persist. That is
consistent with an early-learning benefit, followed by the control catching up.
It is not proof that the extra objective caused late harm, nor that the methods
are equivalent or that the control usually wins. A single training seed cannot
establish the size or consistency of this late difference. The other two seeds
and another task could change the interpretation.

**Checks:** the control ended naturally at 62,500 decisions / 500,000 retained
training steps / 61,500 update calls, with all ten expected evaluation seeds and
mean return 866.9394162304535. Its 2,639 logged update values in this invocation
are finite; no failure appears. The final checkpoint is 11,354,767,581 bytes,
matching its receipt, with a 30.45-second write. Exit was 0 and complete=true.
The last five evaluation means were 854.99, 853.15, 874.71, 860.42 and 866.94.
Both chains pass the same configuration/ancestry and endpoint checks.

**Costs and scope:** CURL used 602k physical training interactions after repeating
102k lost-state work; the control used 500k. Both retain a 500k-step history.
Their physical evaluation totals are 1.53m and 1.27m interactions, respectively,
separate from learning. This is not an equal-physical-cost efficiency comparison.
The chart excludes the abandoned branch and any unfinished training seed.

**Decision:** finish the remaining two prespecified pairs. CURL seed 456 has
already resumed from 100k under the same owner and produced real updates within
12 seconds, then finite learning and checked evaluation through 108k. Its control
and both seed-789 extensions remain queued. No live source was changed.

The learning guide gains one page showing both curves and the limits above.
The original 100k result is preserved, now clearly labeled as early performance.
This is a useful reproduction lesson, not a novel general result. Any experiment
about *when* to stop the extra matching objective must be declared as a new
comparison, not inferred from this changing ordering alone.

## Earlier 500k result at 20:08 UTC: a higher score, not yet a comparison

The first CURL model scored **842.7436** at the fixed final checkpoint, compared
with **678.0208** at 100k: a gain of **164.7227** reward points for the same
training seed. The score averages ten fixed evaluation episodes on cartpole
swing-up. Those episodes are not ten separately trained models. The final
reward was not chosen from the best checkpoint; the last five measured means
were 843.62, 832.22, 847.59, 847.46 and 842.74.

**Interpretation:** this model learned a stronger controller with longer
training. It does not yet show that image matching remains beneficial compared
with the same-crops control. That control is now running and had already learned
substantially before its planned technical stop. Nor does one score close to a
published average reproduce the paper's multi-seed result on its original stack.
Continue the prespecified paired endpoints rather than declaring success early.

**Native completion checks:** 62,500 decisions, 500,000 retained training steps,
61,500 update calls, all ten expected evaluation seeds, a natural final episode,
no failure and all 1,928 logged update values finite in the repaired invocation.
The final checkpoint is 11,386,654,987 bytes, matching its receipt; its write took
29.68 seconds. The owner reports exit 0 and complete=true. Full original ancestry
and recovery metadata pass the continuation analysis.

**Cost:** this one chain used 602,000 physical training interactions because
102,000 had to be repeated after the saving failure. Its three invocations also
performed 1,530,000 actual evaluation interactions (260k + 770k + 500k), none used
for learning. The restored counter ends at 1,270,000 evaluation interactions;
that counter omits the abandoned branch, so it is not the physical total.
No equal-physical-budget claim is justified by the repaired comparison.

**Next:** finish its matched control, then the remaining two pairs, all already
queued. The control resumed at 198k, produced real update metrics within
18 seconds of launch and reached 203k with a checked ten-episode evaluation.
No extra GPU owner or new training variant was started. See the
[completed recovery records](extension-data/recovered-curl-seed123) and the
[dated continuation summary](extension-data/first-500k-summary.json).
The guide now shows this single completed endpoint explicitly while retaining
the three-pair 100k figure. A paired 500k plot will be more informative once both
versions have finished; no unpaired line is presented as comparative evidence.

## Recovery check at 19:54 UTC

The repaired run successfully saved its first large live checkpoint at
**408,000 retained training steps**: 9,316,517,003 bytes, written in 23.24 seconds.
The file size matches the native receipt, and finite updates and natural episodes
continued beyond the old 409k failure point. This verifies the repaired save path
on the real workload; it is not a new final performance score.

The repeated portion of training provides a useful consistency check. All 1,024
overlapping logged update values, 260 evaluation episode returns and 102 training
episode returns after the restored 307k boundary exactly match the old branch.
This is observed agreement for these records on this machine. It does not prove
that all unlogged state matches or that every future restart will reproduce an
uninterrupted CUDA run. Repeated interactions still cost real compute and are
not additional independent evidence. Five other extensions remain queued.

The new review alert itself was the already documented clean control stop,
not an additional completed scientific result. Continue to the fixed endpoints.
The learning PDF's completed-results cutoff and figure remain unchanged.

## Earlier recovery decision at 19:42: a save-format limit, not a learning result

The first CURL extension reached **409,000 training steps**, then failed while
saving its replay memory. Our checkpoint adapter relied on a serialization
format that cannot encode a single NumPy array larger than 4 GiB. The learning
metrics remained finite; the last measured evaluation was 823.41 at 408k.
That intermediate reward is not a substitute for the missing 500k endpoint.
This is a defect in our adapter, not evidence of a failure in the paper's method.

The previous checkpoint at **307,000 steps** survived because saves replace
the old file only after the new one is complete. We repaired the format and
verified a real save-and-reload of an array larger than 4 GiB. The independent
root check took 26.82 seconds, about 16.5 GiB peak RAM and 4.30 GB temporary
disk; all three focused checkpoint tests passed. The small pilot and earlier
checkpoint checks did not cover this size boundary.

The fixed recovery rule is to use the latest intact natural-boundary checkpoint,
regardless of its reward. We preserved the failed run and asked its owner to stop
the control before the same limit. The control saved cleanly at **198,000 steps**.
The repaired owner then acquired the same GPU lock and resumed CURL at **307,000**.
Its first real evaluation returned within 37 seconds of launch; finite learning
updates and natural episodes followed. At this check it had reached 328,000
retained steps. The control and four unstarted extensions remain queued.
See the [recovery plan](RECOVERY_PLAN.md).
The learning algorithm, evaluation starts and retained training target stay the
same. Startup and resumed learning are verified; completion and a large live
checkpoint under the repaired format still need observation.

**An important accounting distinction:** the failed 307k-to-409k branch consumed
102,000 real training interactions, about 15 minutes since the previous save,
but its learning state was not retained. A recovered CURL path that finishes at
500k will therefore have used **602k physical training interactions**. We must
report this extra cost, not describe the repaired comparison as equal physical
compute. Its saved model will still contain a 500k-step training history. The
abandoned curve stays available as diagnostic evidence but must not be spliced
into the recovered curve or counted as another training seed.

Native [failed-attempt records](extension-data/failed-seed123) include the error,
successful earlier checkpoint receipts and interrupted learning curve. They are
stored outside the original `data/` cohort so the unchanged 100k report cannot
accidentally combine duplicate training seeds. The original six 100k results
and figure remain valid. The [cleanly stopped control](extension-data/stopped-control-seed123)
also retains its native records. There are no completed 500k scores at this update.

## All three original pairs are complete

| Training seed | CURL | Same crops, no image matching | Paired difference |
|---|---:|---:|---:|
| 123 | 678.02 | 454.47 | +223.55 |
| 456 | 446.17 | 240.54 | +205.63 |
| 789 | 587.99 | 463.26 | +124.74 |
| Mean across three training seeds | 570.73 | 386.09 | +184.64 |

The standard deviations across trained seeds are 116.89 for CURL and 126.13
for the control. They describe variation between training runs, not uncertainty
from treating the ten evaluation episodes as independent trained models. The
[summary](data/three-pairs-summary.json) and all six native run directories
provide the numbers behind the figure. No completed run was excluded.

**What changed:** the third, prespecified comparison also favors CURL, though
its advantage is smaller. The direction repeated across all three training
seeds. This supports a local benefit from the additional image-matching update
at 100k interactions. It does not demonstrate a consistently higher curve,
general usefulness across tasks, or a novel method. Our mean is near the paper's
published 582, but agreement on one task with three seeds and a modern simulator
is not a reproduction of the paper's ten-seed benchmark.

**Competing explanations and limits:** the extra update changes both the
learning objective and optimization work. We have not isolated correct image
correspondence from all other effects of that update. Learning curves fluctuate,
and longer training may let the control catch up. The same ten evaluation
starts were reused throughout: this makes comparisons stable, but does not
establish performance on a broad range of new starting conditions.

**Native checks:** all six runs have exact terminal counters of 12,500 decisions,
100,000 training simulator steps and 11,500 update calls; all ten terminal
evaluation seeds are present. Logged updates are finite, no failure events
appear, and final checkpoints match their byte-count receipts at natural episode
boundaries. Each run also used 260,000 evaluation simulator steps, kept out of
training. The third control's loop took 792.07 seconds and its checkpoint write
took 11.48 seconds; the third CURL loop took 853.63 seconds plus 11.62 seconds
for its checkpoint.

**Decision:** continue all six models to the previously proposed 500k endpoint.
The owner launched at 18:36:58 UTC after every original result was checked.
Each continuation retains its original seed, replay, optimizers and counters;
each parent/child chain is one replicate. Original checkpoints remain intact.
The question is whether the early advantage lasts, shrinks or reverses. These
are documented resumed runs, not a claim of bit-for-bit identity to uninterrupted
GPU training. There are no completed 500k scores at this cutoff.

**Publication/teaching impact:** the eight-page guide now shows all three pairs,
the precise score definition and the longer-training question. This is useful
replication evidence and a worked learning example, not a publication novelty
claim. Finish the horizon comparison before choosing the next mechanism probe
or a second task.

## Earlier checkpoint at 18:15: two matched pairs

The second matched control completed with a score of **240.54**, versus
**446.17** for CURL, a difference of **205.63** reward points. The first pair's
difference was **223.55**. All four runs completed the same 100,000 training
simulator steps and 11,500 update calls. Each score averages ten fixed evaluation
episodes. There are two independently trained pairs, not twenty repetitions.

| Training seed | CURL | Same crops, no image matching | Paired difference |
|---|---:|---:|---:|
| 123 | 678.02 | 454.47 | +223.55 |
| 456 | 446.17 | 240.54 | +205.63 |

This strengthens the initial signal: the higher CURL endpoint was not confined
to the first training seed. It remains a small exploratory comparison on one
task and one training budget. We have not selected favorable checkpoints or
dropped unfavorable runs. The difference measures the whole additional update,
not just correct image correspondence independently of extra optimization.

The curves give a more qualified picture than the endpoints alone. The first
pair often traded places. CURL led for more of the second run. As a post-hoc
descriptive check, the second pair's interpolated curve averages are 272.08
for CURL and 187.46 for the control, compared with nearly equal averages in
the first pair. These are not additional primary outcomes or independent
repetitions. We retain the prespecified endpoint as the main comparison.

The second control had no failure records or nonfinite logged update metrics.
Its training/evaluation loop took 785.78 seconds and its final 2.506 GB checkpoint
took another 12.16 seconds. Evaluation consumed 260,000 separate simulator steps,
not training data. Its [native records](data/second-control) are published.

**Decision:** finish the third pair, then extend all three pairs to 500,000
steps if their final records and checkpoints are valid. This tests whether the
early difference lasts, shrinks or reverses. Prepare the queue changes while
the last pair runs, preserving all original checkpoints and analysis. The PDF
now shows both completed pairs and explicitly says that the evidence is limited.

## Earlier checkpoint: the second reference before its control finished

With a new training seed, CURL improved from **18.62** to **446.17** at the
same fixed 100,000-step endpoint. Its first training seed finished at 678.02.
This is direct evidence that the exact result depends on the training run,
even when the task and settings are unchanged. It is not a failure of the
second run: it completed 11,500 updates, all recorded update metrics were finite,
and it saved its full checkpoint. The last three scores were 399.57, 492.46
and 446.17, so this run did not repeat the first run's final upward jump.

The matched control for this new seed is still training. **We cannot yet say
whether CURL's advantage repeated.** Comparing two completed CURL runs against
just one control would change the comparison as results arrive. We will keep
the matching by training seed and report unfinished runs separately.

Native evidence is in [data/second-reference](data/second-reference). There are
ten evaluation episodes per score, not ten independent training runs. The loop
took 851.57 seconds, excluding startup and the final checkpoint write.

**Decision:** leave the queued comparisons unchanged. Prepare the longer-training
comparison on CPUs while these finish. The PDF retains its explicitly dated
first-pair snapshot; this individual, as-yet-unpaired result reinforces its
existing caution rather than changing the main conclusion. Update the PDF when
the next complete pair changes what can be concluded.

## First matched pair: a higher endpoint, not a consistent lead

At 100,000 training simulator steps, CURL scored **678.02** and the control
without image matching scored **454.47**. Both started at **8.44**. These are
means of ten fixed evaluation episodes from **one training seed per version**,
not ten independently trained models. The paired endpoint difference is +223.55.
Both runs completed 11,500 update calls and saved their full checkpoints.
The control's native records are in [data/first-control](data/first-control).

The learning curves trade places. At 76k steps the control scored 354.84 while
CURL scored 222.24. CURL's final jump matters to its endpoint advantage. Because
evaluation uses the same starting seeds and deterministic actions, these curves
are not merely noisy because we drew different evaluation starts each time.
The policies themselves change during training.

As a **post-hoc descriptive check**, linearly interpolating and averaging each
recorded learning curve over 0–100k steps gives 249.29 for CURL and 250.30 for
the control. This was not a prespecified primary metric and does not replace
the endpoint. It cautions against claiming a consistent sample-efficiency gain
from the last point alone. Checkpoints are not independent replicates.

CURL's training/evaluation loop took 866.44 seconds; the control took 782.83.
Their evaluation portions were 371.88 and 362.31 seconds. This is an equal
training-interaction comparison, not equal computing time. Startup and final
checkpoint writes, about 11 seconds each, are excluded from those loop times.

**Decision:** finish both arms for the two remaining prespecified training seeds.
If those runs are valid, a paired extension to 500k can test whether the apparent
late advantage persists. Do not launch a new loss variant merely because one
endpoint is favorable. A fixed additional evaluation panel for all final
policies is another possible check, not a substitute for training repetitions.

## Earlier checkpoint: first 100k reference

Fresh run `curl-seed123-100k-v1` completed exactly 12,500 decisions and 100,000
training simulator steps, with 11,500 update calls. Its fixed ten-episode mean
return was 8.4446 before updates and 678.0208 at the endpoint. This is reward
on a roughly 0–1,000 scale, not a success percentage. Native records and config
are published in [data/first-reference](data/first-reference).

The trajectory fluctuates substantially: the 96k score was 423.07, followed by
678.02 at 100k. The endpoint was prespecified, not picked after looking at scores,
but this last-point jump makes repeatability and later behavior important. Do
not infer a stable 678-level policy or a repeatable advantage over the control.

The paper's 582 ± 146 is a mean and SD over ten trained seeds, not a threshold
that one local result can pass. Our runtime differs from the historical stack.
That result supported continuing the reference/control comparison; it was not a
claim that we reproduced the entire paper or discovered a new method.

The loop took 866.44 seconds, including evaluation; the final 2,540,402,075-byte
checkpoint took 11.43 seconds more. Setup and startup are excluded. Evaluation
consumed 260,000 separate simulator steps, not training data. The control
started automatically within about one second of reference termination and
produced scientific records within 20 seconds of launch.

Next decision: finish the already-queued three-seed pair before choosing another
arm. The [follow-up memo](NEXT_COMPARISONS.md) explains why a matched extension
to 500k and a second task are more informative than declaring victory from one
score. A shuffled-positive control is conditional, not launched.

## First pilot: execution works

Run `pilot-123-v1` used 1,200 decisions, with the first 1,000 collecting random
actions. It performed 200 subsequent learning updates. The 9,600 training
simulator steps and 6,000 evaluation steps are counted separately. The timed
training/evaluation loop took 29.90 seconds; final checkpoint writing took
another 1.37 seconds. Startup and environment installation are not included.

The deterministic two-episode mean return went from 8.40 before training to
22.58 at the endpoint. This short check is **not** evidence of a reliable
learning improvement, and is excluded from the main comparison. The contrastive
loss was near log(128) at the first updates, consistent with initially weak
image matching. Neither that observation nor a finite loss proves good control.

The final checkpoint was 366,475,547 bytes and includes optimizers, temperature,
replay data and random states. A separate CPU test using the actual upstream
pixel agent reproduced its next fixed-batch update exactly after restoration,
including tied parameter identities. That is narrower than exact GPU trajectory
reproduction. The pilot ended partway through an episode; its checkpoint labels
that truncation, and a later resume would start a fresh episode.

## What could still explain a poor full run?

The modern simulator/renderer differs from the historical stack. The task may
also need more experience than this endpoint, and one random training seed can
be misleading. We preserve the authors' learning code, including its two
overlapping encoder optimizers, rather than silently changing the method.

The matched control retains random crops and all SAC updates, but omits CURL's
extra update. Therefore a difference would measure the contribution of that
whole extra update, including extra optimization and computing time. It would
not isolate the mathematical objective from the number of encoder updates.

## Next decisions

1. The fresh CURL and no-contrastive runs for seed 123 are complete at 100k steps.
2. Complete both arms for seeds 456 and 789 regardless of the first pair's sign.
3. If both are valid, consider a 500k extension of both arms. Do not select only
   a favorable seed or checkpoint. If learning fails, inspect actual observations
   and learning signals before expanding the budget.
4. Independent contrastive calculation check passed: NumPy's direct calculation
   agreed with the authors' Torch logits, loss, feature/weight gradients and one
   SGD matrix update to within 5.56e-17 absolute error. Correct matches gave loss
   0.29334; deliberately mismatched pairs gave 2.06667. These are invented small
   inputs, not learned features or RL evidence. The actual training uses Adam.

No novelty or publication claim is made. The immediate output is a reproducible,
understandable learning example that improves our experimental practice.

## Operations and honest cost accounting

Execution access was restored around 16:30 UTC. The first real pilot started
around 17:11 UTC after environment installation, adapter implementation and
focused checks. This preparation interval was not GPU training. An initial
background shell launch disappeared before producing records; a persistent
execution session then ran successfully. That brief failed launch consumed no
scientific budget and is retained in this account. Future owners must verify
actual episode/update records, not assume a returned PID means work is running.

The advisor deck is not changed by this first reproduction run. The user changed the priority to
paper reproduction and requested a learning document; that document is the
appropriate place for these execution findings.
