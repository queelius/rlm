# Learning RL by reproducing CURL

Start with [the eleven-page PDF](learning-guide.pdf). The
[LaTeX source](learning-guide.tex) and Makefile are next to it.

This edition includes **all three completed pairs at both 100k and 500k**,
plus all three original walking pairs, their supplemental tests, and two
completed pairs from the new training cohort. Latest cutoff: October 5, 13:39 UTC.
It explains the cartpole task,
reinforcement learning, self-supervised image matching, our comparison, and
what source review found. CURL scored 678.02 versus 454.47 in the first pair,
446.17 versus 240.54 in the second, and 587.99 versus 463.26 in the third.
All six models continued to the fixed 500k endpoint. The longer comparisons
have different winners: **842.74 versus 866.94**, **866.14 versus 810.96**, and
**855.22 versus 870.53** (CURL first in each).
The mean advantage shrank from **+184.64 to +5.22**. That is an early benefit
without a consistent late winner in this small study, not proof of equivalence.
Page 8 preserves the early comparisons; page 9 shows all three longer pairs
with limits and recovery cost. The fresh-model walking comparison is also complete:
**482.83 versus 241.28**, **385.23 versus 360.96**, and **435.83 versus 698.06**,
with CURL first. Its two wins are offset by a large loss in the third pair.
The means, **434.63 versus 433.44**, hide those differences. Page 10 shows
all six curves. Three pairs establish neither a dependable winner nor equivalence.
Page 11 asks whether changing the test starts changes that pattern. On 50 new
starts, the same pairs favor the same methods: differences **+287.28, +23.01
and -240.92**. The third pair's control advantage persists. This is still three
training pairs; testing their frozen models again does not add training replicates.
Published scores and invented teaching
examples are identified explicitly.

The [running findings](FINDINGS.md) explain the latest comparison and preserve
the earlier interpretations as dated checkpoints.

The first two additional training pairs also have opposite winners:
**502.95 versus 200.43** and **254.63 versus 335.20**, CURL first. Their
[native records](walker-additional-data) and
[learning curves](figures/walker-additional-two-pairs/learning-curves.pdf) are
reported separately from the original cohort. The third new pair is still
running. Page 11 includes this limited update without claiming a complete cohort.

Update at October 5, 14:58 UTC: the third CURL model finished at **436.58**,
but its control is still running. This is not yet a third paired result.
The [running findings](FINDINGS.md) explain it and link the native records.
The PDF retains its earlier, explicitly dated comparison cutoff.

The [fresh-start evaluation](WALKER_FRESH_STARTS.md) is complete for all six
unchanged walking models. [Three additional training pairs](WALKER_ADDITIONAL_SEEDS.md)
have begun to check variability between independently trained models.
This decision follows the original mixed cohort, not favorable selection from
the supplemental evaluation. New test episodes and new training runs answer
different questions; neither should be counted as the other.

The [research protocol](../../experiments/curl_replication_20261004/README.md)
contains the selected comparison, run order, settings, resolved access issue and
resumption instructions. The [machine-readable manifest](../../experiments/curl_replication_20261004/manifest.json)
records source revisions and the tested compatibility runtime. Read the
[running findings](FINDINGS.md) for native result pointers and limitations, and
the [implementation review](IMPLEMENTATION_REVIEW.md) for the focused independent review.

Build on a machine with a normal LaTeX installation:

```sh
make -C docs/curl-replication-2026-10-04
```

Or use Tectonic with the already-populated cluster cache:

```sh
make -C docs/curl-replication-2026-10-04 tectonic \
  TECTONIC=/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic
```

The Tectonic target deliberately uses only cached dependencies. On a new machine,
use `tectonic learning-guide.tex` from this directory to allow ordinary package
retrieval. The document uses Beamer as a letter-sized portrait reading layout,
not as a slide deck, because the article class was unavailable in the cluster's
offline cache. It uses readable 12-point text.

## Updating after a run

Add the actual run identity and cutoff, a curve sourced from per-episode records,
individual-seed results, a plain-language interpretation and a competing
explanation. Update the first and last pages' status at the same time. A successful
pilot is not a reproduced learning result; a missing endpoint is not a zero score.
Keep published reference scores separate from our measurements.

## Verification of this edition

Compiled with Tectonic 0.17.0 using cached dependencies on October 5, 2026.
The eleven-page edition includes the complete supplemental walking test and
the first two pairs of the additional training cohort, clearly marked incomplete.
Changed pages 1, 7 and 11 were rendered and visually inspected, along with the
new two-pair plot. The page-1 footnote crowding was corrected. No TeX overflow
warnings or out-of-page text were found; other pages retain their earlier inspection. The numerical
teaching example was independently recalculated. The manifest parses as JSON,
and its decision count times action repeat equals the stated environment budget.

PDF SHA256:
`ecd9c915d12b10d197cb6d95ebba70677dbb2775777062c91e3cc4a7ea638e7a`.
The paired-figure code passed eight focused selection/endpoint tests, Ruff checks
and independent code review. It uses the recorded joined curves without
smoothing and excludes incomplete seeds and the abandoned branch.
The runtime checks include real simulator behavior, CPU checkpoint restoration,
and a completed GPU pilot. Three full matched pairs also completed, but this is
still a small exploratory comparison, not a reproduction of the paper's mean.
All six completed 500k endpoints passed native terminal and ancestry checks.
An independent CPU audit recalculated the 12 endpoint means from 120 episode
records across both budgets, checked the ten evaluation starts and matched configurations, and
confirmed the paired differences. Recovery cost is reported separately.
For the three walking pairs, the analysis and an independent CPU audit recomputed all
156 means from 1,560 episode records, checked the ten declared starts, native
endpoints, finite learning values and paired configurations. All 24 public
record files match their external originals byte for byte. The six checkpoints'
sizes were checked against receipts across the completed reviews.
The existing summarizer was reused without a code change. Its previous plot-title
addition passed a failing-first rendered-PDF regression, all seven focused
tests and Ruff checks. This data/document update required no new test campaign
or live training-source change. The prior first-pair figure remains preserved.

For the supplemental panel, root and independent CPU audits recomputed all six
means from 300 raw returns and checked exact starts, successful endpoints,
zero learning updates and parent provenance. All six original final models
are included. Native public records preserve the complete panel, not just the
favorable comparisons. All 30 copied evidence files match the native originals
byte for byte. The [machine-readable summary](walker-fresh-starts-data/summary.json)
includes exact values, parent identities and file checksums. No large checkpoint
weights are committed to Git.

For the two additional pairs, root and independent CPU audits recomputed all
104 means from 1,040 episodes, verified the exact test starts, native endpoints,
finite learning, matched configurations and checkpoint receipts. Scientific
settings match the original cohort except for the new training seeds. Their
four curves preserve late fluctuations and crossings, not just final winners.
All 16 public record files match their external originals byte for byte.

## Evidence and next comparisons

- [Correct versus wrong image matches](CORRESPONDENCE_CONTROL.md): a three-run
  cartpole follow-up admitted and waiting behind the final walking control. It retains
  the original correct-matching and no-matching comparisons, with an explicit
  warning that false targets can harm learning. Its training has not started yet.
- [Declared fresh-start evaluation](WALKER_FRESH_STARTS.md) and its
  [fixed panel and budget](walker-fresh-starts.json): a supplemental check of
  all six final walkers on 50 new starts each. This separate evaluation is
  complete, with [all native records](walker-fresh-starts-data). Page 11 reports
  it separately; page 10's original results remain unchanged.
- [Additional training-seed protocol](WALKER_ADDITIONAL_SEEDS.md): three new
  matched pairs with unchanged settings, now running after the evaluation batch.
- [First walking reference](walker-data/first-reference) and
  [matched control](walker-data/first-control), plus
  [second reference](walker-data/second-reference) and
  [second control](walker-data/second-control), and
  [third reference](walker-data/third-reference) with
  [third control](walker-data/third-control): three completed pairs at fixed 100k.
- [Current walking paired summary](figures/walker-three-pairs/summary.json) and
  [learning curves](figures/walker-three-pairs/learning-curves.pdf).
  All six original runs are included. To rebuild this complete-cohort snapshot:

  ```sh
  python experiments/curl_replication_20261004/summarize.py \
    --runs docs/curl-replication-2026-10-04/walker-data \
    --output docs/curl-replication-2026-10-04/figures/walker-three-pairs \
    --title 'Learning to walk: the three pairs have different winners'
  ```

  Compare the individual pairs as well as the averages. The near-zero mean
  difference does not establish that the methods are equivalent.

- [Historical two-pair snapshot](figures/walker-two-pairs/summary.json): preserved
  to show how the interpretation changed when the final pair arrived.
- [Historical first walking pair](figures/walker-first-pair/summary.json) and
  [its original figure](figures/walker-first-pair/learning-curves.pdf).
- [First reference's native records and configuration](data/first-reference):
  100k training steps, ten fixed evaluation seeds, no best-checkpoint selection.
- [Matched control's native records and configuration](data/first-control):
  the same crops and training budget, without the extra image-matching update.
- [Second CURL seed's native records and configuration](data/second-reference):
  another completed training run with the same scientific settings.
- [Second matched control's native records and configuration](data/second-control):
  completes the second pair, without selecting a favorable checkpoint.
- [Third CURL seed's native records and configuration](data/third-reference) and
  [its matched control](data/third-control): both complete at the fixed endpoint.
- [Machine-readable three-pair summary](data/three-pairs-summary.json): the
  complete original cohort, with individual curves and paired differences.
- [First completed 500k CURL run](extension-data/recovered-curl-seed123) and
  [dated continuation summary](extension-data/first-500k-summary.json): one
  trained model at the earlier 20:08 cutoff, before its control finished.
- [Completed longer control](extension-data/recovered-control-seed123),
  [first completed pair's summary](extension-data/first-500k-pair-summary.json) and
  [paired curve](figures/first-500k-pair.pdf): the 20:50 comparison, with CURL's
  additional recovery cost reported explicitly. To rebuild its figure, run
  `python plot_first_500k_pair.py` here with Matplotlib installed.
- [Second completed longer CURL model](extension-data/curl-seed456-500k) and
  [21:53 continuation snapshot](extension-data/second-curl-500k-summary.json):
  866.14 at the fixed 500k endpoint; this earlier snapshot predates its control.
- [Second completed longer control](extension-data/no-curl-seed456-500k),
  [22:48 two-pair snapshot](extension-data/two-500k-pairs-summary.json), and
  [completed-pair curves](figures/completed-500k-pairs.pdf): different winners
  in the two pairs. Rebuild with `python plot_completed_500k_pairs.py` here.
- [Third completed longer CURL model](extension-data/curl-seed789-500k) and
  [23:46 continuation snapshot](extension-data/third-curl-500k-summary.json):
  855.22 at the fixed endpoint; this historical snapshot predates its control.
- [Final completed control](extension-data/no-curl-seed789-500k),
  [complete six-run summary](extension-data/three-500k-pairs-summary.json) and
  [all three longer curves](figures/three-completed-pairs/completed-500k-pairs.pdf):
  the complete cohort. Rebuild the current figure with
  `python plot_completed_500k_pairs.py --summary extension-data/three-500k-pairs-summary.json --output figures/three-completed-pairs`.
- [Failed longer-run records](extension-data/failed-seed123) and
  [cleanly stopped control](extension-data/stopped-control-seed123): preserved
  execution evidence, not completed 500k scores. The [recovery plan](RECOVERY_PLAN.md)
  explains the saved-state choice and the extra physical training cost.
- [Actual pilot input images](figures/pilot-observations.png): first, middle and
  last stored observations, each containing three frames. Top row: raw frames;
  bottom row: center crops used during evaluation. They were not selected for
  attractive task performance. Training uses random rather than center crops.
- [Ranked next comparisons and primary research](NEXT_COMPARISONS.md): the
  augmentation-only question has prior work; this is a learning reproduction,
  not a novelty claim.
- [Walker/walk protocol](WALKER_PROTOCOL.md): all three original pairs complete.
  It tests whether the early benefit also appears when fresh
  models learn to walk; it does not transfer the cartpole model's weights.
