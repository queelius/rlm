# Learning RL by reproducing CURL

Start with [the ten-page PDF](learning-guide.pdf). The
[LaTeX source](learning-guide.tex) and Makefile are next to it.

This edition includes **all three completed pairs at both 100k and 500k**,
plus two completed walking pairs, with an October 5, 05:43 UTC evidence cutoff.
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
with limits and recovery cost. A fresh-model comparison on a walking task is
now running to test the scope of the early finding. Its first two pairs finished
at **482.83 versus 241.28** and **385.23 versus 360.96**, with CURL first.
Both favor CURL, but the second advantage is much smaller. Page 10 shows all
four complete curves. The third pair is running. We do not yet know how stable
the benefit is, or whether longer training would let the control catch up.
Published scores and invented teaching
examples are identified explicitly.

The [running findings](FINDINGS.md) explain the latest comparison and preserve
the earlier interpretations as dated checkpoints.

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
The ten-page edition includes all three matched cartpole pairs at 500k and the
first two walking pairs. Updated pages 1, 7 and 10 were visually inspected after
compilation; earlier unchanged pages retain their previous inspection. The
current build has no overfull/underfull messages, and text stays inside all
ten page boundaries. No font or plot shrink was needed. The numerical
teaching example was independently recalculated. The manifest parses as JSON,
and its decision count times action repeat equals the stated environment budget.

PDF SHA256:
`48a752b97ec8f1178a9515ce1c7a9ff36c5c2a21ada7141a11e4ec801eccea08`.
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
For the two walking pairs, root and an independent CPU audit recomputed all
104 means from 1,040 episode records, checked the ten declared starts, native
endpoints, finite learning values and paired configurations. All 16 public
record files match their external originals byte for byte. The four checkpoints'
sizes were checked against receipts across the completed reviews.
The existing summarizer was reused without a code change. Its previous plot-title
addition passed a failing-first rendered-PDF regression, all seven focused
tests and Ruff checks. This data/document update required no new test campaign
or live training-source change. The prior first-pair figure remains preserved.

## Evidence and next comparisons

- [First walking reference](walker-data/first-reference) and
  [matched control](walker-data/first-control), plus
  [second reference](walker-data/second-reference) and
  [second control](walker-data/second-control): two completed pairs at fixed 100k.
- [Current walking paired summary](figures/walker-two-pairs/summary.json) and
  [learning curves](figures/walker-two-pairs/learning-curves.pdf).
  Rebuild from the closed copied records with:

  ```sh
  python experiments/curl_replication_20261004/summarize.py \
    --runs docs/curl-replication-2026-10-04/walker-data \
    --output docs/curl-replication-2026-10-04/figures/walker-two-pairs \
    --title 'Learning to walk from images: two training pairs'
  ```

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
- [Walker/walk protocol](WALKER_PROTOCOL.md): now running after the cartpole
  queue completed. It tests whether the early benefit also appears when fresh
  models learn to walk; it does not transfer the cartpole model's weights.
