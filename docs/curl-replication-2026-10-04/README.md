# Learning RL by reproducing CURL

Start with [the nine-page PDF](learning-guide.pdf). The
[LaTeX source](learning-guide.tex) and Makefile are next to it.

This edition includes **three completed 100k pairs and two completed 500k pairs**,
plus the third completed 500k CURL model, with a 23:46 UTC evidence cutoff.
It explains the cartpole task,
reinforcement learning, self-supervised image matching, our comparison, and
what source review found. CURL scored 678.02 versus 454.47 in the first pair,
446.17 versus 240.54 in the second, and 587.99 versus 463.26 in the third.
All six models are included in continued training to 500k. The longer comparisons
have different winners: **842.74 versus 866.94** in the first pair, and
**866.14 versus 810.96** in the second (CURL first in each).
The early advantage is more consistent so far than the later one. Two pairs
cannot establish general late harm, superiority or equivalence.
Page 8 preserves the three early comparisons; page 9 shows both completed
longer pairs and explains the limits and recovery cost. The third CURL model
finished at **855.22**, up from 587.99; its control is still training. The
three-seed CURL mean of 854.70 is not a matched comparison with only two
completed controls. Published scores and invented teaching
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

Compiled with Tectonic 0.17.0 using cached dependencies on October 4, 2026,
at approximately 23:48 UTC. The nine-page edition was updated after the third
CURL model finished at 500k. Changed pages 1, 5 and 7 were visually inspected; unchanged pages retain
their earlier inspection. The build reports no overfull
boxes, and extracted text stays within every page boundary. The numerical
teaching example was independently recalculated. The manifest parses as JSON,
and its decision count times action repeat equals the stated environment budget.

PDF SHA256:
`abf6dc792444ec73e672b4f8205b3d42c6c451d217f6b80b9a2c2bf39c09b5be`.
The paired-figure code passed eight focused selection/endpoint tests, Ruff checks
and independent code review. It uses the recorded joined curves without
smoothing and excludes incomplete seeds and the abandoned branch.
The runtime checks include real simulator behavior, CPU checkpoint restoration,
and a completed GPU pilot. Three full matched pairs also completed, but this is
still a small exploratory comparison, not a reproduction of the paper's mean.
All five completed 500k endpoints passed native terminal and ancestry checks.
The three-seed CURL mean and sample SD were recalculated from native final
episode returns. Recovery cost is reported separately. There are still only
two completed longer pairs; the last control remains incomplete.

## Evidence and next comparisons

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
  855.22 at the fixed endpoint; its control remains incomplete. No third paired
  claim follows from this endpoint alone.
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
- [Prepared walker/walk protocol](WALKER_PROTOCOL.md): the next task tests
  whether the early benefit extends to learning to walk. It has not launched;
  the current cartpole queue keeps its original owner and settings.
