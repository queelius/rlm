# A larger reward update learns answer delivery; delegation is still an open question

Exploratory checkpoint: September 12, 2026, 23:45 UTC. This follows the
[earlier checkpoint](2026-09-12-small-rl-signal-and-helper-controls.md); it does not
replace the unsuccessful experiments reported there. New comparisons are still running.

## What changed our view

The reward-training code can teach the model something. With a larger update, it
learned to return the answer it had already found much more reliably on its
training examples. A small improvement also appeared on a previously examined
evaluation panel. We do **not** yet know whether that improvement carries over to
fresh conversations, and we have not demonstrated better problem decomposition.

| Same model before and after one reward update | Before RL | Smaller update | Ten-times larger learning rate |
|---|---:|---:|---:|
| Original training conversations: 8 contexts, 4 samples each | 7/32 | 17/32 | 24/32 |
| Previously examined evaluation conversations: 16 contexts, 2 samples each | 25/32 | 25/32 | 28/32 |

These are exact final answers, including spaces and line breaks. Every planned
outcome was available. Repeated samples of one conversation are not independent
new problems. Both updates began from the same supervised checkpoint and used
the same saved training batch. Each took about 33 seconds of training; collecting
the answers and evaluating the resulting models took several additional minutes.
The larger setting changed the learning rate from 0.00001 to 0.0001, not the
number of training steps. The backward calculations were not bitwise identical,
so this is a recipe comparison, not a perfectly isolated arithmetic intervention.

All 17 gains on training samples preserved the same retrieval program and its
returned information. The model mainly learned to preserve required spaces and
avoid adding a line break. The two training contexts where it searched for the
wrong wording still failed on every sample: the reward update did not teach it
to inspect the wording or devise a different search.

On the examined evaluation panel, there were **five gains and two losses**, not
five gains alone. All five gains followed the same correctly retrieved information;
four repaired edge whitespace and one corrected another copying difference.
One loss failed to produce a usable tool action; another generated a much longer,
incorrect answer despite retrieving the target. The number of samples with clean
retrieval fell from 30 to 29. Output tokens rose from 20,739 to 23,618.

Our interpretation is narrow: update size mattered for learning answer delivery,
but optimizing only the final answer is not a demonstrated route to learning
decomposition. A fixed comparison on 32 unused conversations, balanced across
four requested occurrence positions, is being prepared. Both models will be run,
regardless of the first model's score. This is fresh project data within the same
public task family, not a new task or proof of no pretraining exposure.

## What we learned about helper design

We also tested a small, generated database task with checkable intermediate
answers. A helper must identify every eligible implementation from its local
records; a complete solution must combine compatible choices from three stages.
This separates finding useful options from copying their attributes and combining
them correctly.

The initial full-record interface was too difficult for this model. Even when
given perfect helper reports, the parent returned no fully correct solution in
four cases. We corrected a missing instruction and verified the actual model
service, but failures remained. This is a feasibility result on a candidate task,
not evidence about general recursive reasoning.

The next comparison asked helpers to return only record IDs. Python looked up
their attributes from the public records, retaining exactly the model's choices:
it did not decide eligibility or add missing choices. On 24 matched helper calls,
strictly correct ID sets rose from 2 to 4. However, six lists violated the required
alphabetical ordering. If order is ignored in an explicitly post-hoc diagnostic,
the new interface selected the exact set in 6/24 rather than 2/24. It recovered
44 of 64 eligible IDs rather than 40, with two ineligible IDs selected in each arm.

That distinction matters: an unsorted list is not the same failure as selecting
the wrong records. Nevertheless, ignoring order and letting Python combine the
selected records still produced **0/32 correct combinations**. These combinations
reuse four problems and are not 32 independent problems. Easier formatting alone
did not solve the information-selection problem.

The split-input experiment is now complete: one helper versus two or four, on
nine new stages containing 6, 12, or 20 candidate records, with two samples each.
All 126 helper calls returned. Each condition had the same total output allowance.

| Helpers | Fully correct, including required sorting | Correct ID set, ignoring order afterward |
|---|---:|---:|
| One | 0/18 | 2/18 |
| Two | 0/18 | 0/18 |
| Four | 2/18 | 2/18 |

There is no clear accuracy improvement. With order ignored, four helpers gained
two successes but lost two others. Their selections contained a higher fraction
of eligible records, but missed more eligible records overall: precision rose
from 72% to 81%, while recall fell from 85% to 72%. Input tokens increased by
42%. All 48 invalid individual helper outputs failed only the ordering rule;
none was truncated or invented an out-of-scope ID. These findings argue against
simply adding more helpers. They also motivate scoring the actual information
choices separately from harmless output order. This was fixed splitting, not
a learned decision about whether or how deeply to decompose.

## Where the research should go next

The most promising question is now more precise: **can we reward the decisions
that find useful information, rather than mainly reward how the final answer is
copied?** Our current update gave no learning signal to groups in which every
sample used the same unsuccessful search. Better training examples, useful
alternative actions, or rewards for intermediate information choices may be needed.

We are also opening a different task family: questions requiring calculations
from financial reports. The first small comparison of direct numbers and a
restricted calculation interface needs diagnosis: neither matched the supplied
targets, and all generated calculation programs were invalid. Some supplied
targets themselves appear inconsistent with the question or source. We will not
mistake this for proof that code execution is unhelpful or quietly repair the
targets to improve the score.

For publication, the promising direction is a mechanism study: which failures
belong to information selection, which belong to passing information between
components, and which belong to answer delivery? A strong result would show a
specific intervention improving the relevant component on fresh data, with an
honest cost comparison and an improvement in the whole task. We do not yet have
that complete result.

## Evidence and resumption

The active store is `/project/alex_phd/runs/rlm-research-r4`. Relevant records are:

- `analyses/openai-mrcr-fresh8-rloo-dose10-paired-2026-09-12/readout-001.json`
  and `.md`: all three arms, native response checks, paired changes and costs.
- `sidecars/openai-mrcr-cp32-fresh8-final-rloo-lr1e4-v1/`: immutable training
  specification, optimizer checkpoint, and step commit.
- `analyses/b05-source-visible-native-outcome-v3-2026-09-12/REPORT_V2.md`:
  corrected response-hash interpretation and the qualified full-report run.
- `analyses/b05-eligible-ids-independent-2026-09-12/`: native comparison and
  additive `ORDER_DIAGNOSTIC.json`. Its unordered-set metric is secondary; the
  original strict result remains unchanged.
- `analyses/b05-helper-width-independent-2026-09-12/readout-001.{json,md}`:
  complete split-input comparison, strict and unordered metrics, and costs.
- `sidecars/b05-helper-width-v1/` and
  `sidecars/openai-mrcr-fourneedle-balanced32-transfer-data-v1/`: frozen next
  comparisons. Prepared inputs are not completed results.

The GPU experiments here use the Prime native research harness, not the main
repository's Responses-API runtime. Source and compact reports are periodically
published to [the research notebook](https://github.com/queelius/rlm-research).
GitHub is not a backup of the external model weights or raw trace store.
