# Helper adaptation improves this frozen-plan development panel

Completed September 21, 2026. Exploratory comparison on 32 development parents,
two saved root plans per parent, with 29 connected atomic-component clusters.
This is not a fresh test set or a result about general decomposition ability.

## Result and next fixed contract

Use the fixed helper-SFT checkpoint at update 36 for the next root-learning
comparison, retaining the existing helper prompt without the extra reminder.
Keep the base final model and its full-source prompt unchanged. Compare any new
root policy against SFT48 under that same frozen helper contract; do not attribute
a difference from the old base-helper runs to root learning alone.

| Helper policy | Correct / planned | EM | F1 | Invalid-helper episodes | New calls | New tokens |
|---|---:|---:|---:|---:|---:|---:|
| Released base | 19/64 | 29.69% | 39.28% | 11 | 193 | 565,428 |
| Helper SFT36 | 28/64 | 43.75% | 52.07% | 0 | 210 | 618,721 |
| Base + JSON reminder | 22/64 | 34.38% | 44.87% | 0 | 210 | 620,519 |

The primary trained-minus-base difference is **+14.06 EM points**, with a
component-cluster bootstrap 95% interval of **+3.12 to +26.56**. F1 improves by
12.79 points, interval +0.38 to +25.74. Ten attempts improve and one worsens:
five wins involve recovery from a protocol failure; five wins and the one loss
have valid final answers in both arms.

The JSON reminder also removes all 11 base-helper protocol failures, but its EM
gain is smaller: +4.69 points, interval −2.70 to +13.79. Its four wins over base
are all protocol-related, with one both-valid loss.

Trained helpers outperform the reminder by six net answers: seven wins on six
parents and one loss, all with valid final answers in both arms. The trained-minus-
reminder EM difference is +9.38 points, interval 0.00 to +20.31; F1 is +7.20,
interval −2.83 to +17.95. The point estimate favors training beyond adding this
format instruction, but this small secondary comparison is not decisive evidence
of general semantic improvement. Its interval reaches zero, and some end-to-end
successes occur despite incorrect intermediate answers.

All intervals use 20,000 paired connected-component bootstrap draws, seed
2026092113, preserving parent weighting and the two repeats within each parent.
The comparisons are exploratory and unadjusted.

## What was held fixed

The exact same 64 previously generated SFT48 root plans were reused in all three
arms. Helpers answer one resolved generated question at a time from the full
public documents. References bind actual earlier helper predictions, never
annotated answers. The total helper output budget is 384 tokens, divided by the
number of generated questions; each final has 128 tokens. Temperature is 0.5,
with common per-step seeds across arms.

Only the trained-helper arm enables the separate helper adapter, and only during
helper calls. All finals disable adapters. The reminder arm changes only the
helper prompt by appending:

`Return ONLY a JSON object with one string field named answer.`

The helper adapter was trained for one fixed epoch, 36 updates, on 570 annotated
steps from 256 train parents. Training used annotation questions and gold previous
answers to resolve dependencies; deployment uses generated plans and predicted
answers. This teacher-forcing gap remains. There was no validation checkpoint
search in this comparison.

## Receipt-grounded examples and limits

- **A clean repeated answer change:** parent `2e758fe476c0307441cf8d82`, both
  repeats, asks when the death-place of the author of *Paenitentiam Agere* became
  a country. All arms identify Pope John XXIII. The base and reminder helpers
  answer “never” to the dependent question; the trained helper answers
  “11 February 1929,” matching the supplied Vatican City passage and reference.
  Finals retain those respective answers. These are two repeats of one parent,
  not two independent examples.
- **A protocol-only recovery:** parent `f53fc0e37b69553d8db1a4ec`, both repeats,
  asks what month Richard left the Holy Land. The base helper returns the bare
  string `October`, so strict parsing correctly stops the episode. Trained and
  reminder helpers return valid answer JSON and finish correctly. This is useful
  execution repair, not evidence that training taught the historical fact.
- **A correct final is not proof of a correct chain:** parent
  `ff160993a96adb267736388d`, repeat 1, has a plan that asks when Pope John XXIII
  died and then treats that date as a city. The trained last helper answers
  “11 April 1963,” yet the full-source final returns the reference
  “11 February 1929.” The base/reminder finals answer “never.” This is an
  end-to-end win, but not successful execution of a semantically correct plan.
- **The one trained-arm loss is reference-sensitive:** parent
  `dca7f00bfe686119d74f3c4f`, repeat 0, asks for the body of water near George
  Mills' birthplace. All arms infer Deptford. Base/reminder answer the reference
  “River Thames”; trained helper/final answer “Deptford Creek.” The supplied
  Deptford paragraph names both water bodies alongside the area. Official EM
  remains unchanged, but this should be called a scored regression rather than
  an unambiguously false geographic answer.
- **The final can also rescue a trained helper:** parent
  `533c4e21119afe9c2ff19644`, repeat 1, concerns the formation year of the group
  that performed *Attics to Eden*. The trained second helper answers “Chicago”
  to `Madina Lake >> formed`; the final nevertheless recovers “2005.”

The narrow [annotation-compatible first-step audit](HELPER-ANNOTATION-COMPATIBLE.md)
scores only21/64 generated first questions with literal unique reference matches.
Base/reminder score14/21 and trained11/21; the three losses represent one
reference-granularity disagreement and one clear factual error repeated twice.
This selected subset cannot estimate general intermediate accuracy, but reinforces
the need to distinguish better final answers from reliably better helper steps.

The two Vatican-related parents share an atomic component; the clustered analysis
keeps them together. Outside the literal-compatible subset above, no annotated-step
accuracy was assigned to generated questions. Exact helper answer strings changed at 55/129 both-valid step pairs
between trained and base; 18 compared steps had different resolved questions
because upstream predictions changed. These are agreement diagnostics, not
ground-truth intermediate reasoning scores.

## Cost, verification, and research decision

The collector completed all 192 planned episodes and 613 new native calls, with
no unavailable outcomes, unresolved starts, unlinked calls, or unknown usage.
The new calls used 1,799,174 prompt plus 5,494 completion tokens. Historical root
acquisition is excluded from these totals. More complete execution increases
trained-versus-base cost; trained and reminder have nearly identical token totals.
Recorded call-service time was 113.2 seconds for base, 174.2 for trained, and 130.7
for reminder. Those are observed service totals, not a controlled adapter-overhead
experiment or end-to-end speedup estimate.

The completed receipt analyzer reconstructed prompts, checked frozen-root and
sampling identities, reparsed helper outputs, and regraded native finals. An
independent pass checked adapter routing on all 613 calls and inspected complete
triples. A preliminary read while collection was active encountered a snapshot
race; the completion-gated analyzer then ran successfully without source changes.

This result supports freezing the trained helper before another bounded root-RL
test: it removes a known protocol bottleneck and offers a positive end-to-end
development signal beyond the tested JSON reminder. It does not establish that
remaining reward variation measures planner quality, that full-source finals use
the plan faithfully, or that a larger root-RL dose will help. Invalid root references
also remain outside the helper's ability to repair. A new root-RL readout needs a
matched unchanged-root baseline under the new helper, with independent transfer
and direct-policy comparisons reported separately.

## Evidence

Study root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`

- `analysis-helper-001.json/.md`: complete planned-denominator comparison,
  component clusters, changed-answer rows, receipts and source/checkpoint hashes.
- `helper-eval-001/episodes/` and `calls/`: exact case/repeat/condition examples
  named above, including raw invalid helper answers and base-model final receipts.
- `helper-eval-001/PLAN.json`: frozen contracts and source-root identities.
- `helper-sft-001/checkpoint-0036`: fixed helper adapter; weights SHA-256
  `adb29ee3db8d2f5acd721a99f194f88a3084c191a4596e868dce0b1a56a5735f`.
- `planner-sft-001/checkpoint-0048`: source planner; weights SHA-256
  `5f6da1742ae73b76244c39f256e5918a3b2923b31d619d9741e1a0a9edde4b14`.

Analysis JSON SHA-256:
`e03ef6100d8778c65e29bab71ada247694d312336dc88fe104bd11ec6529e6f9`.
Evaluation PLAN SHA-256:
`983b0bb65dd7b6ac5455a3f29e3af2a80ab6e39be4d55f0771f491f4190616b1`.
