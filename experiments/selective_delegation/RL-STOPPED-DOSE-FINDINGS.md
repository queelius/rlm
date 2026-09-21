# Stopped RL dose: no demonstrated gain from five additional updates

Checkpoint 21 scores **54/128**, versus checkpoint 16's **56/128** on the fixed fresh003
panel. The paired EM difference is −1.56 percentage points, with a 47-component-cluster
95% bootstrap interval of [−5.88, +2.63]. F1 changes from 0.52450 to 0.51713:
−0.74 points, interval [−4.62, +3.22]. This is not a useful demonstrated improvement;
it also does not establish harm. The evidence does not justify another unchanged
root-training continuation. The accepted execution-credit diagnostic can inform a
different objective or architecture; it should not be treated as permission for more updates.

## Coverage, ancestry, and cost

Both policies have all 128 planned outcomes observed, with no missing outcomes, failed
native calls, unresolved starts, extra attempts, unlinked calls, or unknown token/latency
measurements. Checkpoint 16 has 128 valid finals. Checkpoint 21 has 127 valid finals and
one observed invalid dependency: question 2 references itself as `#2` on parent
`8266c125c61d7378c843f0a6`, repeat 0. Checkpoint 16 was also incorrect on that slot;
the protocol defect contributes no EM loss in this comparison.

All three wins and five losses are **both-valid** comparisons on eight different parents.
An independent reread regraded all 255 actual final receipts and reproduced 56 versus 54;
native request digests were checked. A second check verified all 2,526 report-bound files
at most 2 MB, including native receipts, source, plans, checkpoint metadata, and terminal
receipts. Large weight/optimizer hashes were not recomputed: their binding comes from
the completed comparator's checkpoint audit. The evaluation terminal is clean, not capped.
No partial evaluation outcomes were inspected for this review.

The ancestry audit binds committed checkpoint 21 to the compared checkpoint 16. Training
stopped at batch 22 with `admission_failed_no_update`, optimizer step 21, five additional
committed updates, no failure, and no unresolved starts. Checkpoint 21 was selected as
the last committed checkpoint under that rule **before its development outcomes**, not
as a best checkpoint or a relabeled checkpoint 24. Batch 22's flat within-parent rewards
despite distinct valid plans remain a local sampling observation, not convergence.

| Readout | Native calls | Input + output tokens | Tokens/attempt | Summed call time |
|---|---:|---:|---:|---:|
| Checkpoint 16 | 563 | 1,353,578 | 10,574.8 | 636.7 s |
| Checkpoint 21 | 550 | 1,313,583 | 10,262.4 | 621.7 s |

Checkpoint 21 uses about 3.0% fewer tokens and 13 fewer calls, but is not more accurate.
There are 307 versus 296 planned helper steps, and 307 versus 295 actually executed
helpers; the invalid dependency prevents one helper and its final. Root calls remain
128 each. These are readout costs, not the continuation's additional 1,612-call training cost.

Descriptively, two-hop EM changes 29→30/64 and three-hop EM 27→24/64. Planned helper
steps change 128→127 on two-hop and 179→169 on three-hop. This is consistent with some
shorter plans, not proof that shortening caused the score change. Word choice, dependency
structure, and per-helper token cap (`384 / number of steps`, rounded down) change together.
The panel is deliberately balanced 32 two-hop/32 three-hop parents, not a natural-dev mix.

## All EM-changing parents, with both repeats retained

Entries are exact-match results in repeat order `[0, 1]`. These eight cases were inspected
because their measured outcomes changed; they are illustrative, not a representative sample.

| Parent ID prefix | Question topic | Checkpoint 16 | Checkpoint 21 |
|---|---|---|---|
| `ae310f50` | ECB mission location | [0, 0] | [1, 0] |
| `7311d7df` | Surzhyk / Russian leader meeting | [0, 0] | [0, 1] |
| `1dcb55ef` | Hayek / veto power | [0, 0] | [1, 0] |
| `99b7998f` | Memoirs of a Geisha / Cold War nation | [1, 1] | [1, 0] |
| `0fd10e6a` | Spain scorer / 2009 club wins | [0, 1] | [0, 0] |
| `a721d202` | Man in Space Soonest / population | [0, 1] | [0, 0] |
| `3661554d` | Stalin / published pact account | [1, 0] | [0, 0] |
| `5846d254` | The Hobbit episode / Stan's voice | [1, 1] | [1, 0] |

### A useful wording change, but not a stable two-repeat improvement

For `ae310f507d323d8fec7e50f8`, repeat 0, both plans first ask which agency controls EU
monetary policy and receive `ECB`. The second question changes from
`Where is #1 mission located?` to `Where is #1 mission found?`. The helper and final
change from `Frankfurt` to `Article 2 of the Statute of the ECB`, the annotated answer.
The supplied ECB document explicitly names Article 2. This is a concrete lexical
disambiguation mechanism without an extra helper step. However, repeat 1 still answers
`Frankfurt` under both checkpoints, including checkpoint 21's similarly worded question.
One favorable sample does not show reliable disambiguation learning.

### Shorter or less specific plans can lose the relation being asked about

For `5846d254a9a963cedb2a34a2`, repeat 1, checkpoint 16 produces three questions,
including identifying the series containing the episode. Its helper sequence is
`Trey Parker` → `South Park` → `Denver`, followed by scored-correct `Denver`.
Checkpoint 21 emits just `Stan on The Hobbit >> voice actor` and `Where was #1 born?`.
The helpers return `Richard Crispin Armitage` → `England`, and the final repeats
`England`. The supplied documents contain both the South Park cast and a Richard
Armitage biography. This trace is consistent with film/episode ambiguity after a
changed plan, not proof that three steps are generally required. Repeat 0 is identical
across checkpoints and correct, even though its first helper identifies the film trilogy;
the subsequent question already explicitly says South Park.

There is a related loss on `0fd10e6ac698d0eb58569b29`, repeat 1: checkpoint 21 removes
the team-identification question and asks what wins David Villa achieved in 2009;
the final changes from `the continental treble` to `none`. Conversely,
`3661554dd1e0261b6dfad031`, repeat 0 loses despite retaining three steps: its last
question drops the specified person's version of the pact, and helper/final change
from `1948` to `1917`. Thus the inspected losses do not reduce to step count alone.

### Correct finals do not certify correct execution

For `7311d7df1cb80bf414ba19d3`, repeat 1, checkpoint 21's last helper answers
`the first secretary of the Communist Party of Ukraine`, but its full-source final
answers the scored-correct `Pope John Paul II`. The saved plan also refers to visiting
the second answer, a date, rather than the country. This is a final-stage rescue or
bypass, not evidence of a faithful improved chain. Repeat 0 is incorrect for both.

The Hayek/veto win has an additional metric caveat. Checkpoint 21's repeat-0 plan
expands from three to four steps, but the last helper answers `the United Kingdom`.
The final answers `the United Nations Security Council`, which earns EM because the
official aliases include that organization, although the primary annotation is
`permanent members of the United Nations Security Council`. Repeat 1 is incorrect
under the official aliases for both checkpoints. Preserve official scoring, but do not
present this win as verified reasoning about exactly who holds veto power.

## Limits and provenance

This is an exploratory repeated readout on an already exposed panel. Five more optimizer
updates also add second-pass training exposure; there is no isolated update-dose effect.
The root checkpoint is the intended intervention, while helper36, base final, documents,
seed schedule, prompts, caps, and source contracts remain matched. Common seeds do not
remove downstream stochasticity when the root changes the prompt. The bootstrap intervals
are unadjusted and wide. Nothing here establishes a general limit of reinforcement learning.

The immutable report is
`R/analysis-fresh-rl-stopped-dose-001.json`, SHA-256
`1acceff3a5d951927b353e3f10b50f7d33f83dbdda6db923f8cbfaaa8fe090e1`.
`R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Its `input_source_checkpoint_sha256` map binds cases, sources, checkpoints, and consumed
receipts. The cases hash is
`6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153`.
Trace locations are `fresh-contract-rl-001` and `fresh-contract-rl21-001`, with native
call names `<full parent ID>-r<repeat>-rl-isolated-{root,helper-N,final}.json` in `calls/`.
For the Hayek and Surzhyk examples, the full IDs are respectively
`1dcb55ef13a90129633ee9e2` and `7311d7df1cb80bf414ba19d3`.
The completed source report was not edited; only this human findings document was added.
