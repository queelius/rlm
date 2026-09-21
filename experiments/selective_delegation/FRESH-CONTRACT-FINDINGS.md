# Fresh fixed-helper readout: small, uncertain root-RL difference

Root RL finished at **56/128** exact answers versus **53/128** for its SFT48
starting policy. The paired difference is +2.34 percentage points, with a
component-cluster 95% interval of **−1.67 to +6.25 points**. This does not
establish an RL gain. Direct base answered 54/128 using about 28% of the
planner policies' inference tokens. None of these results establishes a
decomposition advantage over direct answering.

## Panel and fixed contract

`fresh-dev-inputs-003` contains 64 development parents, deliberately balanced
32 two-hop/32 three-hop, with two repeats per parent. It is not an estimate of
the natural full-development benchmark mixture. There are 47 connected
atomic-component clusters. All planned attempts remain; no missing outcomes,
failed native calls, unknown usage, or unlinked calls were reported.

All planner arms use the same title-index-only root observation, strict
question-list interface, frozen helper SFT36 on isolated questions, and the
unadapted base full-source final. “Planner base” means the **root** has no
adapter; its helpers still use the selected trained helper. SFT is checkpoint48;
RL is `rl-fullpass-001/checkpoint-0016`, warm-started from SFT48. The final sees
original documents plus the generated plan and actual helper trace, so it can
override or bypass the plan. Root/helper-total/final generation caps are
128/384/128; isolated helper cap is floor(384 / generated plan length).
Matched case/repeat seed mapping and T=0.5 are preserved across policies.

This is a frozen development readout within an adaptive research campaign, not
confirmation of generalization. Exact TRAIN paragraph overlap occurs in 14/64
parents (including distractors); the no-exact-overlap group is not guaranteed
semantically unseen. The separate predeclared hop/exposure analysis must keep
all64 primary and acknowledge the strong hop/exposure imbalance.

## Answer quality and actual inference cost

Official MuSiQue alias-max EM/F1; 128 planned attempts per policy.

| Policy | Exact | EM | F1 | Valid finals | Calls | Total tokens | Tokens/attempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Planner base + helper SFT36 | 46 | 35.94% | 44.14% | 119 | 571 | 1,378,310 | 10,768.0 |
| Planner SFT48 + helper SFT36 | 53 | 41.41% | 51.04% | 127 | 550 | 1,310,292 | 10,236.7 |
| Planner RL16 + helper SFT36 | 56 | 43.75% | 52.45% | 128 | 563 | 1,353,578 | 10,574.8 |
| Direct base | 54 | 42.19% | 48.64% | 128 | 128 | 373,440 | 2,917.5 |

Base root had eight invalid plans and one invalid dependency; SFT had one
invalid dependency; RL had none. There were no helper-protocol failures. Root
calls are 128 per planner; helper calls are 324/295/307 for base/SFT/RL; final
calls are 119/127/128. Direct has 128 final calls only. RL costs 3.30% more tokens
than SFT. Direct costs 28.50% of SFT tokens and 27.59% of RL tokens. These are
measured inference costs, excluding training. Summed native-call latencies are
666.96/621.41/636.72/79.37 seconds in table order, not end-to-end job wall times.

Paired 20,000-draw, parent-weighted connected-component bootstrap, seed
2026092115:

| Contrast | EM difference, 95% interval | F1 difference, 95% interval |
|---|---|---|
| RL − SFT | +2.34pp [−1.67, +6.25] | +1.41pp [−3.39, +5.93] |
| Direct − SFT | +0.78pp [−7.89, +9.23] | −2.40pp [−11.20, +5.99] |
| Direct − RL | −1.56pp [−9.65, +6.15] | −3.81pp [−11.42, +3.41] |

RL versus SFT has five wins and two losses, each on a different parent; all
seven changes have valid finals on both sides. The aggregate gain is therefore
not a returned-protocol recovery. However, “both valid” only means the output
contracts were satisfied, not that intermediate reasoning was correct.

## All seven changed parents, including the other repeat

Entries show final strings, SFT → RL. Repeats are zero-indexed; an unchanged
correctness result is retained even when the answer strings differ.

| Parent / question shorthand | Repeat 0 | Repeat 1 |
|---|---|---|
| `0fd10e6ac698d0eb58569b29`, Spain scorer's team's 2009 wins | Same list of La Liga, Copa del Rey, UEFA Champions League; both EM-wrong | `none` → `the continental treble`; RL win |
| `3661554dd1e0261b6dfad031`, Stalin's pact-version publication | `1917` → `1948`; RL win | `1917` → `1917`; both wrong |
| `5846d254a9a963cedb2a34a2`, Stan voice actor's birthplace | `England` → `Denver`; RL win | `Denver` → `Denver`; both correct |
| `99b7998f2d7f7b3ee4b175d6`, Geisha/runways conflict's opposing nation | `Soviet Union` → `Soviet Union`; both correct | `Japan` → `Soviet Union`; RL win |
| `a721d202f78bfab4bb4e8792`, Man in Space Soonest collaborator's population | `two million` → `29`; both wrong | `29` → `6 million`; RL win |
| `0877fe5b9e7345e59ebda083`, Lloyd Dane birthplace's county | `Miller County` → `Miller County`; both correct | `Miller County` → `Dane County`; RL loss |
| `8126620129bc38833104a288`, Girlfriend performer's Wire role | `Bobby Brown` → `Bobby Brown`; both wrong | `Western District uniformed officer` → `The Blind Boys of Alabama`; RL loss |

All five winning parents are three-hop and both losing parents are two-hop.
This is a descriptive property of the changed cases, not an established
hop-by-training interaction or a replacement for the complete stratum analysis.

### Illustration 1: retaining the requested relation can help, inconsistently

For the Molotov–Ribbentrop parent (`366155…`, repeat 0), both policies' first
two helpers answered `Joseph Stalin` and `Stalin`. SFT's third question was
`#1 >> published in`; after binding, `Joseph Stalin >> published in` elicited
`1917`, also the final. RL asked `what year was the #2 version of the
molotov-ribbentrop pact published`; its helper and final answered `1948`.
The supplied paragraph 10, “Molotov–Ribbentrop Pact,” discusses the 1948
publication and Stalin's responding version. This is consistent with a more
specific question preserving the requested relation, under the same three-call
helper allocation—not merely adding another helper.

The other repeat prevents a stronger story. SFT's more explicit publication
question still elicited `1917`. RL emitted only two questions and bound
`Joseph Stalin` into a phrase asking about a version “published in #1”; the
helper answered `Vladimir Lenin`, while the final again answered `1917`.
One local success is not stable relation-binding competence.

### Illustration 2: a root can also add a wrong constraint

For Lloyd Dane (`0877fe…`, repeat 1), the same first helper returned `Eldon`.
SFT asked `What county is #1 in?` and obtained `Miller County`. RL added
`in Wisconsin`, yielding `What county is Eldon in Wisconsin?`; its helper and
final answered `Dane County`. The public documents explicitly identify Eldon,
Missouri, in Miller County (paragraphs 0 and 18). The extra state constraint is
unsupported by those relevant paragraphs and is consistent with steering the
helper away from the correct referent. Both use two helpers with the same caps.
Repeat 0 is identical across policies and correct, despite an imprecise first
helper answer `Missouri` followed by “Which county does Missouri belong to?”

### Contradictions: correct finals are not faithful-chain evidence

Several RL wins cannot be presented as cleanly repaired chains:

- Stan/The Hobbit repeat 0: both first helpers incorrectly identify the series
  as Peter Jackson's film trilogy. RL's next question explicitly names South
  Park, bypassing that result, then obtains Trey Parker and Denver.
- Geisha repeat 1: SFT's last helper says `the Soviet Union`, yet its final
  says `Japan`. RL has `Kyoto` and `Cold War` as its two helper answers, then
  produces the correct `Soviet Union` final. Repeat 0 is correct for both even
  though `Kyoto` is bound where a conflict/setting relation was intended.
- Population repeat 1: RL follows US Air Force → NATO → Belgium → `10 million`,
  but the full-source final outputs the gold `6 million`. Repeat 0 is wrong for
  both. The correct final does not validate the displayed population chain.
- Spain/2009 repeat 1: RL binds `Spain` as the team, but the next helper answers
  `the continental treble`. The final matches the reference while the helper
  chain's team referent remains problematic. Repeat 0's competition list is
  EM-wrong; this metric difference alone does not establish a reasoning failure.

These observations motivate the already-accepted frozen-execution and TRAIN-fit
diagnostics: distinguish learnability, helper response changes, and final
source-based rescue. They do not justify declaring an RL/generalization win or
training a broader recursive architecture from this small difference alone.

## Evidence and rendering correction

Canonical numerical source:
`R/analysis-fresh-contract-policy-001.json`, SHA256
`63a0a9d86b3994b94017e45a09b120ad850ed560b81970f0f610b274e2b8234e`, where
`R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
The report records native provenance for `fresh-contract-sft-001`,
`fresh-contract-rl-001`, and `fresh-contract-direct-001`. Case SHA256:
`6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153`.
For this qualitative audit, all 28 SFT/RL episodes across the seven parents and
both repeats, and their 126 native calls, were hash-checked against that report.
Helper answers were checked against native JSON and final EM regraded. Thus
the examples use actual saved executions, not reference decompositions.

The immutable report Markdown incorrectly starts **“Four-hop MuSiQue”** because
the sealed renderer inherited a hardcoded title. Its numerical panel and source
identities are correct: this is the fresh two/three-hop panel, not the exposed
four-hop transfer panel. The sealed source and report were not edited. The
working-tree renderer now uses “MuSiQue: end-to-end policy comparison”; a focused
fixture first failed on the old title and passes after the change. Both focused
comparison tests and Ruff check/format pass. No models were run for this audit,
and no outcomes from the not-yet-read direct-adapted control were inspected.
