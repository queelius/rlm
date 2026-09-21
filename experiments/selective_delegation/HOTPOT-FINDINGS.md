# Hotpot transfer: SFT planning changed both protocol and content

Source: official cached-Hotpot regrade in
`analysis-transfer-hotpot-001/REPORT.json`, over 32 frozen transfer parents x
two repeats. These are the authoritative Hotpot EM/F1 values, not the native
MuSiQue diagnostic scores retained in evaluation receipts. The direct-answer
control is from `analysis-transfer-hotpot-direct-001/REPORT.json` on the same
questions and repeats. These 32 questions come from the official website's
100-question explorer sample, not the full development set.

| Root condition | Planned | Official EM | Official F1 | Protocol failures |
|---|---:|---:|---:|---:|
| Base | 64 | 30/64 (46.9%) | 51.9% | 17 |
| MuSiQue plan SFT | 64 | 23/64 (35.9%) | 45.4% | 15 |
| MuSiQue plan RL | 64 | 24/64 (37.5%) | 46.4% | 15 |
| Direct answer, no planner/helpers | 64 | 36/64 (56.3%) | 65.5% | 5 |

For base versus SFT, base-only exact successes are 10 and SFT-only successes
are 3; both are correct in 20 slots. Eight base-only and all three SFT-only
slots involve a protocol failure. The remaining two base-only successes had
valid final answers in both conditions, so the 7/64 EM difference is not
explained solely by missing JSON/finals. This is one frozen, exploratory
Hotpot sample with two repeats, not an estimate of SFT's general effect. The
direct-answer control is descriptively strongest; no paired uncertainty interval
for that Hotpot contrast has been computed here. The supplied documents fit into
one model call, so this does not test the advantage of an RLM on inputs that
cannot fit in the model's context window.

## Trace examples (descriptive, not causal proofs)

- **Repeated helper-interface loss:** `bfa07...` asks where Mike DiNunno's team
  is based. Base used two explicit questions and returned the official answer
  on both repeats. SFT's title-style first question found the team, but its
  second helper response was invalid on both repeats, so no final ran. This is
  consistent with a plan-language/interface mismatch, not a demonstrated model
  incapacity.
- **Repeated multi-entity loss:** `75f9...` asks whether two people are both
  Armenian-American. Base asked two direct yes/no evidence questions and was
  correct twice. SFT emitted three shorthand steps including a final comparison;
  helper parsing failed twice. This is another protocol-associated regression.
- **Both-scored content loss:** `d19c...` (Ralf D. Bode/Loretta) was correct
  under base. SFT's first helper identified the relevant film, then its second
  question switched to a different Loretta and the final returned that wrong
  entity. Both runs reached a valid final, localizing this example to erroneous
  entity/chain selection rather than answer formatting.
- **Both-scored content loss:** `28e9...` (Black Crusade designer) had the
  correct designer in both traces, but SFT's second helper supplied a broader
  related franchise instead of the required game; base returned the official
  answer. This is a wrong helper fact/use conditional on the SFT plan, not a
  final JSON failure.
- **Protocol gains also exist:** `2a24...` was a base invalid-plan zero but an
  SFT two-step chain with a correct final; `31b4...` similarly includes a base
  invalid-plan repeat and an SFT correct repeat. They caution against treating
  SFT as uniformly harmful.

## Interpretation and limits

SFT changes the root's question wording and chain structure. Its lower official
score is compatible with (a) title/relation shorthand being less robust for the
frozen isolated-helper contract, and (b) occasional semantically wrong chain
selection even when parsing succeeds. The traces do not distinguish whether
those differences are learned-plan effects, checkpoint-specific sampling, or
their interaction with an out-of-family Hotpot wording/distribution. They do
not show that a shorter chain is intrinsically better: base has invalid plans,
and SFT has real wins.

The original cached explorer records have `_id`, title/paragraph fields,
question, answer, supporting facts and distractors, but no bridge/comparison
type label. Therefore the examples above are described by observed trace
behavior only; no question type is inferred from wording or used for scoring.
All frozen cases remain in the official planned denominator, including protocol
zeros. This panel has now been examined and is no longer an untouched test set.
Later comparisons on it must be labeled exploratory; a fresh panel should be
used to evaluate choices made from these results. Report protocol and both-scored
content changes separately. The predeclared one-epoch SFT checkpoint readout is
now complete; it is an exploratory dose check, not a search for a better-looking
primary checkpoint.

## Early versus full SFT dose: no early-checkpoint rescue

The same 32 parents and two repeats were evaluated at update 16 (one epoch) and
update 48 (three epochs). The source009 collector, case hash, model, sampling
seeds, caps and isolated execution contract match. Only the root adapter
checkpoint changes; helpers and finals remain the released base model. Native
finals were independently regraded using the official Hotpot scorer.

| Fixed checkpoint | Correct / planned | Official F1 | Valid finals | Protocol failures |
|---|---:|---:|---:|---:|
| SFT16, one epoch | 20/64 | 34.15% | 35/64 | 29 |
| SFT48, three epochs | 23/64 | 45.42% | 49/64 | 15 |

Full dose minus early dose is +4.69 EM points, with an exploratory paired-parent
95% interval of −9.38 to +18.75. F1 changes by +11.27 points, interval −4.18 to
+26.36. These intervals use 20,000 parent-bootstrap draws, seed 2026092116;
repeats stay together. Hotpot atomic-component metadata are unavailable, so this
is not a verified independent-component uncertainty estimate.

SFT48 wins nine attempts on seven parents, all involving protocol recovery. It
loses six attempts on four parents: five protocol-related and one with valid
answers in both conditions. The early model has 27 invalid-dependency episodes
and two invalid-helper episodes; the full-dose model has no invalid-dependency
episodes and 15 invalid-helper episodes. Thus later training improves one
execution interface while failures remain elsewhere. For example, an early plan
for Lars Lunde consisted of one question containing `#1` and `#2`: neither answer
exists when the first question is executed.

The sole both-valid scored regression is also answer-granularity-sensitive:
parent `1a849a6f3ce9cf54dfc17685`, repeat 1, returns the reference `Runaways` at
SFT16 but `Runaways vol. 2 #7` at SFT48. Official EM remains unchanged; the more
specific response should not automatically be called a semantically unrelated
answer.

The early checkpoint does not recover the released planner's 30/64 performance.
This weakens the simple explanation that the additional two SFT epochs alone
caused the observed transfer loss. It does not establish that SFT is free of
overfitting, that more training is always beneficial, or that the dose difference
is reliable on other panels. No new primary checkpoint is selected: the fixed
SFT48 warm start remains unchanged for the accepted next training run.

All 441 selected-SFT native calls returned, with no missing outcomes or unknown
usage. SFT16 used 197 calls and 240,055 tokens; SFT48 used 244 calls and 316,917
tokens. This is an actual-cost comparison, not a compute-matched one. Reproducible
source: `audit_hotpot_dose.py`. Immutable paired audit and source/receipt hashes:
`analysis-hotpot-dose-001.json/.md` in the external study root. Original official
reports remain at `analysis-sft-epoch1-hotpot-001/REPORT.json` and
`analysis-transfer-hotpot-001/REPORT.json`.
