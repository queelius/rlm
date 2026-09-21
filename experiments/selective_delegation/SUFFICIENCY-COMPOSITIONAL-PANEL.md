# Compositional sufficiency panel: prospective CPU freeze

The frozen panel deliberately contains 16 three-hop and 16 four-hop MuSiQue DEV parents,
with both official evidence-availability variants retained. It is a higher-hop
new-question diagnostic, not a natural DEV mixture or an unseen-facts evaluation. No model outcomes
were used for selection or inspected during preparation.

Training already includes these depths. The 128-parent paired-RL/extra-SFT input
contains95 two-hop,26 three-hop and7 four-hop questions; the earlier supervised
initializer also included all three depths. Consequently this readout cannot
establish extrapolation beyond training depth. It complements the separately
frozen two-hop panel by testing new questions with more reasoning steps.

Selection takes the first 16 SHA256(`2026092205:` + official parent ID) values within
each hop stratum after exact parent-ID or normalized-question exclusion. The exclusion
inventory contains 15 existing case files, both breadth inventories, the proposed
128-parent sufficiency TRAIN panel, and all official TRAIN parents/questions. Retained
CPU-only panels are conservatively included; this does not imply they produced model
outcomes. No atomic-component, document, answer, or outcome filter was applied.
Eligible inventories were 385 three-hop and 189 four-hop parents.

## Realized exposure and dependence

- 32 parents, 64 variants; evaluation seeds 2026092181 and 2026092182.
- 20 connected atomic-component groups: eleven singletons, six pairs, three triples.
- All 32 parents share atomic IDs with inventoried prior non-TRAIN study panels.
- **Zero parents share official TRAIN atomic IDs.** The previous atom-clean restriction
  removed deeper questions through prior DEV-panel exposure, not through official TRAIN
  atomic overlap. Do not describe this panel as testing training-familiar atoms.
- 172 of 756 unique title/text documents also occur in official TRAIN; 63 of 64 variants
  contain at least one such document. All 64 have some shared title.
- These are new composed questions relative to the exclusion inventory, with prior-study
  component exposure and TRAIN document reuse. Neither exact matching nor atomic IDs
  establish semantic or pretraining independence.

Public projections contain only question and documents with contiguous docid, title,
and text fields. The qualified label-blind document-order seed remains 2026092192.
Official labels, answers, aliases, hops, and component IDs stay host-only. Official
sufficiency labels are unchanged; the earlier semantic audit does not relabel this panel.

Native chat tokenization measured 1,857–5,181 prompt tokens, maximum 5,309 including the
128-token output allowance. All 64 fit the existing 8,192 runtime bound without truncation.

## Immutable artifacts and verification

Research-store root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- `sufficiency-compositional-inputs-001/cases.jsonl` SHA256:
  `b43ba4e36a92014f49752681b26f0f98697170840e0d4c735249cf96e243f9c2`.
- Its `MANIFEST.json` SHA256:
  `bfe2e82bdb8e7548440fd962cb73552013cd0a3845b456cb0db5e706746f9c2c`.
- Source: `source-045-sufficiency-compositional-inputs/prepare_sufficiency_compositional.py`,
  SHA256 `e0e52b92f0b6b320e39801ccc877ec7142ec82f2b689bd7de2f059e432655d50`.

The manifest binds every consumed input file, official archive/member hashes, preparation
dependencies, tokenizer-model manifest, exact selected IDs, component memberships, and
all prompt lengths. Two sealed focused fixtures pass (0.18 s), covering hash selection
and exact-only exclusions. Ruff passes. A separate CPU check inspected all 64 public
schemas, contiguous document IDs, and complete positive/negative pairing.

## Source046 readout — accepted, awaiting predecessors

Main accepted source046 at18:57 UTC after reviewing the profile-only shared-reader
change and rerunning nine focused tests (all passed). Supervisor64594 waits for
source044. No model outcomes were inspected to make this decision.

The additive 046 seal uses the validated 039 reader's endpoint authentication, shared
three-adapter loader, native calls, and scoring unchanged. The worktree shared reader
now accepts an optional panel-profile parameter; original defaults remain intact and
the sealed 039 source is untouched. A thin 046 entry point supplies the frozen hash,
seed 2205, manifest identity, 20-group count, and official metric-source hashes.
There is no duplicated collector or global-validator monkeypatch. The resulting PLAN
binds both shared reader and wrapper/profile identities.

The proposed comparison remains warm joint32 versus the actual stopped RL endpoint
versus its matched SFT endpoint: 32 parents × 2 variants × 2 seeds × 3 models = 384
calls, with the existing public prompt and sampling. Endpoint selection is not DEV-score
selection. A zero-real-update RL outcome must skip the identical triple readout.
The input freeze and subsequent GPU acceptance are separate receipts. Any complete-panel
uncertainty analysis must retain all 32 parents and account for the 20 atomic groups.

The proposed source is `source-046-sufficiency-compositional-readout`; its conditional
launcher and `SUFFICIENCY-COMPOSITIONAL-READOUT-DECISION-001.json` remain proposed until
main acceptance. It waits for authenticated `textcraft-pilot-001` completion (32 episodes,
16 flat and 16 recursive, no missing slots/unresolved calls), then authenticates training
endpoints or the explicit zero-update skip. No endpoint-dependent PLAN is fabricated
before the actual training terminals exist. Nine sealed tests pass in 4.98 seconds;
launcher validation, Ruff, and an independent native tokenizer check of all 64 frozen
prompts pass. This is CPU readiness, not evidence of GPU execution or scientific gain.
