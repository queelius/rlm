# Teaching-repair transfer feasibility — 2026-09-29

Conditional proposal only, after the fixed Phi repair readout. No experiment code, task dataset, GPU descriptors, model calls or queue changes were produced. [FEASIBILITY.json](FEASIBILITY.json) records exact candidates, exclusions, source hashes and CPU checks.

## Decision

Eight roots meeting the name/exposure constraints exist, but **not eight comparable depth2–5 roots**. Only six remain: one depth2, two depth3, three depth4, none depth5. Keep these as a fixed six-root core. Two prospectively selected depth6 roots are feasible but should remain separate harder diagnostics, never pooled to imply a matched eight-root breadth replication. No selected task was replaced after qualification.

| Official VAL ID | Root name | Declared/actual depth | Public calls | Maximum context + response cap |
| --- | --- | ---: | ---: | ---: |
| 10 | m1_i2_11 | 2/2 | 5 | 1086 |
| 542 | o4_i3 | 3/3 | 9 | 1403 |
| 268 | o7_i3 | 3/2 | 7 | 1200 |
| 601 | a7_i4 | 4/4 | 19 | 2241 |
| 497 | o0_i4_10 | 4/4 | 23 | 2670 |
| 32 | a6_i4 | 4/4 | 17 | 2043 |
| 132 — harder | o0_i6_6_10 | 6/6 | 87 | 8155 |
| 522 — harder | c0_i6_6_10 | 6/6 | 71 | 6825 |

Every ID above has prefix `textcraft_synth.val.`. All eight public-only scripted continuations score1 while charging every action/EOS under96 calls,8192 generated tokens,256 per response and8192 context. VAL132 has only37 context tokens and9 calls of headroom; modest policy mistakes can exhaust it. This is feasibility, not model efficacy.

## Selection and provenance

Selection seed2026092903; ascending SHA256(`seed:official_id`) within declared depth, lexical-ID tie-break, skipping used/excluded roots. The original breadth quota2/2/1/3 proved impossible using metadata alone. Before world construction, the rule became all six available depth2–4 roots plus the first two depth6 roots. Quantity variants do not create new roots.

The exclusion audit read59 task inventories and416 TextCraft-containing PLANs, plus small selection/manifest/queue receipts at one-to-three directory levels of active R. It covers all eight breadth panels, pilots, older fresh goals, readiness, delegation, current fresh TRAIN A/B and prepared readouts. It protects137 official IDs/135 goal roots, then unions126 original SFT queried products (221 names total). Original privileged, corrected-known and discovery rows all independently have the same126 query products. None of the eight candidates is in that union. Whole official VAL listed only as a prior TRAIN exclusion was correctly not treated as prior evaluation exposure.

The machine-readable ledger pins every inspected input. Two historical truncated quantity-input manifests were preserved; their intact task inventories provide exclusions. Three zero-model-call payload-fixture PLANs contain one synthetic ID and are explicitly identified, not mislabeled official tasks. Registered metadata is the boundary: this does not prove absence from unregistered assets outside R. Refresh that inventory if promoted.

Source: [ApGa/platoon](https://github.com/ApGa/platoon/tree/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe), commit `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`, MIT. The original632 VAL rows contain256 roots, disjoint from official TRAIN root names. Source rows and line hashes remain official VAL provenance; world53 versions would be **derived VAL-root diagnostics**, not unchanged official examples.

## World and overlap

Seed53 is the smallest integer≥42 absent from recorded recipe/world fields and queued world arguments; registered seeds were42–52. Native world digest: `c0bd213529b1c1dc9bfbaf0777f5181af3e3db4906ba29b1ec7dfe17491c37ac`. It retains the same1452-item namespace. All eight root recipes and named closures differ from world42.

Six of eight closures reuse SFT product names; two share an exact named SFT recipe (`o4_i1`). Three rooted unweighted DAG shapes, and one quantity/yield-weighted shape, occur in SFT. Thus this is not unseen vocabulary, recipe-disjoint structure or a new generator. Graph checks preserve shared nodes but omit starting inventory and requested quantity.

Construction reused the inspected official conservative base-stock extractor, preserving irrelevant original distractors; inventories change with world. Offline gold stayed outside actor/public-solver inputs. Full public histories were separately replayed with actual budget accounting and cached tokenizer—no model load.

## Required adaptation if promoted

Use a thin new data/world adapter: current accepted evaluators restrict panel/world lists. Freeze derived task bytes, truthful manifests and world53 binding; explicitly override the reused constructor’s hardcoded `derived_world_seed=43`. Reuse unchanged flat native collector/auditor and strict response/score semantics. Bind actual known/discovery/stable cp23 actors at predetermined fit2208, paired rollout seeds, and separate core/harder reporting; no checkpoint selection on this slice. A saved-request fixture and refreshed exposure ledger suffice before parent admission.

Budget expectation: roughly1–2 GPU hours across the three fixed actors, potentially3 with hard-case failures; no grid or job caps are prepared. A null Phi repair should narrow the claim first. Brainstorming kept this a bounded feasibility spike; no implementation branch was opened.
