# Conditional sufficiency RL/SFT held-out panel

**Status update, 18:11 UTC:** Main accepted the separate readout implementation
`source-039-sufficiency-readout` after full code review and seven sealed CPU tests.
Supervisor75166 waits for matched training or an authenticated zero-update skip.
The source and frozen panel are bound now; actual endpoint hashes enter the runtime
plan only when those checkpoints exist. Earlier preparation-only statements below
describe the panel freeze, not the current launch status.

Frozen before any conditional 037/038 updates. Ready data are `sufficiency-heldout-inputs-003` in the research store; corrected preparation source is `source-039-sufficiency-heldout-repair-001`. This is a CPU-only preparation, not acceptance of a GPU readout.

Selection takes the first 32 eligible official DEV parents ranked by SHA256(`2026092198:` + original parent ID), retaining both official variants. No hop, answerability, document-length, or outcome filter was used. The eligible pool has 164 parents. All 32 selected parents happen to be two-hop. Their atomic components form 30 connected clusters (29 singletons and one three-parent cluster).

The inventory excludes exact parents, normalized questions and component unions from every existing input panel, both breadth inventories, the fixed 128-parent TRAIN proposal, and all official TRAIN examples. CPU-abandoned panels were conservatively excluded too; that is not a claim they were outcome-exposed. The manifest records each consumed inventory and hash. Shared text remains substantial: 132 of 736 unique title/text pairs occur in official TRAIN, affecting 63 of 64 variants. This is not a document-clean or semantic-clean evaluation.

Public inputs remain only question plus contiguous document IDs, titles and text, using the qualified label-blind ordering routine (document-order seed 2026092192). Labels, original IDs, hop counts, components and annotations remain host-only. Cached native chat tokenization gives 1,859–4,020 input tokens; input plus 128 output tokens is at most 4,148, below the unchanged 8,192 limit without truncation.

The prospective comparison is warm joint-SFT32 versus the actual terminal RL037 endpoint versus matched SFT038, with seeds 2026092181/2182: 32 parents × two variants × two repeats × three endpoints = 384 calls. No evaluation runtime is implemented here. The qualified 031 validator hardcodes its older step32 endpoint and must not be reused or monkeypatched for new endpoints. New endpoint validation must bind the committed boundary, optimizer step, sample cursor, warm-start identity and source plan. If RL commits no updates, do not run a pointless three-way identical comparison. A stopped or capped endpoint is selected by its predeclared stopping rule, never by DEV performance.

Cases SHA256: `52d4c74fedbb89b3b6b56d26538d95d2d512445d0bf7c52883d765c7b1195150`.

Manifest SHA256: `fac4f7d9f768e868514710da1c2b061e5362f49d476586915bcd1303b841f103`.

Four focused tests pass from the corrected seal, and Ruff passes on the preparer/tests. Independent `BaseClient.ids` token counts agree for all 64 variants. Two failed CPU preflights are preserved with path maps: the first had a path-type provenance error, the second counted tokenizer mapping keys instead of tokens. All three attempts have byte-identical case files. Original sealed sources and erroneous audit receipts were not overwritten.
