# MuSiQue full-pair context audit

This is a CPU-only schema and provenance audit. It neither selects a panel nor
inspects model outcomes. It extends [MISSING-EVIDENCE-ASSET.md](MISSING-EVIDENCE-ASSET.md).

The official full-development file has 2,417 two-row parent groups. Every group
has one `answerable=true` and one `answerable=false` row with the same question
and answer. The benchmark target is supplied-document answerability: the
official grouped answer-sufficiency metric keeps the answerable member's answer
EM/F1 only if the two predicted answerability values exactly match the official
pair labels. It does not establish that the answer is globally false or that a
deployment retriever would necessarily require more evidence.

## Context is deliberately different, but not an isolated deletion treatment

After projecting only `idx`, `title`, and `paragraph_text`, no pair has identical
public documents. Paragraph count is unchanged for 2,392/2,417 pairs, but title
sequence changes for all pairs and title set changes for 2,416/2,417. Median
title-plus-text characters are 10,119 answerable versus 9,719 missing-evidence;
the answerable-minus-missing range is -5,552 to +5,771 characters. Thus length,
title, and distractor/context rewriting are plausible shortcut cues. The raw
missing variants also retain some designated supporting paragraphs, so the test
must not be described as "all support removed."

Never expose `answerable`, `is_supporting`, decomposition, answer, or aliases in
a prospective action prompt. A label-free public-document projection is
straightforward, but it does not remove the context-distribution confound.

## Exact known-overlap inventory

The audit excludes every source parent, normalized question, and atomic component
in `inputs-001` (all train/validation/transfer rows), `fresh-dev-inputs-003`,
and breadth `cases-v2`. This leaves 438 parent pairs component-disjoint from
those known sources: 434 two-hop, three `3hop1`, and one `3hop2`. It is enough
for a later label-blind 32-pair two-hop screen using seed `2026092180`, but not
for a balanced-hop claim. These are **known-component-disjoint**, not semantic,
document, or pretraining-clean pairs.

No panel was materialized. The external immutable receipt records exact hashes,
all counts, nonexclusive exclusion reasons, and a checksum of the prospective
label-blind ordering:

`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-missing-evidence-pairs-001.json`

## Script provenance

The immutable receipt was created by the original script SHA-256
`4f2b98a238ea9696b91bf9ecfaaeee24db24ec6c53031382738c8e034a8c747c`.
That exact source is retained outside the mutable worktree at:

`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-source-missing-evidence-001/audit_missing_evidence_pairs.py`

The current worktree script SHA-256 is
`01470284f51c086391606194142b66dd264bc3d6d5ab616b574f4e41574e7c41`.
It differs only by line wrapping to satisfy Ruff E501; it does not rewrite the
receipt. The old-to-current source mapping is therefore the external archived
path above to `experiments/selective_delegation/audit_missing_evidence_pairs.py`.

Reproduce with:

```bash
/project/alex_phd/envs/prime-rl-5990b1b/bin/python audit_missing_evidence_pairs.py \
  --archive /project/alex_phd/research-cache/datasets/musique-v1.0-922ac98f19a201998dbdae6d7f2887a5258dbdeb/musique_data_v1.0.zip \
  --inputs /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/inputs-001/cases.jsonl \
  --fresh003 /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/fresh-dev-inputs-003/cases.jsonl \
  --breadth /project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/data/cases-v2.jsonl \
  --output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-missing-evidence-pairs-001.json
```

The immediate experimental prerequisite remains a full-context direct
sufficiency baseline with structured answerability, before any constrained
paragraph-action policy.
