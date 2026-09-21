# Canonical Hotpot replication panel: prepared, unevaluated

128 new parents are frozen at
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/hotpot-fresh-inputs-001/`.
Preparation used CPU tokenization only; no model answers, policy evaluation,
training, or outcome-based selection occurred.

- `cases.jsonl` SHA256:
  `b0dab5873987583614a6182b82382ddecef7f51d6a448821ad358e1778b65828`.
- `MANIFEST.json` SHA256:
  `45b74b6cb61805ee5366c68ffcb0b3d1330f04dcb4ca23c41160f6aa18dec322`.
- Source:7,405 canonical distractor-validation rows in the
  [HotpotQA HF mirror](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab/distractor),
  revision `1908d6afbbead072334abe2965f91bd2709910ab`, CC-BY-SA-4.0,
  acquired2026-09-21. Cached Parquet SHA256:
  `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
  This pins an HF mirror, not byte identity to the official JSON distribution.

## Fixed selection and exclusions

Sort by `SHA256('2026092175:'+original_id)` and take the first128 eligible
parents. Exclude all100 official explorer IDs and normalized original questions,
all previous September21 input IDs/questions, duplicate source IDs, and duplicate
normalized questions within the new pool. Normalization matches the existing
fresh-panel preparation: NFKC, casefold, collapse Unicode whitespace.

The complete previous-input inventory is `inputs-001`, `hotpot-inputs-001`, and
`fresh-dev-inputs-001/002/003`, including all256 TRAIN parents. File hashes,
explorer IDs, exclusion reasons, previous-question hashes, selected IDs and the
entire eligible rank order are recorded in the manifest. Exactly100 source rows
were excluded, leaving7,305 eligible:32 explorer IDs also occur in selected
previous inputs; all100 have prior-question overlap. These reason counts overlap.
No additional prior-input, duplicate-ID or new-pool duplicate-question exclusion
was needed on this source.

This is hash-uniform, **not type-stratified**: the resulting98 bridge/30 comparison
balance is descriptive only. Gold answer, support, type, level, observed results,
and predicted difficulty never determine selection. Natural source document
counts are retained:126 parents have10 documents, one has5, one has2. No
ten-document filter or outcome-based replacement was applied.

## Host-only exposure and connectivity audit

Cases use `split='transfer'`, `dataset='hotpotqa'`; canonical origin remains
`metadata.source_split='validation'`. Public prompts expose only the question
and document ID/title/text. Gold answers, supporting facts, original IDs, labels,
document exposure and connectivity stay in host metadata. Document order is
hashed independently of support labels.

Across1,267 selected documents and the exact256 TRAIN-parent source contexts:
zero exact title+text overlaps, zero normalized title+text overlaps, and four
normalized-title overlaps. The manifest records every document's matching TRAIN
parent IDs; these measurements did not filter the panel. Zero exact exposure is
not proof of no semantic or pretraining exposure.

Both normalized title+text connectivity and title-only connectivity yield128
singleton parent clusters. `metadata.component_ids` describes public-document
fingerprints, **not annotated atomic reasoning components**. Shared documents
were checked, but this does not establish full statistical independence.

## CPU native prompt audit

| Prompt | Maximum input tokens | Output cap | Maximum plus cap |
|---|---:|---:|---:|
| Exact initial title-index root | 343 | 128 | 471 |
| Exact direct full-source answer | 3,006 | 128 | 3,134 |
| Original-question helper scaffold | 3,005 | 384 | 3,389 |
| One-question/empty-trace final scaffold | 3,072 | 128 | 3,200 |

All128 cases were tokenized with the cached4B native chat template, with no
truncation and zero8,192-token flags. Helper/final scaffold lengths are **not**
actual future generated-context lengths: predicted questions, dependency
substitutions and reports still require the existing runtime context check.
Tokenizer hashes, per-parent counts and code/input hashes are in the manifest.

## Accepted next comparison—queued September21,15:40UTC

Compare frozen SFT48 root + helper-SFT36 + base final against direct base, all128
parents ×two paired repeats, official Hotpot answer EM/F1, total two-hour cap.
At most2,816 calls:256 planner attempts ×(root +at most8 helpers +final), plus256
direct calls. Source021 and `HOTPOT-FRESH-DECISION-001.json` now bind inputs,
source and checkpoint commits. Supervisor16674 waits for authenticated completion
of the fixed RL24 readout, then runs the two collectors and separate official
regrades. No Hotpot model outcomes exist at acceptance. Retain every planned
parent, protocol/missing accounting, paired parent/document-cluster uncertainty,
and native physical/deployed costs. Repeats are not independent parents.

Current `eval_planner`/`analyze_planner` still call the MuSiQue scorer. Their raw
Hotpot F1 is not authoritative: the next owner/readout must use the existing
dataset-aware `eval_helper.grade_final`/official Hotpot scorer, including
exact-only yes/no/noanswer F1. No runner or analyzer was changed in this task.

This replication tests whether the small32-parent Hotpot result extends beyond
the exposed explorer panel; no win is assumed. Further plan-only/source controls
remain conditional. Do not select new training or adapt models from these
answers. Even a positive contrast would be within-dataset architecture evidence,
not proof of faithful decomposition or clean pretraining generalization.

Preparation command (immutable destination; reruns refuse overwrite):

```bash
$PY experiments/selective_delegation/prepare_hotpot_fresh.py \
  --output "$R/hotpot-fresh-inputs-001"
```

Three focused tests pass: gold-independent hash order and exclusions, host-only
projection/document audit, and within-pool normalized-question deduplication.
Ruff passes. Four launcher fixtures also pass. Source and staged data are separate;
the supervisor is queued, not an active Hotpot GPU collector at this cutoff.

### Predecessor amendment,15:51UTC

RL continuation stopped at batch22's predeclared admission check, leaving
checkpoint21 rather than24. Its exact24 readout was not run. The independent
Hotpot comparison was therefore advanced after authenticated training release,
with no panel, model, prompt, seed, or cap change. Original supervisor16674 was
stopped while waiting; amended supervisor11339 now owns the queue via
`HOTPOT-FRESH-AMENDMENT-001.json`. The first native root/helper/final responses
were checked within90seconds of launch. This is an operational dependency
amendment, not selection on Hotpot outcomes.
