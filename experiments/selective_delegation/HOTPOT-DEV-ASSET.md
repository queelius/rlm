# Canonical Hotpot distractor-validation mirror

Acquired 2026-09-21 into external cache:
`/project/alex_phd/research-cache/datasets/hotpotqa-official-dev-20260921/distractor-validation-00000-of-00001.parquet`.
It is the Hugging Face `hotpotqa/hotpot_qa` mirror pinned to revision
`1908d6afbbead072334abe2965f91bd2709910ab`, retrieved from
`https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet`.
SHA-256: `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
The dataset card reports CC-BY-SA-4.0. This is an HF Parquet mirror, not a
claim of byte identity to CMU's official JSON distribution.
Machine-readable acquisition receipt: [`ACQUISITION.json`](/project/alex_phd/research-cache/datasets/hotpotqa-official-dev-20260921/ACQUISITION.json).

PyArrow reads 7,405 validation rows with fields `id`, `question`, `answer`,
`type`, `level`, `supporting_facts`, and `context`. `type` has both official
bridge and comparison values; context stores title plus sentence lists.
There are 5,918 bridge and 1,487 comparison rows (all 7,405 are `hard`).

The existing `hotpot_official_explorer_sample100.json` is expected to be drawn
from this official development split: all 100 old explorer IDs and exact
questions occur here. That fact does not invalidate it as an exploratory
official-dev diagnostic. It does mean all 100 previously exposed explorer cases
must be excluded from any future fresh selection. No selection was performed.

Smallest useful future expansion: choose a new label-blind 32-case canonical
distractor-dev panel after excluding all explorer-100 IDs/questions, all prior
Hotpot selected IDs/questions, and future adaptive readouts; stratify 16 bridge
and 16 comparison if availability permits. Compare direct versus frozen planner
under the same public ten-document projection and official Hotpot EM/F1. This is
within-dataset diagnostic evidence, not clean OOD or pretraining-clean evidence.
