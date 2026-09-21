# Selective delegation implementation plan

> For agentic workers: use executing-plans or subagent-driven-development; keep exploratory checks proportional to the scientific decision.

**Goal:** Collect controlled action-value evidence and use it to choose informative controller training.
**Architecture:** Small research scripts around the completed breadth campaign's native client and service owner; no production RLM changes.
**Tech Stack:** Python 3.12, existing Prime/vLLM environment, existing HF/PEFT training environment.
**Spec:** DESIGN.md (approved conversational research design).

## Global constraints

One GPU; main is sole launcher. No generated code execution. No gold in prompts.
Full source remains available in every final arm. Immutable completed artifacts.
Preserve 10% account allowance. User explicitly requested uninterrupted autonomous
research, so no intermediate approval questions; record decisions in LEDGER.md.

## Review focus

Gold-bearing fields reaching prompts; parent/component split leakage; resumed
requests changing identity; unknown transport usage counted as zero; search data
misrepresented as ordinary on-policy RL. Focused fixtures and independent audit,
not a production-wide test campaign before a useful GPU pilot.

## Task 1: Freeze inputs (data.py, test_data.py)

- [ ] Test public projection excludes source IDs, supports, gold and decomposition.
- [ ] Test seeded parent selection and split/component disjointness.
- [ ] Implement `prepare(output: Path, train_count=256, validation_count=64, transfer_count=64)`.
- [ ] Write `cases.jsonl`: {id, split, dataset, question, documents:[{id,title,text}], answer, answer_type:'string', metadata:{source_id,answer_aliases,hops,component_ids}}.
- [ ] IDs are opaque hashes; answer/metadata remain host-side, not prompt fields.
- [ ] Write MANIFEST.json with archive URL/revision/license/checksum, selection seed, split provenance, exclusions and code hashes.
- [ ] Run focused CPU tests, create immutable external prepared input directory, report paths/counts.

## Task 2: Collect common-state alternatives (probe.py, test_probe.py)

- [ ] RED: tests for shared full-source final prompts, checkpoint validation, alias EM/F1 and cost accounting.
- [ ] GREEN: native client + owned service; `run --cases PATH --output PATH --split train --limit 32 --repeats 3 --hours 2`.
- [ ] Save checkpoint/calls/episodes incrementally, STATUS.json, SUMMARY.json, owner start/release/terminal.
- [ ] Focused tests then launch pilot immediately; check real first response.
- [ ] Queue validation and training expansion only from pilot signal; separate source versions for changes.

## Task 3: Analyze and train (analysis.py; trainer added after observed signal)

- [ ] Per-parent repeatability, outcome and cost effects; distinguish information loss, useful extra work, and random resampling.
- [ ] Freeze training decision and baseline comparisons; use existing successful LoRA checkpointing code.
- [ ] Complete bounded learning curves, checkpoint and evaluate on untouched parents.
- [ ] Record finding/limitations/pivot and prepare two informative follow-on jobs when practical.

## Task 4: Checkpoint and publication assessment

- [ ] Independent focused source/results audit while GPU is busy.
- [ ] Update research queue and readable findings; update deck only for a material result.
- [ ] Review staged source/docs, commit and ordinary push at meaningful milestones.
