# ALFWorld action representation implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; parent reviews and dispatches.

**Goal:** Prepare one exact-command SFT endpoint and four fixed flat-policy readouts to measure
whether action representation changes within-interface learning gains.

**Architecture:** Reuse the original524-row TRAIN dataset,33-step optimizer and native episode
loop through private modules. Replace only response instruction, strict decoder and SFT targets.
Preserve the indexed public list, executed-command histories, native bridge and budgets.

**Tech stack:** Existing Qwen3-4B, BF16 base/FP32 LoRA, accepted torch/PEFT environment and
isolated ALFWorld0.4.2/TextWorld native bridge. No installs or new architecture.

**Spec:** Parent's September28 bounded task and approved standing autonomy: one original epoch,
one new command checkpoint, authenticated old index checkpoint33, base/trained controls through
each own interface,12 prospectively selected valid_unseen games×two seeds, no GPU/Git here.

## Global constraints

- Both prompts serialize identical public contexts and indexed admissible lists. No action order
  intervention, hidden evaluation expert, fuzzy matching, retries or output repair.
- Exact command JSON uses one string field `command`; index JSON stays original `action_index`.
- All524 row IDs/order and native target commands preserved; old33-step index endpoint is reused.
- One epoch,LR1e-4,rank8/alpha16/dropout0,seed2026092200,effective16,last12,original token-mean loss.
- Both interfaces retain full public history. Context overflow is unknown, never silent truncation.
- Native limits50 actions,2048 output tokens,128 per response,8192 context,T0.5/top-p1/top-k0.
- Panel selection seed2026092812; execution seeds2026092813/14; exclude prior20 screened games.
- One new training cap20min; four readouts cap24min each; expected useful1–2h. All slots retained.
- Only new owned source and external prepared outputs; existing pins immutable, no GPU launch.

## Review focus

Strict command equality and duplicate fields; identical context bytes across interfaces; unknown
checkpoint admission; no hidden fields in projected TRAIN prompts; incomplete episodes never losses.

## Tasks

- [x] `alf_rep.py`, `alf_rep_data.py`: fail focused strictness/context/projection/selection tests,
  implement interfaces plus frozen TRAIN projection and outcome-blind native-reset qualification.
- [x] `alf_rep_train.py`, `alf_rep_collect.py`: reuse native optimizer/episode functions, truthful
  lineage and actual endpoint admission; shared lock and finite owners/caps.
- [x] `alf_rep_audit.py`, `alf_rep_fixture.py`: saved response/token decoding and native transition
  replay in both interfaces, invalid command charged without a transition; fixed four-cell analysis.
- [x] `alf_rep_jobs.py`: prepared pinned descriptors, pending new checkpoint, focused tests/Ruff,
  source preservation verification and parent handoff.

Ruling: reuse old index training rather than retrain identical source/data/seed — avoids redundant
GPU work; inference is regenerated on the same new panel in all four cells. Training software
versions are checked against the original receipt; unequal target exposure remains explicit.
Ruling: do not use skill Git/approval scaffolding or broad suites — the parent explicitly assigns
autonomous CPU-only preparation and the repository requires proportional research verification.
