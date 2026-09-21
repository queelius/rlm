# Joint answer/sufficiency SFT: bounded competence qualification

Question: does one epoch of paired answerable/unanswerable supervision improve the
official paired answer+sufficiency score beyond the same number of positive-only
SFT examples, without merely suppressing supported answers? This is neither a novel
training objective nor a pure calibration experiment.

## Fixed data and controls

CPU-prepared immutable inputs: `sufficiency-sft-inputs-draft-001` in the September21
research store. Select 256 official MuSiQue TRAIN parents by sorted
SHA256(`2026092189:` + parent ID), independently of answers/outcomes/hops/length.
Exclude parent ID, normalized question, and atomic-component overlap with all
official DEV/TEST and inventoried September21 non-TRAIN inputs. All19,938 TRAIN
parents passed this exclusion; the selected parents include177 two-hop,59 three-hop,
20 four-hop. The manifest records exclusion sources and hashes. No DEV answer is
used for selection or training. These exclusions are not a pretraining-contamination
claim.

Both arms use the exact current public sufficiency prompt, label-blind document
shuffle, full source, and strict `answerable` boolean + `answer` string JSON.
Only the supervised targets use official annotations. Joint uses each parent's
positive and negative variants once; positive-only repeats the same positive variant
twice. Negative targets have an empty answer. Raw public question/documents only enter
the model. Full official negatives can still contain alternative evidence; training
learns the dataset's labels, not an infallible semantic support oracle.

| Arm | Examples | Target tokens including EOS | Prompt tokens | Maximum total |
|---|---:|---:|---:|---:|
| Joint |512|6,316|1,477,727|5,665|
| Positive-only repeated |512|7,512|1,511,946|5,448|

Actual CPU tokenization found zero examples over8,192 context/128 target tokens;
no truncation or length-based reselection. Same slot order/shuffle,512 examples,
32 updates, effective batch16/micro1, fresh rank8/alpha16/dropout0 LoRA, LR1e-4,
seed2026092189, one epoch. Base4B frozen, target JSON+EOS loss only. This matches
updates/examples, **not tokens, unique inputs, labels, or information**. Native tiny
Qwen3+PEFT forward/backward checks target positions, EOS, and adapter-only gradients.
Checkpoints every8 updates;30-minute cap per arm. No extension or checkpoint selection.

## Fixed endpoint readout and queue

Only a fully committed step32 is eligible. A capped earlier checkpoint is incomplete,
not a substitute. Reuse all64 existing DEV variants/32 parents, both original seeds
2026092181/2182,128 calls per trained arm, exact baseline prompt and native sampling
T.5/top-p1/top-k0/128 output tokens. One trained sufficiency adapter is frozen for each
readout; no planner/helper/additional final or reminder. Existing base128 native
requests are checked for prompt/seed/sampling/model agreement; no recollection needed.
Report official group EM/F1 and joint label score, supported-answer EM/F1, false
abstentions, negative overanswers, protocol/missing counts, and native costs.
Two seeds are repeated measurements, not64 independent parents.

Source030 training; source031 readout. Proposed decision
`SUFFICIENCY-SFT-DECISION-001.json`; main alone accepts/launches. Conditional launcher
waits for authenticated complete `qampari-reading-001` release, then joint train/readout
and positive-only train/readout. Maximum64 total updates,256 new readout calls,
80 minutes (2x30 training +2x10 readout), with600-second allocation reserve and native
exclusive owners. Logs are preserved; failed/capped stages halt, no silent retry.

Promote only if joint improves paired scoring over the positive-only control while
retaining supported-answer competence. A gain solely from more abstention is not
evidence of improved reading. Mixed results motivate a targeted support/answer audit,
not a larger hierarchy or RL claim. This previously examined small DEV panel is an
exploratory qualification, not held-out confirmation.

## Small public-feature check before the trained readout

All512 training variants and all64 initial DEV variants have20 documents, so
document count alone cannot distinguish the official labels here. Mean document
title-plus-text length is10,242 characters for TRAIN positives and9,656 for
negatives, versus9,571 and9,595 on the initial DEV panel. These descriptive
length checks do not rule out other shortcuts or establish evidence-sensitive
reasoning. They were computed from the frozen public inputs before reading any
trained evaluation outcome; no classifier was fitted or example removed.
