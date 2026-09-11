---
title: Two realistic publication paths
meeting_date: 2026-09-11
evidence_cutoff: 2026-09-11T12:55:00Z
status: discussion_draft
---

# Option 1 — Verified record handoffs inside an RLM

**Possible paper.** When a helper answers many questions together, it can return a sensible label
for the wrong record. Study when that failure appears, how explicit record-to-answer links change
it, and whether a verified handoff improves the complete RLM answer at a useful cost.

## What is established

- Wrong visible-record identifiers redirect answers toward the named record. On a 16-context
  replication, unrelated aliases scored 80.47% while aliases pointing to a different visible
  record scored 38.80%, a +41.67-point paired difference.
- Matching input/output identifiers repair the large-batch failure. On the shared Qwen panel,
  arbitrary matching tags scored 84.9%, sequential row numbers 85.4%, and labels without matching
  keys 34.2%.
- A control in which both conditions contained arbitrary tags scored 78.8% when tags matched and
  32.6% when input and output tag sets were disjoint. This supports literal matching within the
  tested interface, not merely adding more output text.
- The exact same nested batch-size panel was reproduced with Mistral-7B. At 64 records, Mistral
  improved from 34.7% without keys to 58.1% with sequential keys and 52.0% with opaque keys. Qwen
  on the same records scored 43.8%, 84.6%, and 83.4%. The direction transfers across families;
  the high absolute keyed accuracy does not.
- In the output-order crossing test, Qwen opaque-key accuracy remained approximately 83%, 82%,
  and 81% for identity, reverse, and interleaved output orders. This is useful retention evidence,
  but not a statistical equivalence result or proof of an internal binding mechanism.
- The equal-record-work test processed the same 768 records in every condition with released
  Qwen3-4B, no adapter or cache, and four workers. All 848 native calls were authenticated:

| Policy | Correct / 768 | Accuracy | Workload time | Prompt tokens | Output tokens |
|---|---:|---:|---:|---:|---:|
| 48 records, no matching keys | 373 | 48.57% | 19.27 s | 49,307 | 2,537 |
| 48 records, opaque matching keys | 651 | 84.77% | 91.04 s | 60,768 | 15,228 |
| 16 records, no matching keys | 619 | 80.60% | 18.67 s | 53,821 | 2,846 |
| 1 record, no matching keys | 665 | 86.59% | 31.08 s | 154,594 | 4,076 |

This is a quality/time/token tradeoff, not universal speed or cost dominance. Single-record calls
were slightly more accurate and faster than keyed 48-record calls locally, but repeated far more
input text. Sixteen-record calls were fastest but less accurate. It is one counterbalanced,
16-context local test; provider billing and GPU-kernel time were not measured.

## Prior art and contribution boundary

[Batch Prompting](https://aclanthology.org/2023.emnlp-industry.74/) already uses position identifiers,
and [BatchPrompt](https://openreview.net/forum?id=Agyicd577r) studies position and permutation
effects. [Cascaded Batch Prompting](https://arxiv.org/abs/2608.27038) separates semantic class
generation from symbol mapping. [Entity-binding work](https://arxiv.org/abs/2310.17191) studies
internal binding with causal interventions. Input-dependent output rules also improve task scores
in [Geng et al.](https://aclanthology.org/2023.emnlp-main.674/), while
[CRANE](https://arxiv.org/abs/2502.09061v4) and
[The Hidden Cost of Structure](https://aclanthology.org/2025.ranlp-1.124/) show that constraints can
hurt.

The contribution cannot be “identifiers help”: identifiers and structured generation are already
known. The strongest possible paper would connect a controlled wrong-record failure to an
identifier-keyed, verified record handoff **inside an RLM**, then show end-to-end utility and its
cost frontier. The current `ask_batch` interface aligns whole requests and responses; the proposed
adapter would preserve semantic record identity within each helper request and verify the join.
That adapter is proposed, not yet implemented or validated.

## Decisive missing comparison

Keep model, inputs, seeds, and a fixed resource budget constant. Compare the current positional
handoff with the proposed record-linked handoff on complete tasks. Score helper labels, record
uptake, requested calculation, final answer, availability, elapsed time, and all token dimensions.
Include smaller unkeyed calls as a serious baseline: the equal-work result shows that this simple
alternative can win on latency or accuracy. A managed or adaptive batch selector is a conditional
future direction, not an implemented contribution.

The strongest paper requires a complete-answer benefit beyond known identifier techniques. Another
component-only tag study would improve precision but would not close the publication gap.

**Conditional path:** 2–4 days for the first fixed-budget complete-task comparison if the adapter
and audit reuse current infrastructure; 2–3 weeks for a workshop paper if the effect survives a
second model or task; 4–6 weeks for a stronger submission with independent contexts, clustered
uncertainty, cost-sensitive baselines, and end-to-end utility. These estimates assume informative
results and continued compute access; a null whole-task result would redirect or retire this path.

# Option 2 — Training routines that survive composition

**Possible paper.** Train a small main model to acquire evidence and execute requested calculations,
then measure whether local improvements survive composition. Distinguish a correct final number
from a correct number produced by the wrong calculation.

## What is established

- On newly selected record groups, faithful-and-correct execution improved from 12/72 for the
  starting model (missing-result bounds 12–19) to 55/72 and 53/72 for two separately trained SFT
  models (bounds 55 and 53–55). Both trained models improved all eight context clusters.
- On the newly tested compound operators, the trained model achieved 29/72 faithful-and-exact
  solutions versus 2/72 for the start, with all-planned bounds 29–30 and 2–13. The independent
  replay covered the same 144 endpoints byte-for-byte. This is the strongest evidence that SFT
  changed execution rather than merely exploiting zero-valued answers.
- The compound prompts explicitly specify the two-stage operation. The result shows execution of
  described combinations of familiar operations, not autonomous discovery of a decomposition.
- Mixed-contract helper SFT improved the compact A/B/other interface without preserving a universal
  downstream gain. In the supplied-plan composition test, the original helper produced 0/8 exact
  totals, the mixed helper 1/8, and known-correct labels 8/8. Local classification gains are not
  automatically system gains.
- Two terminal-reward recipes did not improve held-out final-answer counts at their tested doses.
  The later authenticated pair started from 57/72 and produced 55–57/72 with answer reward and
  54/72 with an extra calculation-consistency bonus. These are secondary final-answer counts;
  the full predeclared faithful-and-correct population review is incomplete.
- Each authenticated reward arm attempted 576 solutions. The answer-only and bonus arms made 19
  and 21 optimizer updates from 108 and 147 admitted episodes, so they are exploratory and not
  dose-matched. A bounded review of all nine bonus paths whose strict outcome changed found only
  three faithful calculations and two faithful-and-correct paths; it is not a population estimate.

## Prior art and contribution boundary

[Recursive Language Models](https://arxiv.org/abs/2512.24601) establishes recursive inference and
released transition training; [lambda-RLM](https://arxiv.org/abs/2603.20105) motivates typed
functional decomposition; [SRLM](https://arxiv.org/abs/2603.15653) emphasizes program selection.
[Steno's cost-aware RLM thesis](https://essay.utwente.nl/essays/110199) trains a Qwen3-4B root with
Prime-RL and LoRA in a much larger, differently rewarded setup. Training a small RLM with supervised
or reward feedback is not itself novel, and our bounded negative reward results do not contradict
success under another recipe.

The defensible contribution is measurement plus a learned execution result: SFT can transfer a
task-sensitive evidence-and-calculation routine to new record groups and explicitly described
compound operators, while component accuracy and terminal reward can misstate downstream progress.
Any paper must retain native availability, executed-operator faithfulness, zero-answer coincidences,
optimizer dose, and physical cost rather than reporting only final accuracy or training loss.

## Decisive missing comparison

The stale gaps of “try new combinations,” “run fresh Mistral,” and “wait for reward training” are no
longer accurate: compound operators, same-panel Mistral, and the longer reward pair are complete.
The remaining question is whether the supervised execution gain transfers to a genuinely different
task family or realistic complete RLM workload, and whether a more informative credit signal can
improve that endpoint under matched opportunities and dose.

A smallest credible next study would pair the trained and starting policies on one new task family,
freeze nonzero and compound cases prospectively, and manually audit requested execution as the
primary endpoint. A reward follow-up is secondary until it has matched doses and authenticated
intermediate evidence. Do not promote calculation consistency to semantic faithfulness: a program
can faithfully compute from a wrong helper map, or agree numerically while executing the wrong
operator.

**Conditional path:** 3–5 days to build and audit one new complete-task transfer panel; 1–2 weeks
for another training realization and matched comparator; 4–8 weeks for a conference-ready study
spanning interface SFT, terminal reward, an execution-grounded alternative, and full cost
accounting. These estimates depend on a reproducible transfer effect; without it, this remains a
useful negative/measurement study rather than a broad training contribution.

# Evidence at this cutoff

All paths below are under `/project/alex_phd/runs/rlm-research-r4/analyses/`.

- S3 fresh-input SFT: `root-question-sensitive-fresh-input-three-policy-live-2026-09-11/REPORT.md`,
  SHA-256 `df5db6c9fe798bebe9e4606a3d6ee7ba273d622078e7b8f1351febeb501547e8`.
- S4 compound execution: `root-qs-unseen-operator-composition-live-2026-09-11/REPORT.md`, SHA-256
  `e4d3f07360c9fe0c7309b5685141e6b206d2a47f75fc1aa2ef982e46fe212d99`.
- H9 same-panel Mistral: `leaf-mnli-mistral-nested-batch-live-2026-09-11/REPORT.md`, SHA-256
  `7bbd3d2a84631f290cd3753eb756d28ed65fac64a974b23c4748bdcd3e6f8815`.
- H10 output-order crossing: `leaf-mnli-output-order-crossing-live-2026-09-11/REPORT.md`, SHA-256
  `423c4d342c336a4200403beb29215297e019e300d6625ea08a5745db56bf6703`.
- E1 equal-record work: `leaf-mnli-equal-record-work-live-2026-09-11/REPORT.md`, SHA-256
  `74df0600080d5c609c618ddeed816ae2aa6566e00a6052d2adcdd68215d04db0`;
  its 848-row native audit is SHA-256
  `ea6e0eb689074bcd58c2181e279365a32161902a649be2031405fe1ae596ea50`.
- R3 reward pair: `root-authenticated-map-reward-pair-live-2026-09-11/final-pair-2026-09-11T0840Z/REPORT.md`,
  SHA-256 `eecbb3f8f56ec5450662990fa80e348074ad64623c2e3ea0638a2d141ad366f0`;
  bounded changed-path semantic review SHA-256
  `6314bea53b9cea4e213f312aa1b3bb7c5666897c32c8d0f1b043729ee7664519`.

# Recommendation for the meeting

Lead with Option 1 because the failure and repair are visually clear, reproduce across two model
families, survive output reordering, and now have a concrete quality/time/token comparison. Frame
the proposed contribution as a verified record-level handoff with end-to-end utility—not as the
invention of identifiers or a universal batching advantage.

The current deck reflects that logic: its practical-cost page shows the equal-record-work tradeoff,
then the next page proposes managed grouping plus record-linked handoffs and a fixed-budget
whole-task test. Adaptive selection remains a hypothesis for that future system, not a result.

Present Option 2 as the broader systems path. Its strongest positive result is supervised transfer
to faithful compound execution (2/72 to 29/72), while the reward studies are bounded negative
evidence. The publication decision should turn on the next end-to-end transfer result, not on adding
another component metric or another unmatched reward run.
