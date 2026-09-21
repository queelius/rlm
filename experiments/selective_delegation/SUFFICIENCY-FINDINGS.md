# Sufficiency: both answer quality and evidence-sensitive abstention need work

Completed frozen-base4B baseline, September21. Independently reconstructed all128
native prompts, seeds, request digests, adapter-disabled states, parsed replies
and output-token counts, then reran the sealed official scorer: SUMMARY matches
exactly. All128 responses were available and strict-JSON valid; no transport or
protocol failures. There are32 official parent pairs, two variants and two seeds,
not64 independent paired examples. All selected parents happen to be two-hop.

| Measure | Observed result |
|---|---:|
| Official paired answer EM | 9/64 =14.06% |
| Official paired answer F1 | 16.80% |
| Both sufficiency labels correct | 17/64 =26.56% |
| Supported-variant answer EM | 16/64 =25.00% |
| Supported-variant answer F1 | 33.99% |
| Abstention on supported variants | 24/64 =37.50% |
| Answerable=true on official negative variants | 23/64 =35.94% |
| Correct individual sufficiency labels | 81/128 =63.28% |

The paired decisions are revealing:17 answer-positive/abstain-negative pairs,
24 abstain-on-both pairs,23 answer-on-both pairs, and zero inverted pairs.
Thus47/64 pairs receive the same availability decision despite different supplied
documents. Among40 answered supported attempts,16 are exact-correct; answering
ability is a separate limitation from abstention. With these fixed answers,
perfect sufficiency decisions could recover seven more paired exact scores,
raising9/64 to16/64—not solve the remaining answer errors.

## Concrete native examples

- **Wrong fact, not JSON failure:** `2hop__391857_162018`, both seeds and both
  variants: asks when the author of *Prince Prigio* died; output is1889. The
  supplied book paragraph gives1889 as publication year. The supported variant
  also contains Andrew Lang's1844–1912 biographical dates, but the answer remains
  the publication year. The official negative appropriately lacks that death
  evidence in the inspected paragraphs.
- **False abstention despite evidence:** `2hop__998_25839`, both seeds: asks the
  maximum load for versions1.0/2.0 of the device replacing FireWire in later
  iPods. The supported context links iPods to USB and explicitly states a maximum
  of five unit loads for USB1.x/2.0. The model outputs `answerable=false` with an
  empty answer. This is not explained by missing input, truncation or bad JSON.
- **Important negative-label limitation:** `2hop__91667_67223`, both seeds: model
  answers Tracy Mosby on both variants. The official negative still contains a
  paragraph describing Ted telling the story and living a married life with
  Tracy Mosby. This may support the requested relationship through a different
  route. Count the reply as an **official-label overanswer**, but do not describe
  it as demonstrated hallucination or proof the model ignored absent evidence.

These examples are diagnostic, not a relabeling of the panel. All32 parents remain
in the official result. The native case IDs are respectively
`2314ca14cf9aba493a159745` (supported author), `0626f95a930e3420fdc5f37c`
(supported USB), and `00915b1e0919d3389540f9ce` (official-negative Tracy).

## What the official score does and does not establish

`GroupAnswerSufficiencyMetric` requires **both** availability decisions to match
their labels before crediting the supported sibling's answer EM/F1. It does not
score an insufficient sibling's answer as an alternative correct answer. This
conjunction can hide a correct supported answer when the negative is answered;
report answer metrics and the two error directions alongside it.

The variants are official natural document sets, not a controlled intervention
that deletes exactly one uniquely necessary fact. Titles, distractors, length
and alternative evidence paths can differ. Original indices, support IDs and
labels were removed from prompts, but that does not remove these content
confounds or validate every negative as semantically unanswerable. Repeated seeds
are not independent parents; atomic components can also overlap within the panel.
There is no learned-policy comparison, retrieval action, delegation advantage,
or calibration probability in this one frozen binary-output baseline.

Cost:128 calls;359,894 input +1,780 output =361,674tokens;91.38summed native-call
seconds and106.99owner wall seconds. No context truncation or outcome selection.

## Most informative conditional training comparison

If pursuing evidence-aware stopping, test **one small paired sufficiency SFT**
against this exact frozen prompt/answer contract before planner RL. Use only
official `musique_full_v1.0_train.jsonl` pairs, an outcome-independent hash-selected
256 TRAIN parents (512 variant rows), excluding all development/test parent,
normalized-question and known atomic-component overlaps. Keep public-only shuffled
documents; targets are official `answerable` plus the supported answer, or empty
answer for the negative. Labels are privileged training supervision, never input.
Do not mine these32 development failures as training examples or silently repair
negative labels based on this analysis.

Predeclare one epoch, fresh rank8 base4B adapter,16 examples/update (32 updates),
LR1e-4, no truncation, CPU token audit and at most30minutes on oneA10040GB. This is
a proposed bounded comparison, **not accepted training**. Evaluate base versus
the single fixed endpoint on the same development contract, report both official
paired metrics and positive-answer/false-abstention/overanswer components. Treat
improved abstention purchased by broad answer suppression as failure; also compare
always-answer/always-abstain decision references, whose paired sufficiency score
is necessarily zero. If promising, freeze decisions before a separate untouched
paired panel and audit alternative-evidence negatives transparently.

This joint answer/sufficiency SFT would not isolate pure calibration from better
reading; a later positive-answer-only SFT control could separate those effects.
Do not jump straight to binary planner credit or a learned retrieval gate: this
baseline gives no reward for which additional evidence to request, and nominal
negative labels can conflict with alternative supplied support.

## Immutable evidence

R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Source `R/source-024-sufficiency/`; outcomes `R/sufficiency-001/`; frozen inputs
`R/sufficiency-inputs-001/`. Official MuSiQue metric commit
`922ac98f19a201998dbdae6d7f2887a5258dbdeb`.

- SUMMARY SHA256: `c32f601f6f459f6cc723b09626a8cf8cb46af6a9299dd45729990cd130600cf5`.
- PLAN SHA256: `f2daf596ff0ccfcc1f95e68960d438bb1d0688c0d11ad5bba845a934ee53b370`.
- Cases SHA256: `ed6f40e7fe23403cbf8d8d708d9375a27acf79c9c6436050a3f7334edb2e50a0`.
- Sealed scorer/collector SHA256:
  `fa5c6acd57fec9e2e40741f63b0bf2540d857b8092ddc39b604e554908e45d0a`.

Read-only audit; no new GPU call, source mutation or outcome relabeling.
