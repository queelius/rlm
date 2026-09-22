# Deeper sufficiency: improved answering, no clear paired-score gain

Completed046 readout, 2026-09-22. All384 native requests/responses, token
decodings, fixed adapter identities, seeds and official grades passed independent
analysis. All384 are valid and available, with zero missing episodes, protocol
failures or unresolved starts. No checkpoint was selected by these outcomes.

The same fixed joint32 warmstart, eight-update paired-RL endpoint and matched
eight-update extra-SFT endpoint previously evaluated on039 now face32 new
composed questions:16 three-hop and16 four-hop parents. Each has supported and
unsupported variants and two seeds:64 paired attempts per policy, **not64
independent parents**. The frozen atomic-component partition contains20 clusters.

| Policy | Paired EM /64 | Paired F1 | Both labels /64 | Supported EM /64 | False abstentions /64 | Negative overanswers /64 |
|---|---:|---:|---:|---:|---:|---:|
| Warm joint32 | 6 | .14278 | 13 | 9 | 44 | 9 |
| Paired RL8 | 7 | .16287 | 17 | 14 | 33 | 16 |
| Matched extra-SFT8 | 6 | .12604 | 14 | 15 | 28 | 23 |

Both continuations answer more supported questions correctly and abstain less,
but also answer more unsupported questions. The gains in supported answering
do not establish improved paired reliability. RL is one paired success above
both other policies; uncertainty includes either direction. There is no
established RL advantage, and overlapping intervals are not equivalence evidence.

| Contrast | Paired EM change, percentage points [95% CI] | Paired F1 change, percentage points [95% CI] | EM wins /losses |
|---|---:|---:|---:|
| RL − warm | +1.56 [−5.36,+8.06] | +2.01 [−4.52,+8.82] | 3 /2 |
| Extra SFT − warm | 0.00 [−6.06,+6.06] | −1.67 [−8.14,+3.74] | 3 /3 |
| Extra SFT − RL | −1.56 [−6.76,+3.70] | −3.68 [−10.36,+2.63] | 2 /3 |

All wins/losses are between protocol-valid pairs. Intervals resample the20 frozen
atomic-component clusters, retaining both variants and seeds and weighting by
parents;20,000 draws, seed2026092200, percentile95. Parent-only companions remain
in the report. These small exploratory, unadjusted intervals do not establish
full distributional independence.

Descriptively, three-hop paired EM is3/4/2 of32 for warm/RL/SFT; four-hop is3/3/4
of32. These small strata do not identify a depth-specific training effect.

## What changed from039

On the earlier two-hop039 panel, warm/RL/SFT paired EM was10/18/19 of64:
both continuations improved paired correctness, with no demonstrated RL-over-SFT
advantage. On046 it is6/7/6. Thus the earlier **paired** improvement does not
clearly replicate on this deeper panel, although the less-abstaining/more-
overanswering tradeoff does. Extra SFT overanswers more than RL on046 by7/64
(+10.94pp, cluster CI[+3.85,+19.12]), while supported EM differs by only1/64.
This is a descriptive tradeoff, not evidence that RL solves the composition task.

Panels differ in more than depth, so do not attribute their score difference
causally to added hops. All32 new questions share atomic components with earlier
non-TRAIN study panels; none shares an official TRAIN atomic ID. Nevertheless,
172/756 unique title/text documents overlap official TRAIN, and63/64 variants
contain a TRAIN document. This is neither unseen-document evaluation nor a
guarantee against semantic/pretraining exposure. It is also not unseen-depth
extrapolation: RL's128 TRAIN parents included95 two-hop,26 three-hop and7 four-hop
questions, and the initializer was not restricted to two-hop tasks.

No decomposition architecture is compared: each policy directly returns an
answerability label and answer. No claim of novel abstention, RL superiority or
recursive planning follows from either panel.

## Cost and next discriminating control

| Policy | Native calls | Input tokens | Output tokens | Native service seconds |
|---|---:|---:|---:|---:|
| Warm joint32 | 128 | 391,778 | 1,387 | 136.39 |
| Paired RL8 | 128 | 391,778 | 1,458 | 139.97 |
| Matched extra-SFT8 | 128 | 391,778 | 1,511 | 143.90 |
| Total | 384 | 1,175,334 | 4,356 | 420.26 |

Owner wall time471.73s. Historical training is separate: RL1,627.75s and extra
SFT1,074.04s. Equal eight updates and1,024 response slots do not match information,
credited tokens or FLOPs. No new training was performed for this replication.

The bounded additive-versus-product reward proposal in
`PAIRED-ADDITIVE-RL-PLAN.md` is more discriminating than simply adding updates:
on exactly retained037 samples, additive scalar reward activates98/128 groups
versus42 for product and52 for the objective-preserving pairing-mean estimator.
But46 of the56 additive-restored groups have no positive exact successes, so
extra credit may mostly train refusals. Any prospective acceptance should keep
paired EM primary, report supported answering and negative overanswer together,
freeze a new held panel before updates, and avoid selecting new TRAIN examples
from039/046 outcomes. This is a conditional objective test, not a promised remedy.

## Evidence and reproducibility

R is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Immutable report `R/analysis-sufficiency-compositional-rl-001.json/.md`, JSON SHA256
`a8ae643e3c6db4279e740a20dec11a37e8bf4579096bdde922a6547048715517`.
Native output `R/sufficiency-compositional-readout-001`; cases
`R/sufficiency-compositional-inputs-001/cases.jsonl`, SHA256
`b43ba4e36a92014f49752681b26f0f98697170840e0d4c735249cf96e243f9c2`.
Prior039 report: `R/analysis-sufficiency-heldout-rl-001.json`, SHA256
`a5276f2be83e54271d2582d9ad5df5c65f41b976e47960f422c707e747e3a419`.

Analyzer002 initially stopped before writing because046 stores scorer hashes in
its frozen profile rather than the older manifest field. The failed-attempt
receipt is preserved at `R/analysis-sufficiency-compositional-rl-attempt-001.json`.
Additive analyzer003 authenticates that profile against PLAN, its frozen file
and entrypoint, the case/manifest/cluster identities, and the actual official
grader files. All old experiment outputs and analyzer002 remain unchanged.
Six focused tests pass, including the actual046 manifest and profile-tamper
rejection; Ruff passes. Analyzer003 SHA256:
`2228638de0a4055a7ee19e5f73ee1fa7e1a8df83d08cdc5c9c3298af3a31d50f`.

Exact command, with R as defined above (choose a new report path for any rerun;
the analyzer refuses to overwrite):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$R/source-037-sufficiency-rl" \
 /project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python \
 "$R/analysis-source-sufficiency-rl-003/analyze_sufficiency_rl.py" \
 --rl-output "$R/sufficiency-rl-001" \
 --sft-output "$R/sufficiency-extra-sft-001" \
 --held-output "$R/sufficiency-compositional-readout-001" \
 --held-cases "$R/sufficiency-compositional-inputs-001/cases.jsonl" \
 --report "$R/analysis-sufficiency-compositional-rl-001.json"
```
