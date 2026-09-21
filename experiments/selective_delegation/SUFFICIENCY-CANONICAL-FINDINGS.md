# Fresh paired replication preserves the answer/refusal tradeoff

The fixed joint-SFT endpoint scores 11/64 paired exact matches versus base6/64;
positive-only SFT scores0/64. All384 native responses were available and valid JSON.
This is a fresh32-parent replication, with both official positive/negative variants
and two generation seeds, not64 independent questions or a selected checkpoint.

| Fixed policy | Pair EM /64 | Pair F1 | Both labels /64 | Positive EM /64 | Positive refusal /64 | Negative overanswer /64 |
|---|---:|---:|---:|---:|---:|---:|
| Base |6|.1128|16|20|25|24|
| Positive-only32 |0|0|0|29|0|64|
| Joint32 |11|.1914|17|16|41|6|

Joint minus base: paired EM+7.81 percentage points (95% parent-bootstrap interval
−1.56 to+17.19), paired F1+7.86pp (−1.56 to+18.10); seven wins and two losses,
all involving valid responses. Supported-answer EM decreases6.25pp (−14.06 to0).
Positive-only SFT increases supported-answer EM14.06pp (0 to+28.16), but answers
every insufficient-evidence variant, destroying the official paired score.

The earlier exposed panel had base/joint paired EM9/10 and positive-answer EM16/11.
The new panel makes joint's paired gain larger but still uncertain; it does **not**
reverse the conservative failure mode. Joint now refuses41/64 answerable attempts
(base25), while reducing negative overanswers24→6. Positive-only again never
refuses. Joint gains are not evidence of universally better reading or calibration.
Both answerability labels must be correct for official paired EM/F1; supported-answer
scores are separately reported to reveal this tradeoff.

This supports the already accepted bounded objective diagnostic: paired-outcome RL
versus dose-matched extra SFT from joint32, then a separately frozen held readout.
It does not justify more dose indefinitely or selecting an endpoint using these scores.
Abstention training and recursive RL are established topics, not novelty claims.

## Cost and method

Every arm used128 calls and363,332 input tokens. Output tokens were base1,778,
positive-only1,789 and joint1,423: total1,094,986 tokens for this384-call replication.
Native service time was91.32s,159.26s and133.87s respectively; this excludes loading,
tokenization and owner/analysis overhead. No historical base calls were reused here.

Sealed analyzer independently reconstructed native prompts/token IDs, decoded outputs,
seeds, adapter/COMMIT identities, and official grades. Twenty-thousand paired
percentile bootstrap draws, seed2026092190, resample32 parents while retaining both
variants/seeds. These are **parent-only** intervals, not verified atomic independence.
The later039 readout instead uses its frozen30-component partition and reports a
parent-only companion. Repeated-seed observations do not double the parent sample.

Artifacts under
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`:

- `analysis-sufficiency-canonical-001.json/.md`; JSON SHA256
  `cd9e8298a4a9c899b9c342abb9ebfdc76fd3c214b766e87cffd7ca17d0e1bd51`.
- `sufficiency-canonical-{base,joint,positive_only}-001`: immutable native receipts.
- `sufficiency-canonical-inputs-003/cases.jsonl`: SHA256
  `6dd8b93b974c1360113ffc27e2ee7a24f611117c224763fdfbb6e135e3c92a23`;
  selection seed2026092192 and explicit exclusions/exposure in its MANIFEST.
- `analysis-source-sufficiency-001/analyze_sufficiency.py`: unchanged sealed analyzer;
  source SHA256`c3e4f6df160c3aba365e73e73c930c43bffaf27c877051954512ec18fda3616b`.

Fresh selection is not proof of unseen pretraining, unseen documents, or independent
atomic facts. The provenance manifest, rather than the word “fresh,” defines exposure.
All32 retained parents are naturally two-hop under the strict selection process,
which excludes every official-TRAIN atomic component and the recorded prior panels.
This replication does not test deeper compositions; we have not isolated the
contribution of each exclusion rule to that selection outcome.
Of729 distinct public documents,183 exactly match official-TRAIN title/text;
58/64 variants contain at least one such match. Natural positive/negative contexts
also differ in length/content; this is not a controlled causal deletion experiment.
