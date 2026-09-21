# Paired supervision trades answer coverage for negative-label compliance

All three completed policies returned128 valid/available native responses, with
zero protocol failures or unknown outcomes. Both trained policies are the fixed
step32 endpoints; no checkpoint was chosen by DEV score. Native prompts, token IDs,
decoded responses, seeds and adapter/checkpoint bindings passed the sealed audit.

| Policy | Official pair EM /64 | Pair F1 | Both labels /64 | Supported EM /64 | Positive abstentions /64 | Negative overanswers /64 |
|---|---:|---:|---:|---:|---:|---:|
| Base |9|.1680|17|16|24|23|
| Positive-only repeated SFT |0|0|0|25|0|64|
| Joint SFT |10|.1842|21|11|37|6|

Positive-only SFT answers every variant, so its improved supported-answer coverage
is accompanied by complete failure on the official negative labels. Joint SFT
substantially reduces those overanswers but refuses more supported cases and has
only one additional paired exact success over base. This is an objective/response-
propensity tradeoff, not demonstrated improved general grounding.

Paired parent-bootstrap differences (20,000 draws, seed2026092190; both seeds and
both variants retained together within32 parents):

- Joint minus base pair EM: +1.56pp,95% interval[−9.38,+12.50]; six wins/five losses,
  all valid. Positive EM:−7.81pp[−20.31,+4.69].
- Joint minus positive-only pair EM:+15.63pp[+6.25,+26.56], but positive EM:
  −21.88pp[−35.94,−9.38].
- Positive-only minus base positive EM:+14.06pp[0,+29.69], while pair EM:
  −14.06pp[−25,−4.69].

Exact rates among answered supported cases are descriptively similar—base16/40,
joint11/27, positive-only25/64—but these are different selected subsets, not a
controlled semantic-accuracy comparison. Official negatives can retain alternative
evidence; label compliance is not a perfect semantic support verifier. The32-parent
panel is previously exposed exploratory DEV, not64 independent examples or a fresh
confirmation; parent bootstrap does not establish atomic independence.

Each policy used128 calls and359,894 input tokens. Output tokens were1,780 base,
1,834 positive-only,1,423 joint. New trained-policy acquisition totals256 calls,
723,045 native tokens; historical base acquisition adds128 calls/361,674 tokens.
Training matched512 examples/32 updates, not target tokens, unique inputs or
information:6,316 target tokens joint versus7,512 positive-only.

The conservatism amendment remains a useful conditional question: can paired-outcome
training recover supported answers without restoring negative overanswers, beyond
additional paired SFT? The current result does not answer it. First check the tradeoff
on the exposure-audited fresh replication; require actual TRAIN reward variation,
finite sampling caps and a separately frozen held readout for any accepted RL pilot.
No novel calibration, decomposition or RL claim follows from this qualification.

R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Immutable report `R/analysis-sufficiency-training-001.json/.md`; JSON SHA256
`ccc8b4fb8440754a04ec7b6f42ed0feb2f228d4a7f821b754eb84400e87a9a03`.
Analyzer: `R/analysis-source-sufficiency-001/`, SHA256
`c3e4f6df160c3aba365e73e73c930c43bffaf27c877051954512ec18fda3616b`.
The report retains exact source, native receipt, model-manifest and endpoint hashes.
