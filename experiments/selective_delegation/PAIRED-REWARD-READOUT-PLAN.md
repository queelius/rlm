# Four-arm reward-control readout implementation plan

CPU preparation only; main owns acceptance, sealing and launch. This follows the
approved053 design, not a new experiment. The writing-plans and focused TDD
workflow will be executed inline in the existing isolated worktree.

Goal: read frozen050's32 parents/64 variants, two fixed seeds, four frozen adapters:
warm joint32, original product-RL8, original matched extra-SFT8, new additive-RL
terminal.512 calls, one hour, unchanged public prompt/T=.5/128 output/native client.
Primary metric remains paired EM with supported accuracy, abstention, overanswer,
protocol/unknown and costs; uncertainty retains29 frozen atomic clusters.

Architecture: optional explicit `additive_rl_output` in mutable shared reader,
new thin `eval_sufficiency_reward_control.py` profile entrypoint. Existing default
three-arm behavior stays intact; sealed039/046/003 remain unchanged.

- [x] Add failing focused fixture: equal shared warm/TRAIN/order/seeds/dose passes;
  wrong reward, seed, optimizer configuration or actual endpoint dose rejects.
- [x] Add named fourth adapter only after existing product-to-SFT provenance
  checks succeed. Never relabel old SFT as additive-matched. Require equal actual
  committed steps/cursors, exactly8/8; mismatch writes authenticated SKIPPED.json
  before GPU, scientific_readout=false, no checkpoint substitution.
- [x] Make loops and summaries use immutable PLAN conditions/call count. Preserve
  all adapter-enable/freezing and owner/deadline/transport-stop behavior.
- [x] Bind exact050 cases/manifest/scorer/29cluster profile and new entrypoint.
  Exercise actual native CPU tokenization and strict official scorer identity.
- [x] Exercise four named adapters via tiny native PEFT forward-hook fixture;
  assert T=.5, truthful receipt identity, all parameters frozen, no CUDA.
- [x] Run only the bounded reader fixtures and Ruff; main reviewed before sealing.

Common training configuration checks include base/model manifest, warm adapter,
TRAIN cases and manifest, blocks, seed, LR, optimizer reset/decay/clip, four
candidates,64-pair denominator, eight-block cap, sampling and context/output caps.
Product's absent historical objective/estimator fields mean product/diagonal;
additive must declare additive/diagonal explicitly. Matched update count is not
equal FLOPs, information, reward distribution or credited-token dose.

No result or adapter endpoint is invented while051 is running. The frozen panel
predates new training; no selection from039/046 outcomes. Any future four-arm
analyzer must accept512 rather than384 planned calls and must independently bind
the additional endpoint; analyzer003 is not silently treated as compatible.

## CPU readiness receipt

Reader sealed at `R/source-053-sufficiency-reward-control-readout`;11 focused
tests pass5.31s. Actual050 CPU token audit matches all64 frozen lengths,
1,963–3,782 input tokens, maximum3,910 including output cap. Actual051 versus
037 PLAN contracts agree; endpoint dose was a fixture, not a completion claim.
Ruff and CLI `--help` pass. Proposed decision with exact argv, source/input
hashes and output is `R/SUFFICIENCY-REWARD-CONTROL-READOUT-DECISION-001.json`.
Main alone accepts/launches. No fake prepare or PLAN was written before051 ends.

Independent readout analyzer sealed separately at
`R/analysis-source-sufficiency-reward-control-001`;10 focused tests pass0.30s.
It verifies512 native requests/grades, actual terminal/checkpoint identity,
original product-to-SFT binding, explicit additive reward, profile and fixed29
component clusters, then computes six exploratory contrasts (20,000 bootstrap
draws, seed2026092200) retaining both variants and both seeds. Primary contrast
is additive minus product RL. Costs/protocol/missing remain explicit. It does
not purport to audit the new additive TRAIN objective/gradients; that is separate.
Sealed analyzer003 remains unchanged and three-arm-only.

After a completed non-skipped readout, with R as the September21 run store:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$R/source-053-sufficiency-reward-control-readout" \
 /project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python \
 "$R/analysis-source-sufficiency-reward-control-001/analyze_sufficiency_reward_control.py" \
 --output "$R/sufficiency-reward-control-readout-001" \
 --cases "$R/sufficiency-reward-control-inputs-001/cases.jsonl" \
 --report "$R/analysis-sufficiency-reward-control-001.json"
```

The JSON/Markdown report is write-once. If051 commits fewer than eight updates,
the collector's explicit skip receipt is the result; no four-arm quality report
is manufactured. Preparing these tools generated no model outcomes or GPU calls.
