# Product pairing-mean RL control

Question: with the same TRAIN input, product reward, warm checkpoint, sampling seeds, Adam
schedule, and eight 16-parent blocks as diagonal source037, does replacing arbitrary
candidate-index pairing with the all-pairings conditional mean alter the learned terminal policy?

The accepted run is `source-058-sufficiency-pairing-mean-rl` (1,024 calls, 45 minutes), followed
only if it reaches a clean committed 8/8 terminal by a two-arm 256-call, 30-minute frozen050
reader in `source-059-sufficiency-pairing-mean-readout`. The primary readout is official EM:
pairing-mean minus diagonal, clustered over the panel's 29 components. Existing source053
warm/SFT results are cached context, not newly paired controls.

This tests the realized clipped-Adam update, not merely the prior CPU result that conditional
mean credit reduces variance of an unnormalized raw-gradient estimator. It is exploratory: frozen050
has already informed prior decisions, so either a favorable or null result warrants a new panel before
a general claim. A terminal below eight updates, interrupted owner, or unavailable native call yields
no substitution or scientific comparison.

## Acceptance and handoff

Main accepted the two jobs on September22 at08:03UTC through
`R/INDEPENDENT-TRAINING-QUEUE-003A.json`, SHA256
`61ec8b7f1173f3af93737183ed80b647f0db553b817a38fc0b0ce3de5962c5ec`.
They wait for authenticated resource release from the crafting readout052,
not for a favorable crafting result. The original unlaunched003 receipt remains;
003A adds pins for unchanged inherited reader imports without changing the jobs.

Actual037 and prepared058 PLANs match on every shared training-contract field.
The trainer bytes match qualified051; only the selected estimator/reward profile
differs from that additive run. Nine focused sealed reader/analyzer tests passed
in5.32seconds. Source058 training and source059 readout have distinct immutable
source/output directories; source051/053 and their completed results are unchanged.

The finite queue has a six-hour allowance including predecessor wait, with
separate45-minute training and30-minute readout caps. A CPU-only four-hour watcher
authenticates release, preserves incomplete work and runs the sealed readout
analyzer. It writes `R/PAIRING-MEAN-ANALYSIS-WATCH-RESULT-001.json` and
`R/analysis-sufficiency-pairing-mean-001.json` when qualified. R denotes the active
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921` store.

This revises a tentative recommendation to defer estimator work: demonstrating
an RL-over-SFT advantage is not a logical prerequisite for this optimizer
diagnostic. One bounded run can distinguish no practical effect from a useful
same-objective change. It does not justify an indefinite estimator sweep.
