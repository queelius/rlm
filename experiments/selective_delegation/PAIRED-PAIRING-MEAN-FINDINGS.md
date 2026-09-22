# Pairing-mean product RL control

The all-pairings conditional-mean credit estimator did not improve the fixed-dose outcome here.
On the already exposed frozen050 panel (32 parents, two seeds, 29 component clusters), the
recollected product-diagonal terminal scored 10/64 joint EM (15.625%) and pairing-mean scored
6/64 (9.375%). The predeclared pairing-mean minus diagonal contrast is -6.25 percentage points,
with component-cluster 95% CI [-12.90, -1.47] points. The four discordant cases are all
pairing-mean losses; 60/64 paired groups tie. Joint F1 is also lower (-5.47 points, CI
[-12.10, 0.00]). This panel was previously used for decisions, so it is exploratory rather than
a fresh confirmation.

This is a genuine estimator comparison, not an accidental reader change. Both arms returned all
128 planned calls with no protocol-invalid or unavailable calls, identical public prompts/seeds,
and 362,792 versus 362,771 total tokens. The new diagonal `rl_terminal` output exactly matches
the prior source053 product-RL episodes on all 128 available/prediction/protocol-error/request-
digest fields. The prior warm/SFT source053 arms remain cached context only; they were not
recollected or relabeled as controls.

The TRAIN audit shows a different credit pattern, not a learning benefit: pairing-mean had
50 effective parent groups over eight blocks versus diagonal's prior 42, but 312 nonzero
response coefficients versus diagonal's 336. It completed cleanly at 8/8 updates: 1,024 calls,
79/512 product-success pairs, 1,022 valid and 2 malformed-JSON protocol-invalid responses, with
no inference failures. Saved token likelihoods, LoRA deltas, and Adam commits were verified per
block. Thus denser active groups did not translate into better held joint correctness at this
unchanged dose.

The new model gives nine correct supported answers rather than 13, refuses 45
supported questions rather than 40, and answers six unsupported questions rather
than eight. Thus it shifts toward refusal, as the separate additive-reward run
did, even though this comparison keeps the product reward unchanged. This is an
observed behavioral similarity, not proof of a shared gradient-level cause.
The runs use one training seed. Cluster intervals describe uncertainty across
this small panel, not training-seed variation or new-dataset transfer.

Training used 2,908,957 tokens and 1,153.65 summed native generation seconds;
its owner elapsed time was 1,605.16 seconds. The two-arm evaluation used 725,563
tokens, 272.37 summed native seconds, and 304.63 owner seconds. Eight real
optimizer updates are distinct from the time spent generating/scoring samples.

The terminal-analysis watcher incorrectly required TRAIN-only step fields in
the readout summary. Its rejected receipt remains preserved; the collector
actually completed all 256 calls. The unchanged sealed analyzer subsequently
succeeded under `PAIRING-MEAN-ANALYSIS-INVOCATION-002.json`. No GPU rerun or
scientific-data repair was needed.

The smallest decision is to retire further unchanged product-estimator/dose variants; this is
not evidence that RL in general is futile. The next informative priority is the public-teacher
TextCraft comparison already queued: it changes the information-realizability of the learned
action policy rather than resampling this same sufficiency credit estimator.

Sources: `R/analysis-sufficiency-pairing-mean-training-003.json` (SHA
`b8bcce773e7fe183e6b515b48dac79e50b3d3422554add79ba7c2be1eba8393c`) and
`R/analysis-sufficiency-pairing-mean-001.json` (SHA
`dca61e6320de54639b4dc83af800bad54cdb943ef520cbd07b75193b98ba7df8`).
