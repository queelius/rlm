# Frozen-policy TRAIN fit diagnostic

Conditional CPU preparation only, not an accepted or launched GPU job. Does the
last committed full-pass RL root improve terminal reward on its original training
panel, under the identical execution contract? This separates observable reward
fit from the question of held-development transfer; neither result alone identifies
training dose, objective, or the best alternative architecture.

Use exactly the first 16 TRAIN parents and four candidate root seeds from
`rl-fullpass-001/batch-0001`. Reuse and officially regrade all 64 original
pre-update SFT48-root/helper36/base-final receipts (45/64 EM); do not recollect
the baseline. Choose the last committed checkpoint only once every training
owner has a terminal receipt, with terminal optimizer-step agreement. Never
choose it using held scores. Preserve terminal status/failure in the new PLAN;
a capped or failed training run is not described as a completed full pass.

The collector reuses `rl_planner.Client` and `rollout` with separate root and
helper model instances. All parameters are frozen, all calls use eval/no-grad,
and no optimizer is instantiated. Root uses the selected RL adapter at T=.8;
helper uses helper-SFT36 at T=.5; final uses root base with adapter disabled at
T=.5. Caps remain 128 / floor(384/n) / 128. Full-source prompts, strict JSON,
actual dependency answers, and the original batch-1 seed schedule are unchanged.
In particular the original helper-step2/final seed collision is retained for
this matched readout, unlike the separate execution-noise diagnostic.

Maximum64 root +512 helper +64 final =640 new calls; expected roughly270.
One exclusive owner, a cumulative30-minute cap, allocation margin600 seconds,
native call/start/token/adapter receipts, and no retries. Recorded completed
episodes may resume only after request reconstruction; interrupted calls cannot
be regenerated. Baseline replay checks reconstructed prompts, role, seed,
sampling, caps, adapter/model identity, token counts, and official grades. The
new root input token IDs must exactly match historical baseline inputs.

Primary analysis uses all64 planned candidates, paired EM/F1 differences,
16-parent averages and20,000 connected-component bootstrap draws (seed2026092167).
Missing outcomes stay explicit and count zero only in planned-denominator
metrics; they are excluded from protocol/content win-loss attribution. Report
protocol recovery separately from both-valid semantic changes, exact parsed-plan
and generated-root-token changes, new physical costs, and reused acquisition
cost once. Reports are immutable JSON plus Markdown. This is a TRAIN fit check,
not64 independent parents, a generalization estimate, or proof that more RL works.

```sh
python eval_rl_trainfit.py --source "$R/rl-fullpass-001" --cases "$R/inputs-001/cases.jsonl" --output "$R/rl-trainfit-001" --hours .5
python analyze_rl_trainfit.py --output "$R/rl-trainfit-001" --cases "$R/inputs-001/cases.jsonl" --report "$R/rl-trainfit-analysis-001/report.json"
```

If TRAIN fit improves but fresh development does not, dose/transfer becomes a
more plausible follow-up than assuming no learning. If neither improves despite
likelihood movement, weak terminal-reward learning remains an explanation before
invoking tree search. The single common-seed diagnostic cannot distinguish these
mechanisms conclusively; compare with the frozen-plan execution-noise readout.
