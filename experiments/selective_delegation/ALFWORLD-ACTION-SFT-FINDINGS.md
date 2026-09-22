# ALFWorld action SFT: no transfer gain in this screen

Completed source042b used the fixed source041b checkpoint0033 (one epoch, 33 updates), with the adapter enabled only for flat/worker actions. Managers remained base. These are the same exposed 12 games, two seeds, six families and four scenes as source035—not a fresh confirmation panel.

| Policy | Successes / 24 | Calls | Total native tokens | Native seconds |
|---|---:|---:|---:|---:|
| Base flat | 4 | 1,052 | 1,692,327 | 443.4 |
| Base manager/worker | 4 | 1,362 | 2,197,259 | 677.7 |
| Trained flat | 2 | 1,114 | 1,829,189 | 758.6 |
| Trained manager/worker | 2 | 1,404 | 2,368,467 | 1,022.6 |

Both training contrasts have zero wins and two losses, a −8.33 percentage-point difference. Each loss pair is two repeats of one game: trained flat loses game10 (two soap bars), while trained manager loses game04 (clean bowl). Both trained arms retain only game02's vase-to-safe success, in both repeats. All outputs are observed; all four policies have zero invalid outputs. This is task-progress degradation, not improved JSON compliance hidden by missing episodes.

Parent/game bootstrap intervals for each training contrast are [−25, 0] points. Scene sensitivity is [−37.5, 0] for flat and [−15, 0] for manager. Exact game/scene sign-flip p-values are 1.0: the two losses are correlated repeats, not two independent negative replications. The interaction `(trained manager − trained flat) − (base manager − base flat)` is zero, with game CI [−25, 25] and scene CI [−15, 37.5] points. These small-cluster sensitivities do not establish either harm in general or a hierarchy benefit.

## Concrete failures

In game10, base flat completed both repeats in nine actions. Trained flat spends both 50-action budgets searching/examining and oscillating between receptacles; one repeat picks up a towel. Soap bars are visible in later public feedback, but neither run completes the requested transfer. In game04, base manager completed both repeats in 22 actions. Trained workers instead exhaust 50 actions: one mostly cycles shelves, the other picks up and moves a pot rather than completing the clean-bowl task. These are grounded action-trace examples, not proof that the adapter learned a particular latent heuristic or positional bias.

## Decision and provenance

Do not scale this action-SFT recipe or describe it as learned hierarchical competence. Base flat remains the cheaper ALFWorld reference. A small action-list permutation diagnostic could still distinguish positional interface learning from robust action selection, but this readout provides no performance gain to pursue with a larger dose.

The qualified independent analyzer verified native request/token/seed/role identities and replayed public actions; expected actor-adapter versus base-manager identity was checked before its explicitly host-only adapter-neutral harness projection. Checkpoint identity comes from the authenticated checkpoint receipts, not a new per-call weight measurement. New physical cost is 2,518 calls, 4,197,656 tokens and 1,781.20 native seconds. No unresolved starts or unlinked calls are reported.

Artifacts under `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`:

- `analysis-alfworld-trained-actor-001.json`: SHA256 `a4458ca86d6b3729f3246324872effa154a06dda16e7885772ad59e7a80cc5c9`.
- Trained terminal SHA256 `ffe786a4fcb8ac1f1ca72ec966c1426f9f9a0abf9db28f375c3b34c3e3bff989`.
- Source042b collector SHA256 `f15d69ca410f9d0ef184a907b05f1cb566ed63ea9b014ba2d7db252e83decdaf`.
- Base source035 terminal SHA256 `8e47575a23425608ebd2f0d06285ade8bac96ed825e89595e5aa5b1eb20b0ab4`.

CPU review on September 22 rechecked report totals, checkpoint step/epoch, changed-game traces and source/terminal identities. No model calls or source changes.
