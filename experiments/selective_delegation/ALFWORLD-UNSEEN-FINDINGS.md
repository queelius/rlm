# ALFWorld unseen-game screen: no replicated manager advantage

The completed, independently audited screen gives flat **4/24**, manager–worker **4/24**,
and local reasoning **6/24** native wins. The earlier exposed eight-game manager advantage
does not replicate in this point estimate. Local reasoning's two additional successes
are uncertain and costly. This is not evidence of policy equivalence, learned hierarchy,
or a general architecture advantage.

All 72 planned episodes are retained: twelve official `valid_unseen` games, two seeds,
three frozen policies. Selection balanced two games per six task families, but covers
only four scenes. The six clean/cool/heat games share scene10. Both repeats of a game
and games within a scene are correlated; 72 is not an independent sample size. All arms
receive the same native admissible-action lists, an affordance-assisted comparison.

## Paired outcomes and uncertainty

| Contrast | Paired wins / losses | Difference | Game-bootstrap 95% CI | Scene-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Manager − flat | 2 / 2 | 0 pp | −25 to +25 pp | −37.5 to +15 pp |
| Local reason − flat | 4 / 2 | +8.3 pp | −16.7 to +33.3 pp | −25 to +50 pp |
| Manager − local reason | 2 / 4 | −8.3 pp | −33.3 to +16.7 pp | −56.25 to +12.5 pp |

The sealed analysis uses 20,000 resamples, seed2026092200, retaining both repeats and
original game weighting. Twelve-game and four-scene exhaustive exchangeable-sign
sensitivities give two-sided p=1 for manager−flat and p=.75 for the other contrasts.
Four scenes are too few for stable population uncertainty; these are exploratory
sensitivities, not confirmation. No contrast establishes a general improvement.

## Costs and observed failures

| Policy | Native calls | Prompt + output tokens | Output tokens | Native generation seconds | Termination |
|---|---:|---:|---:|---:|---|
| Flat | 1,052 | 1,692,327 | 9,190 | 443.4 | 4 wins; 20 action caps |
| Manager–worker | 1,362 | 2,197,259 | 14,528 | 677.7 | 4 wins; 20 action caps |
| Local reason | 923 | 1,616,782 | 40,828 | 1,700.5 | 6 wins; 15 token caps; 3 action caps |

Native time is summed measured model-call latency, not end-to-end wall time or a modeled
GPU saving. Manager uses 30% more total tokens and 53% more native time than flat without
additional aggregate wins. Local uses 4.5% fewer total tokens but 3.8× flat's native time:
its much larger generated output and call structure matter. Cost is not captured by
total tokens alone.

All 3,337 calls returned, with zero inference errors, unknown usage/latency, missing
episodes, unresolved starts, or unlinked calls. Native response IDs, request digests,
tokenization, decoding, seeds, public action transitions, and native won flags passed
the sealed replay. No arm trimmed history; maximum prompts were 3,209/3,154/3,261 tokens
for flat/manager/local, below8,192. Flat and manager had zero protocol-invalid outputs
and all calls ended in EOS. Local had 45 invalid outputs:31 violated the nonempty,
at-most240-character reason contract and14 were malformed JSON. Fourteen local calls
ended at their output cap, which may be reduced by the remaining episode-token budget.
These are observed protocol/budget failures, not missing observations. There was no
repair or retry. Local's protocol burden is part of this fixed contract, not proof that
reasoning itself is harmful.

## Grounded examples, including both repeats

**Useful manager trajectory, game04: clean a bowl and put it in a cabinet.** Manager wins
both seeds in22 actions; flat fails both at50; local fails at45/47 on token budget.
The manager changes from searching countertops to bowl inspection, cleaning, and placing
the cleaned bowl. The worker finds bowl1 on countertop3, cleans it at sinkbasin1, and
moves it into cabinet1. This is a successful sequence, not proof that every goal is
faithfully executed: seed2195's cleaning goal names a dishsponge, while the actual native
action is `clean bowl 1 with sinkbasin 1`. Local seed2196 eventually finds and cleans the
bowl too, but does not place it before exhausting tokens.

**Manager/worker failure despite repeated relevant goals, game10: two soapbars into a
garbagecan.** Flat wins both seeds in9 actions, fetching soapbars1 and3 from countertop1.
Manager fails both at50 actions; local fails at48/43 on token budget. Both structured
policies initially take a soapbottle rather than a soapbar. Manager seed2195 repeatedly
issues a goal to take soapbar2 from cabinet4, while worker actions oscillate between
cabinet4 and garbagecan1. Thus refreshing a relevant textual goal does not ensure
executing it. This single game supplies both manager losses against flat.

**Local reasoning can break an action loop, game01: examine a CD with the desk lamp.**
Local wins both seeds in7/20 actions: it eventually takes CD3 from desk2 and returns to
desk1, where the native environment reports success. Flat fails both at50; manager
also fails both at50, largely repeating lamp-use actions. Local's second seed itself
alternates between desks repeatedly before taking the CD, so the successful outcome
does not establish consistently efficient deliberation. These two wins are part of
local's three light-task successes; it wins3/4 simple-placement attempts and no other
family. All policies fail all eight combined cool/heat attempts.

These examples were chosen after complete analysis to illustrate observed contrasts,
not as independent evidence or a selection rule. Episode IDs are
`game-{01,04,10}-seed-{2026092195,2026092196}-{flat,manager_worker,local_reason}`.
Their raw observations, goals, actions and calls remain in the immutable run and are
hash-bound by the audit report.

## Decision and provenance

Keep flat as the cheapest qualified fixed baseline; do not promote manager on the old
026 result. The already-accepted actor-training factorial can test execution competence
without assuming hierarchy helps. Its key comparisons should retain the game/scene
dependence and ask whether training improves flat as much as manager. This screen does
not justify another manager-only prompt search or training run by itself.

Artifacts under
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`:

- Run: `alfworld-unseen-001`; terminal `TERMINAL-8aadea799fdd.json`, SHA256
  `8e47575a23425608ebd2f0d06285ade8bac96ed825e89595e5aa5b1eb20b0ab4`.
- Independent report: `analysis-alfworld-unseen-001.json` and `.md`; JSON SHA256
  `5ecddc3f5a93a5cdc6d2362954b924e204f0966d92dde9fe7854fb46729b538d`.
- Sealed analyzer: `analysis-source-alfworld-unseen-001/analyze_alfworld_unseen.py`, SHA256
  `bb0d52d86fd274ec94cace792df19db3244099a480c6e3cc46015851a7affda9`.
- PLAN SHA256 `ca766c1a54a972b2b785cfd5b9cad995600d38d4134e2b23aba75b1056f73a49`;
  selected input-manifest SHA256
  `4d69b397bb98e77f988af445f7fbdb7310e0369531495f7df4886d575cbcd7a9`.

The sealed analyzer command completed successfully on21 September2026 after the
authenticated terminal; no live source or raw receipt was modified and no GPU was used.
