# Fresh RL: allocation changes, wording, and final aggregation

RL allocates modestly more questions on three-hop examples, but **the five
answer gains are not all longer plans**. There are two wins with more steps,
two with changed same-length plans, and one with fewer steps. Exact TRAIN
paragraph exposure also does not contain all gains. These are associations in
the completed panel, not identified causes of the small RL–SFT difference.

All64 remains primary: 64 deliberately balanced parents (32 two-hop/32
three-hop), two repeats each, 47 connected atomic components. This is not the
natural whole-development mixture. All 384 SFT/RL/direct attempts are included;
official MuSiQue final EM/F1 reproduce the original completed audit. The primary
20,000-draw result remains RL 56/128 versus SFT 53/128, EM +2.34pp with interval
[−1.67,+6.25]. This secondary audit does not establish a general RL gain.

## All cases by metadata, not just the changed examples

| Stratum | Parents / clusters | SFT exact | RL exact | Direct exact | SFT / RL / direct F1 |
|---|---:|---:|---:|---:|---|
| All | 64 / 47 | 53/128 | 56/128 | 54/128 | .5104 / .5245 / .4864 |
| Two-hop | 32 / 29 | 31/64 | 29/64 | 26/64 | .5855 / .5730 / .4940 |
| Three-hop | 32 / 18 | 22/64 | 27/64 | 28/64 | .4352 / .4760 / .4787 |
| Any exact TRAIN document | 14 / 8 | 6/28 | 7/28 | 6/28 | .4056 / .3738 / .3204 |
| No exact TRAIN document | 50 / 41 | 47/100 | 49/100 | 48/100 | .5397 / .5667 / .5329 |

The secondary 10,000-draw component-bootstrap RL−SFT EM intervals are two-hop
−3.13pp [−8.06,0], three-hop +7.81pp [+2.33,+14.81], exact exposure +3.57pp
[0,+11.54], and no exact exposure +2.00pp [−3.00,+6.73]. Three-hop F1 difference
is +4.08pp [−4.08,+11.54]. These are unadjusted exploratory subgroup intervals,
**not an interaction test**; significance in one subgroup but not another is
insufficient evidence of different effects. The three-hop direct policy is
still 28/64 versus RL's 27/64, with much lower inference cost.

Exposure is literal `(title,text)` overlap with any TRAIN paragraph, including
distractors—not measured memorization. It is strongly confounded with hop count:
the exposed group contains 2 two-hop and 12 three-hop parents; the unexposed
group contains 30 and 20. Cross-tabulated exact counts (SFT / RL / direct):

| Hop / exposure | Parents | Exact answers per policy |
|---|---:|---|
| Two-hop, exposed | 2 | 2/4 / 2/4 / 2/4 |
| Two-hop, no exact exposure | 30 | 29/60 / 27/60 / 24/60 |
| Three-hop, exposed | 12 | 4/24 / 5/24 / 4/24 |
| Three-hop, no exact exposure | 20 | 18/40 / 22/40 / 24/40 |

Four of five winning parents have no exact paragraph overlap; the one exposed
win is the Molotov–Ribbentrop example. Both losses have no exact overlap. This
rules out the narrow description “all gains are on exact-exposed paragraphs,”
not broader semantic exposure or training effects. No new clean subset is
selected; the four cross-cells are descriptive and very small.

## Does RL allocate more helper steps?

Yes, specifically on the three-hop panel. Distinguish generated plan length
from helper calls actually reached before an upstream failure:

| Stratum | SFT → RL planned steps | Mean planned steps/attempt | Actual helper calls |
|---|---|---|---|
| All128 attempts | 297 → 307 | 2.3203 → 2.3984 | 295 → 307 |
| Two-hop64 attempts | 128 → 128 | 2.0000 → 2.0000 | 128 → 128 |
| Three-hop64 attempts | 169 → 179 | 2.6406 → 2.7969 | 167 → 179 |

On three-hop attempts, RL produces 12 longer, two shorter, and 50 same-length
plans; every length change is one question. Mean allocation rises by 0.15625
questions/attempt, component interval [+0.0571,+0.2800]. The two-hop panel has
one longer and one shorter plan, with no net increase. SFT's invalid dependency
on a three-hop attempt accounts for the extra two-question gap between planned
and actual helper calls. This is not two further planned allocation changes.

Full-panel generated plan lengths are SFT: one 1-step, 86 2-step, 40 3-step,
one 4-step; RL: one 1-step, 78 2-step, 46 3-step, three 4-step. Direct has no
generated plan or helper calls. The length-conditioned outcomes are below, but
the policies select different examples into these bins: **length is a
policy-dependent outcome, not a randomized treatment**.

| Generated length | SFT exact / attempts | RL exact / attempts |
|---|---:|---:|
| 1 | 1/1 | 1/1 |
| 2 | 37/86 | 33/78 |
| 3 | 15/40 | 20/46 |
| 4 | 0/1 | 2/3 |

## Paired changes distinguish the candidate explanations

| RL versus SFT length | Attempts | RL EM wins / losses |
|---|---:|---:|
| More steps | 13 | 2 / 0 |
| Same length | 112 | 2 / 2 |
| Fewer steps | 3 | 1 / 0 |

Of the 112 same-length plans, 57 are exact same question lists and 55 have
different strings. None of the seven EM changes occurs with an exact same
list. “Different strings” can mean a changed referent, ordering, or relation,
not merely a harmless paraphrase. With more steps, the helper budget remains
384 total, so the per-call cap changes; extra calls also add input tokens.

The five winning examples already documented in `FRESH-CONTRACT-FINDINGS.md`
fall into three categories:

- More steps: Spain/treble (2→3) and collaborator population (3→4). Neither is
  a clean faithful-chain demonstration: the former still binds Spain as the
  team; the latter's final says 6 million despite its last helper saying
  10 million for Belgium.
- Same length: Molotov publication (3→3) and Stan's birthplace (3→3). The first
  preserves a more specific requested relation in one repeat; the second
  explicitly inserts South Park, bypassing a wrong first helper answer.
- Fewer steps: Geisha/runways (3→2). The SFT last helper says Soviet Union but
  its final says Japan; RL's last helper says Cold War and its final says
  Soviet Union. This is not evidence that a longer decomposition is necessary.

All five wins and both losses have valid finals. Wording and answer binding
can plausibly matter, and aggregation can overturn helpers, but the existing
rollouts do not isolate their causal contributions. The matched frozen-execution
diagnostic is appropriate; no new architecture conclusion follows from these
post hoc length bins.

As a mechanical check across all128 attempts, a correct final coexists with a
last-helper answer that does not exact-match the original gold on six SFT and
ten RL attempts; an incorrect final coexists with an exact-matching last helper
on five SFT and two RL attempts. On three-hop these counts are 1→5 and 2→0.
The last helper is **not annotation-aligned and need not answer the original
question**, so these are discordance counts, not scored intermediate accuracy
or a measured number of causal final “rescues.”

## Reproduction and boundary

`audit_fresh_rl_structure.py` reads only the completed original policy report,
its hash-bound SFT/RL/direct episode/final receipts, the frozen cases, and the
existing exposure audit. It reuses the existing component/bootstrap and official
metric functions, preserves all parent/repeat rows, and asserts aggregate
agreement with the original report. No checkpoint weights are rehashed.
The existing adapted-direct-dependent `analyze_fresh_strata.py` is unchanged;
only its pure partition/cluster restriction functions are reused. No adapted
control or active-job outcomes were inspected.

Executed from the selective-delegation worktree with
`/project/alex_phd/envs/prime-rl-5990b1b/bin/python`:

```sh
python experiments/selective_delegation/audit_fresh_rl_structure.py \
  --root /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921 \
  --report /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-fresh-rl-structure-001.json
```

The completed immutable JSON/Markdown include all384 per-policy rows, all128
paired changes, generated-length bins, nine strata, and 781 consumed-source
hashes. JSON SHA256:
`e1f609ce964a72db3d7b86fc3c0942c56c0039f29d40f412ceb721c855795539`.
Secondary intervals use 10,000 draws, seed 2026092123; their Monte Carlo endpoints
may differ slightly from the primary report's 20,000 draws/seed. Ruff check and
format pass. No model calls, live-source edits, or new experiment launch.
