# Matched Hotpot direct-answer adapter control

Accepted September21 at15:10UTC (rounded). Source019 and the external
`HOTPOT-DIRECT-DECISION-001.json` bind inputs and source. Supervisor61945 waits
for the plan-only owner to release before collecting. No new training.

Question: does helper-SFT36 improve answering the original Hotpot question
directly? The completed frozen-root helper result (40/64 versus base22 and
reminder35) is not itself evidence that decomposition beats adapted direct
reading. The historical direct36/64 used different seeds. The fresh MuSiQue
direct-adapted result (49/128 versus base54) does not settle Hotpot transfer.

## Fixed comparison

- Panel `hotpot_explorer32`: all32 unique transfer parents in
  `hotpot-inputs-001/cases.jsonl`, dataset `hotpotqa`; SHA256
  `f16bc99b6b786920a5fecc516d6e2ecfc7c566cf0c38377764e7a8469e424417`.
- Two repeats, two arms: base weights versus frozen helper-SFT36 enabled for
  one direct-answer call. Same `eval_planner.direct_prompt`, full public source,
  short-span strict answer JSON; no root adapter, planner, helper calls, extra
  final, reminder, retry, or repair. One resident4B PEFT model; all parameters frozen.
- Seed `2026092112 + int(digest(case_id)[:6],16) + repeat*100 + 2`, exactly the
  completed helper-transfer final seeds. Temperature0.5, top_p1, top_k0,
  output128, per-call max_time90 seconds; input never truncated.
- Maximum128 physical calls,64 planned attempts per arm; cumulative20-minute
  cap and allocation-end margin600 seconds, existing exclusive GPU owner/lock.
  Hotpot full summaries every16 episodes and at termination; lightweight status
  otherwise. Default `fresh003` retains its original seeds, panel and summary cadence.
- Adapter `helper-sft-001/checkpoint-0036`, fixed one-epoch checkpoint; its
  identity and training/model manifest are verified and recorded. No selection
  among helper checkpoints from these outcomes.

Official Hotpot EM/F1 is shared with `eval_helper.grade_final`, including
exact-only yes/no/noanswer F1. Both arms retain all planned attempts; missing,
protocol and generation failures are separate. Analyze paired repeat means
within parent,20,000 bootstrap draws, seed2026092116. These32 parents have no
component IDs: singleton-parent bootstrap is not verified atomic independence.
Report both-valid versus protocol-involved wins/losses and native call/token cost.
This panel rejects the analyzer's optional historical baseline/planner inputs;
their prior schema/seeds are not silently treated as paired evidence.

## Main's launch and analysis commands

Use the sealed source path in place of `$SRC`; `$R` is the existing selective-
delegation research store. The training environment is unchanged.

```bash
$PY $SRC/eval_direct_adapted.py --panel hotpot_explorer32 \
  --cases "$R/hotpot-inputs-001/cases.jsonl" \
  --helper-adapter "$R/helper-sft-001/checkpoint-0036" \
  --output "$R/hotpot-direct-adapted-001"
$PY $SRC/analyze_direct_adapted.py \
  --cases "$R/hotpot-inputs-001/cases.jsonl" \
  --output "$R/hotpot-direct-adapted-001" \
  --report "$R/analysis-hotpot-direct-adapted-001.json"
```

Omitting `--hours` selects the panel's20-minute cap; explicit larger values are
rejected. Reports are immutable JSON plus Markdown siblings. Seal the usual
direct/helper/planner/probe dependency closure plus `score_hotpot.py` and its
official evaluator reference; PLAN and analysis record the scoring hashes.

CPU checks:10 focused tests pass, including actual tiny-PEFT forward-hook routing
under both seed bases and native saved-request/official-grade analysis. Ruff and
CLI help pass. The authoritative32 cases validate as `transfer`/`hotpotqa`; all64
seeds match completed helper-transfer native finals. Actual tokenizer inputs span
1,101–2,355 tokens (maximum2,483 including the128-token output cap), with no
truncation. No GPU inference was used for readiness.

Interpretation: strong adapted-direct performance would weaken a
decomposition-specific interpretation of helper transfer. Lower direct scores
would motivate a matched architectural comparison, not prove faithful
intermediate reasoning. The helper adapter was trained on gold-bound
subquestions and is now applied to an original-question prompt; this distribution
shift is deliberate but limits interpreting a null direct effect. The64 sampled
attempts are not64 independent parents or a newly untouched confirmatory set.
