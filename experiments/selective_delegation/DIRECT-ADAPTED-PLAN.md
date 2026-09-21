# Direct answer with the trained helper adapter

Accepted by September21,12:42UTC after main source review and three focused CPU
tests, including actual tiny-PEFT routing and baseline generation equivalence.
Sealed source015 and `DIRECT-ADAPTED-DECISION-001.json` pin the queued job;
`launch_direct_adapted.py` waits for authenticated Hotpot helper-transfer release.
The external decision's12:45 timestamp was mistakenly entered ahead of the clock;
the session checkpoint records the correction. Scientific inputs are unchanged.
Question: does helper-SFT36 improve answering the original composed question
without decomposition? The current untrained direct baseline cannot distinguish
better reading from a benefit specific to planned helper execution.

## Fixed comparison

Use every parent in authoritative `fresh-dev-inputs-003/cases.jsonl`, SHA-256
`6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153`:
64 development parents, two paired repeats, two arms:

| Arm | Single answer model | Adapter |
|---|---|---|
| `base_direct` | Pinned Qwen3-4B-Instruct-2507 | Disabled |
| `helper_sft_direct` | Identical base instance | Frozen helper-SFT36 enabled |

Each attempt is exactly one `direct_answer` call. Both see the identical
`eval_planner.direct_prompt`: original question, all public documents and the
same short-span answer-JSON instruction. No root SFT is loaded; no plan, helper
turn, reminder, repair, fallback or additional final answer is generated.

Sampling matches the existing direct evaluator: T=.5, top_p1, top_k0,128 output
tokens; seed `eval_planner.SEED + int(digest(case_id)[:6],16) + repeat*100 + 2`.
Retain8192 context limit without truncation, native token receipts, at most90s
per call, first-response check, exclusive GPU lock, cumulative one-hour cap and
allocation deadline minus600s. Maximum256 new calls; no implicit retry of an
unresolved start. Adapter COMMIT/config/weights/STATE, helper role, step36/epoch1,
base manifest, source/dependencies/environment and input identity are pinned.

`fresh-contract-direct-001` did not exist when this runner was prepared, so exact
reuse could not be verified. This implementation deliberately recollects the
128-call base arm. A tiny CPU fixture compares its native prompt/token IDs,
sampling, seed, disabled adapter and actual generated tokens with the existing
direct-call client. Production bitwise equivalence is not claimed; timings and
hardware kernels can differ. No speculative cache reuse is implemented.

## Readout and interpretation

Primary denominator:128 planned attempts per arm, not only valid answers. Report
official MuSiQue alias-max EM/F1, malformed answer and generation-failure counts,
missing/incomplete attempts, paired wins/losses split by valid/protocol status,
and physical/deployed calls and tokens. Missing outcomes remain explicitly
incomplete lower bounds. There are64 parents, not128 independent observations;
parent/component uncertainty estimates belong in the completed-result analysis.

Compare this direct-adapter effect with the fixed-root helper effect on matched
parents, while reporting the latter's extra calls. Strong direct gains weaken a
decomposition-specific explanation. Weak direct gains do not prove decomposition
is necessary: helper SFT was trained on isolated annotated subquestions, whereas
this control transfers its adapter to a differently worded original-question
prompt. Preserve that prompt-transfer caveat; do not tune a direct prompt after
seeing results. This is adaptive development evidence, not untouched confirmation.

Entry point, after acceptance:

```sh
python eval_direct_adapted.py --cases FRESH003_CASES \
  --helper-adapter HELPER_CHECKPOINT36 --output NEW_OUTPUT --hours 1
```

Seal `eval_direct_adapted.py`, `eval_helper.py`, `eval_planner.py`,
`prepare_plan_training.py`, `probe.py` and the focused test. Existing immutable
external metric/runtime dependencies remain in place. No change to planner mode
or sealed sources013/014 is needed.
