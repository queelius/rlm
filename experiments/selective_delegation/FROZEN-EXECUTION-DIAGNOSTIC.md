# Frozen execution-noise diagnostic

Accepted September 21 at 14:20 UTC, queued after the direct-adapter and TRAIN-fit
readouts; collection had not started at acceptance. The question is whether a fixed plan receives
unstable downstream EM reward, making single-execution RL credit noisy.

It binds all 64 saved plans from `rl-fullpass-001/batch-0001`: 16 TRAIN parents
and four candidates each. Helpers use the frozen helper-SFT36 adapter; the final
uses the same base 4B model with that adapter disabled. No root generation,
optimizer, repair, silent retry, or host gold enters a generation prompt. Final
reports preserve training's `execution` and `steps` schema, including `step`,
question, resolved question, and actual predicted answer.

The realized frozen distribution is 47 two-step, 14 three-step, two four-step,
and one one-step plan. Thus the fixed maximum is **836** downstream calls:
four times 145 helper plus 64 final calls. Helper cap is `384/n`, final cap 128,
and temperature .5. All four candidates share each downstream seed. The new
diagnostic seed policy is `2026092161 + parent_index*10000 + repeat*1000`,
with helper offsets 1 through 8 and final offset 100. This separates steps,
executions, and parents and deliberately does not reproduce any training seed
collision. Preflight rejects overlap with historical batch request seeds.

The native runner reuses `eval_helper.HelperClient`, explicit generation
configuration, native token receipts, frozen PEFT routing, the exclusive GPU
lock, authenticated PID owner, and allocation deadline minus 600 seconds. Its
one-hour cap is cumulative across owners; unresolved started requests are not
retried. Source, cases, adapter, model manifest, and implementation hashes bind
the immutable PLAN. Returned protocol failures are observed reward zero;
unavailable generations have null reward and halt collection. All 256 planned
slots remain in completion and missing counts.

Primary analysis reports each plan's successes out of four, same-plan seed-pair
disagreements, and candidate RLOO advantage-sign and pairwise rank agreement
across seeds, with ties separate. These descriptive pairs are not independent
observations. Secondary analysis rotates each excluded seed `h`: the one-seed
selector uses `(h+1)%4`, the three-seed selector uses all other seeds, and both
evaluate only on `h`. Candidate-index ties are deterministic and receipt order
cannot alter selection. A parent with any missing/null slot is excluded from
selection/rank analyses, not from overall inventory. The selector uses TRAIN
labels and same-parent plans; it is not a deployable router or oracle.

The paired selector difference receives a connected-component cluster bootstrap
(20,000 draws; seed 2026092165). Actual native cost is reported separately from
historical root acquisition, counted once. Sixteen training parents do not
establish generalization or the efficacy of additional RL training.

Run the sealed copies with the training Python environment:

```sh
python frozen_execution_probe.py --source "$R/rl-fullpass-001/batch-0001" --cases "$R/inputs-001/cases.jsonl" --output "$R/frozen-execution-001" --hours 1
python analyze_frozen_execution.py --output "$R/frozen-execution-001" --cases "$R/inputs-001/cases.jsonl" --report "$R/frozen-execution-analysis-001/report.json"
```

Reports are immutable JSON with a readable Markdown sibling. Focused CPU tests
cover native PEFT enable state observed during forward passes, actual prediction
binding, final trace shape, held-seed leakage, missing values, deterministic ties,
receipt ordering, and reward-variability arithmetic.

Acceptance is recorded in `R/RL-DIAGNOSTICS-DECISION-001.json`, binding sealed
`R/source-016` and the authenticated serial supervisor `R/launch_rl_diagnostics.py`.
The actual report target is `R/analysis-frozen-execution-001.json`. The helper and
final policies are frozen throughout; no additional optimizer update is hidden
in this diagnostic.

## A reward change that would not change this update

The final model can sometimes solve the question without using the helper trace.
It is tempting to subtract a direct-answer score from each plan's reward to
reward only the value added by decomposition. With our current four-candidate
leave-other-three-out advantage, a shared per-question subtraction cancels:

`(r_i - b) - mean_j!=i(r_j - b) = r_i - mean_j!=i(r_j)`.

Thus one shared direct baseline per question would be useful for interpreting
performance, but would not change these root-policy gradients if all other
conditions stay fixed. It is not a new RL treatment worth an otherwise identical
training run. This statement concerns a common baseline, not different
candidate-dependent interventions. Repeated execution can change estimated
candidate returns; action-dependent computation costs can change preferences;
and adding a genuine finish-versus-delegate choice changes the decision problem.
Those require separate comparisons and are not accepted by this note. A final
answer gain alone still does not establish faithful use of a decomposition.
