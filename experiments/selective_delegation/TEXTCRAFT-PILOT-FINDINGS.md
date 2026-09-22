# TextCraft pilot: basic action/termination bottlenecks, no exercised recursion

Source044 ended at its one-hour owner cap, not a completed 32-episode comparison. The sealed independent analyzer passed native request/token/decode checks and trusted environment replay for observed episodes. All 1,239 recorded model calls returned without transport errors. One interrupted episode has native bindings checked but remains outcome-unknown; its incomplete state transitions are not claimed fully replayed.

| Policy | Successes | Observed failures | Missing | Interrupted/unknown | Planned |
|---|---:|---:|---:|---:|---:|
| Flat | 1 | 12 | 3 | 0 | 16 |
| Recursive interface | 3 | 9 | 3 | 1 | 16 |

Full-denominator success bounds are 1–4/16 and 3–7/16, respectively. Twelve fully observed pairs give three recursive-interface wins, one loss and eight ties; four pairs are unknown. Only six of eight task parents have complete paired repeats, so the analyzer correctly emits no complete-panel effect estimate or CI. No imputed failures or extrapolated full-panel rates are appropriate.

Twenty observed episodes ended at the 8,192-token context cap. Four finished successfully; one explicitly finished unsuccessfully. There were **zero delegate actions or child calls**, including the interrupted recorded trace. Any score difference reflects different root prompts/trajectories, not demonstrated recursion. Identical seed numbers do not make different prompts yield identical draws.

## What actually failed

Across all saved histories, 171 schema rejections comprise 169 multiple-JSON-object replies and two duplicate-field replies. Of 424 native action errors, 318 report insufficient inventory. Another 81 repeatedly pass extra `raw_o7` ingredients to the `o0_i1` recipe; remaining examples include omitted ingredients, incorrect scaled quantities, and one batch-size divisibility error. These are returned trusted environment errors, **not inference exceptions**.

The prompt already asks for exactly one JSON action, explains batch quantities and requires explicit finish. The bridge correctly executes one parsed action, returns native feedback and presents updated inventory in every subsequent prompt. `get_info.can_craft` reports that a recipe exists, not that current inventory can execute it; `in_inventory` and ingredients are separately public. That naming may be confusing, but no bridge mutation or hidden inventory mismatch was observed in replay. Every inspected multi-object reply is rejected whole—no partial execution or parser repair.

Two examples distinguish competence from mere formatting:

- `t00` / `c3_i2_23`: both flat repeats fail; both recursive-interface repeats finish, still entirely at depth zero. In flat repeat0, the model initially proposes the correct intermediate-plus-target action sequence but emits several JSON objects together. Later it repeatedly crafts the same intermediates, exhausts ores, and repeats rejected actions despite updated inventory. Recursive repeats also overproduce (9 and 12 target units for a request of 3), but eventually finish.
- `t03` / `m0_i2_20`: both flat repeats craft three target units by call006 for a request of one. Flat repeat0 only finishes after 79 calls; repeat1 never finishes before its context cap. Recursive repeats likewise reach sufficient inventory by call005 and then consume 82 and 81 additional calls without finishing. Native success requires explicit finish, so these remain failures where applicable; this is not a proposed score correction.

The analyzer counts 171 calls made with already-sufficient target inventory in each arm, 342 total across eight observed episodes. This includes the four required successful finish calls, so 338 are other post-sufficiency calls. Four observed failures reach sufficient inventory but fail to finish. Full untrimmed history carries repeated invalid/action text into later prompts and eventually fills context; this suggests a retention/attention confound, not proof that trimming alone repairs it.

A separate public-state CPU audit (`analysis-textcraft-finish-opportunity-001.json`, source `audit_textcraft_finish_opportunity.py`) checked every pre-action state of all 25 observed episodes against a hypothetical native finish on an inventory copy. The public net-inventory criterion agreed with the native checker throughout. Eight episodes had an opportunity; four were observed failures. Replacing continuation at the first opportunity by one finish call would avoid 334 recorded calls. This accounts for the necessary finish call even in formerly unsuccessful episodes. It is a conditional execution opportunity, **not 334 measured saved calls or eight model successes**, and ignores the seven unknown slots rather than extrapolating them.

## Smallest next comparison

First test a **flat prompt-only control on all eight fixed tasks and both seeds**, not only selected failure cases. Keep base weights, full history, public evidence, strict parser, native checker, 96-call/8,192-generated-token budgets, 256-token call cap and 8,192 context limit unchanged. Append one fixed trailing instruction after the state/history: “Choose exactly ONE next action and return ONE JSON object, then stop. Use current_inventory, not previous action text, to decide what is still missing. can_craft means a recipe exists, not that its ingredients are present. If all target increments are already present, return the finish action now.” Do not compute an oracle finish flag, execute multiple actions, repair JSON, or force delegation.

This is a post-hoc baseline qualification, not fresh confirmation or an isolated test of one sentence. Measure native success, protocol rejects, repeated native errors, post-sufficiency actions, context termination and actual cost. It can determine whether better action/finish salience cheaply rescues basic competence before spending a training run. Do not simultaneously trim history: changing retention would complicate interpretation and could remove recipes. If the prompt control fails, the already-frozen source047 action-SFT set is the next actionable option: 366 strict public-interface rows from 32 TRAIN tasks, including query/craft/finish and JSON+EOS targets, one fixed small epoch; evaluate flat first with a matched base prompt. Those demonstrations train successful trajectories, not recovery from the error-loop states observed here. They do not train recursive invocation.

## Costs and exact audit

Flat used 650 calls, 2,785,367 prompt tokens, 48,380 output tokens and 2,120.16 native seconds. Recursive-interface used 589 calls, 2,352,172 prompt tokens, 32,289 output tokens and 1,443.87 native seconds. Total 5,218,208 tokens and 3,564.03 native seconds; 3,600.04 owner-wall seconds. No unresolved starts or calls without an episode.

Artifacts under `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`:

- `analysis-textcraft-pilot-001.json` SHA256 `ea875de29d9e2e1af999027ff1eaf20080a1237fb94981d71a336ec35542aa79`; companion Markdown preserves unknowns.
- `analysis-source-textcraft-001/analyze_textcraft.py` SHA256 `e996cf530e7791bb15a22e6968e6763aff9a46ffb9d4405b44d9be0e06f011d2`.
- `textcraft-pilot-001/TERMINAL-436e4921c4bc.json` SHA256 `1d493943dee728c49004e3a95d13427c5d550ca7f647af95c56399c4469ac9f8`; failure `RuntimeError: TimeoutError: owner cap/interruption; incomplete episode unknown`.

CPU command: run that sealed analyzer with `--output R/textcraft-pilot-001 --report R/analysis-textcraft-pilot-001.json`, `PYTHONPATH=R/source-044-textcraft-pilot`, the existing training Python, offline Hugging Face settings and `CUDA_VISIBLE_DEVICES=''`. Completed successfully September 22. No GPU/model calls, no sealed-source edits and no training implementation.
