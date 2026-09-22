# Action SFT fixes JSON but leaves discovery and premature stopping failures

Source052 completed all 32 fixed flat episodes: eight exposed VAL tasks × two
seeds × original/reminder prompts, using the preselected source048 checkpoint23.
The terminal is clean. The sealed independent analyzer replayed all native
prompts, token IDs, seeds, model/adapter receipts, public transitions and scores.
There are no missing episodes, unavailable calls, unresolved starts or unlinked
calls. Analysis began only after authenticated resource release.

## Discovery regresses despite better formatting

The matched native query audit makes the tension explicit: the trained policy
emits cleaner actions but is much less likely to begin by inspecting its public
goal, and frequently queries names that the environment says do not exist.

| Completed condition | Root query on first call / first valid query | Ever queries root | Nonexistent query calls / valid queries |
|---|---:|---:|---:|
| Base original | 13/13 / 13/13 observed | 13/13 | 0/272 |
| Trained original | 3/16 / 3/16 | 11/16 | 344/522 |
| Base reminder | 16/16 / 16/16 | 16/16 | 0/81 |
| Trained reminder | 3/16 / 3/16 | 11/16 | 593/1078 |

**On the 13 matched observed original-prompt slots, root-first falls from13 to2,
and ever querying the root falls from13 to8.** The three missing base slots remain
unknown—not failed root queries. Reminder has all16 matched slots: root-first
falls16→3 and ever-root16→11. There are no reverse transitions for either metric.
The full trained original counts above include its three unmatched slots.

Nonexistent queries occur in10/16 trained original and14/16 trained reminder
episodes, with264 and509 repeated nonexistent-item mentions respectively. These
are query/action counts, not independent observations. A strict-schema query is
“valid” here even when its requested item does not exist; the count excludes
known base materials and uses the actual empty-recipe/non-base native feedback.
Root names come only from the first public goal frame. For t00/r0, both base
prompts first request root `c3_i2_23`; trained original instead requests nonexistent
`c1_i1_1`, and trained reminder requests nonexistent `c2_i1_1`.

Thus information gathering regresses alongside the observed format gain; an
increase in valid JSON is not sufficient evidence of better planning. This is an
adapter-associated behavioral change on an exposed eight-task panel, **not proof
that privileged teacher action order caused it**. Teacher histories, item-name
memorization, the broader policy shift and termination/budget differences remain
possible explanations. Root-first is not necessary for every success (see t06
below), and no future public-teacher outcome is assumed. All61 observed condition
slots were audited, without selecting successes or failures.

## Paired result, not a solved planning benchmark

| Contrast | Observed successes | Paired wins / losses / ties | Unknown pairs |
|---|---:|---:|---:|
| Trained original versus base original | 3/16 versus 1/13 observed | 1 / 0 / 12 | 3 |
| Trained reminder versus base reminder | 2/16 versus 0/16 | 2 / 0 / 14 | 0 |
| Trained reminder versus trained original | 2/16 versus 3/16 | 1 / 2 / 13 | 0 |

The original-prompt comparison retains the three missing base slots as unknown;
there is no complete-panel effect estimate or CI. Its one known win is t05/r0;
t03/r0 is a retained success. The trained t06/r1 success has an unknown baseline,
so it is not a second observed training win. Reminder training improves by 12.5
percentage points, task-cluster bootstrap 95% CI [0,31.25]; adding the reminder
to trained weights changes success by −6.25 points, CI [−25,12.5]. These are only
eight exposed tasks with correlated repeats in one recipe world. The intervals
do not establish general competence gains. No child executes in any episode,
and this is not evidence for recursion or a new hierarchy.

## Protocol improves; most trained failures are not missing finish

| All 16 episodes per trained profile | Original | Reminder |
|---|---:|---:|
| Successful native tasks | 3 | 2 |
| Invalid JSON/schema replies | 0 | 0 |
| Native action errors | 52 | 90 |
| Queries / crafts / finishes | 522 / 109 / 14 | 1078 / 155 / 4 |
| Rejected delegation attempts | 1 | 1 |
| Finished but unsuccessful | 11 | 2 |
| Global-call / context-cap terminations | 2 / 0 | 6 / 6 |
| Root queried in first action | 3/16 | 3/16 |
| Root never queried | 5/16 | 5/16 |

The original flat baseline had 118 schema failures over its 13 observed episodes;
base reminder had four over all 16. Action training produces valid JSON throughout
both trained profiles, but valid actions and task completion remain distinct.
Querying accounts for 81% and 87% of the two trained call budgets. The reminder
reduces premature finish but often trades it for long query loops, not success.

The qualified replay's public-goal-attainment count is particularly useful:
**none of the 27 failed trained episodes ever reaches sufficient net target
inventory before another action**. Only the five successful episodes do. All
three original successes finish immediately afterward; one reminder success
also does, while t03/r0 takes 13 calls after readiness including its eventual
finish. Thus the positive public-goal stopping rule proposed earlier would rescue
zero observed trained failures and could shorten one successful trajectory. This
is a retrospective opportunity statement, not a new policy result. Rejecting an
early finish would be a different controller intervention with unknown subsequent
outcomes; it cannot be claimed as a rescue from these traces.

## Grounded successes and contradictions

The following outcomes include both repeats; call numbers are zero-indexed.

- **t03 (`val.19`), original 1/0 and reminder 1/0:** original repeat1 queries
  `m0_i1_10` first and crafts it at call001, consuming all two `m0_ore` units.
  Only at call003 does it query root `m0_i2_20`, whose recipe needs the different
  intermediate `m0_i1`. That item also needs the now-exhausted ore. It ultimately
  finishes at call022 with the wrong intermediate and no root. This is an actual
  irreversible wrong-product choice, not merely missing JSON or a stale current
  inventory. Repeat0 succeeds, so the failure is not universal for this task.
- **t00 (`val.325`), original 0/0 and reminder 1/0:** reminder repeat0 initially
  guesses nonexistent names, then queries the root at call004, discovers both
  prerequisites, crafts the root at call011 and finishes at call012. Repeat1
  instead spends 93 of 96 calls querying candidate names, never queries the root,
  and ends with unchanged inventory. Its 41 queries for nonexistent `c1_i1_19`
  illustrate a loop despite repeated empty-recipe feedback. One successful
  recovery is not evidence that the policy reliably follows public dependencies.
- **t05 (`val.435`), original 1/0 and reminder 0/0:** original repeat0 starts
  with its root `o8_i4`, discovers prerequisites, and eventually completes and
  finishes in 19 calls. It makes rejected premature crafts and must replenish
  a shared intermediate, so this is not an error-free plan. It is the one known
  original-prompt training win. The other repeat and both reminder repeats fail.
- **t06 (`val.207`), original 0/1 and reminder 0/0:** original repeat1 succeeds
  in 40 calls after substantial exploration and actual multi-step crafting.
  It starts from an intermediate rather than the root. This is a counterexample
  to claiming root-first action is necessary for success, and its base slot is
  missing. Root-first demonstrations remain a testable training hypothesis,
  not an established explanation of every success or failure.

These examples come from saved native action traces, not an inferred faithful
chain from final answers. The training coverage audit shows fixed-world overlap:
21/47 required VAL recipes occur in TRAIN demonstrations, while none of the eight
VAL roots does. This panel is not unseen-recipe-world or out-of-range-depth transfer.

## Next decision

The smallest useful **learning** comparison is now the CPU-qualified source055
public root-first demonstrations against the fixed privileged-order source048
training procedure, on the same 32 TRAIN tasks, with a separately accepted fixed
endpoint. Both procedures naturally have 366 rows and 23 one-epoch updates, but
their histories and prompt-token exposure differ. Do not call it a perfectly
compute-isolated ordering intervention. The all-query visibility audit finds
135/167 old teacher query names absent from prior public prompts versus zero for
the public teacher; that aligns with the observed guessed-name failures but does
not causally prove the teacher induced them. Arbitrary item queries are legal.

Prioritize discovery/quantity competence over another stop-only readout. Current
inventory was already visible, and the proposed public positive-goal guard does
not repair the trained failures above. A queried-recipe cache may still help use
available information, but cannot supply unqueried dependencies or undo exhausted
materials. Keep it a conditional presentation control, not a new memory claim.
Do not change source048/052 or retrospectively choose another checkpoint. This
recommendation accepts no GPU job or additional training by itself.

## Costs and immutable provenance

Original: 646 calls, 1,859,461 prompt tokens, 12,729 output tokens, 1,080.44 native
seconds. Reminder: 1,238 calls, 4,939,103 prompt tokens, 22,746 output tokens,
2,098.54 native seconds. Total new cost: **1,884 calls, 6,798,564 input and 35,475
output tokens, 3,178.98 native seconds**; owner wall time 3,228.47 seconds. Reused
base costs are not new work. The shorter original trajectories partly reflect
premature termination; lower cost is not proof of better planning efficiency.

Artifacts under the existing research-store root `R`:

- Report `analysis-textcraft-trained-readout-001.json`, SHA256
  `f84ea2aa2a86c1190fe4db0cb069242f0c1e88e35849e6a1a63eca819f8e9887`.
- `textcraft-trained-readout-001/TERMINAL-7396b0fc0269.json`, SHA256
  `431d663b647440919c7bd66aee900de620912ac80f738cc9f26e3d774ce88443`;
  failure null, stopped false. PLAN SHA256
  `c060faeb0efb5fb78d07767870d847af51000d25822060916d1eff801f1dcc2f`.
- Fixed `textcraft-action-sft-001/checkpoint-0023`: step23, epoch1, cursor0;
  adapter SHA256 `471a108441970ad157096e37c4218f17a9827b1e1f6d3b94fda7b6de80753f2c`,
  COMMIT SHA256 `749d9d2f57354586b76e33cb98554674fd2a54b1d23ac1305e663da7c421537f`.
- Expected collector SHA256
  `117738164fa27f46e69361690e20890999ce799c293e59fde44c40167a8e41ba`;
  sealed profile analyzer SHA256
  `1c4062a2e06a307a38173d28dab405e497aba8576fc5d170dc3f9f29acaca88b`.
- `TEXTCRAFT-TRAINED-ANALYSIS-WATCH-RESULT-001.json` records authenticated release,
  exact analyzer command, exit0 and report hash. Historical baseline hashes remain
  in the report; its three original-flat unknown slots are unchanged.
- Query-transfer report `analysis-textcraft-query-transfer-001.json`, SHA256
  `d2a961fe22a1b661a1156a8dc61bbc95fb4d55b0cf15bc83ca4cb1ef1401481d`,
  is preserved unchanged. Standalone `audit_textcraft_query_transfer.py` and its
  actual-data fixture reproduce its group summaries, matched contrasts, per-slot
  rows and native-receipt inventory hash exactly. Source seal:
  `analysis-source-textcraft-query-transfer-001/SOURCE.json`, SHA256
  `33f58ee8f034f72a9d503ae8bf47c56893a7a88eeae8f2e4524ac1f956e53e4f`;
  auditor SHA256 `27c4b6b97f6ecf5ce1d4559416907cfbeb636e5c4217815dd28be8533f5ac69f`.
  The source052 strict parser is copied byte-identically. One focused sealed
  actual-data test passed in3.54s; Ruff and the actual CLI replay passed.
  Additive replay receipt `analysis-textcraft-query-transfer-replay-001.json`
  SHA256 `8baaa6a0abfa6a0578968a1fd4ffbdbcf3db17ccc71731f8c209676ec9997714`.

Reproduce without writing any report: using the existing training Python with
`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1`, run
`R/analysis-source-textcraft-query-transfer-001/audit_textcraft_query_transfer.py --verify-against R/analysis-textcraft-query-transfer-001.json`.
An optional `--report NEW_JSON_PATH` writes once and rejects overwrite. The audit
binds every used call/node/episode to its earlier independent native audit;
it does not generate new trajectories or inspect later058 training batches.
