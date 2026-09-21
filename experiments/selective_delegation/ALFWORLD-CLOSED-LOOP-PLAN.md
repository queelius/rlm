# Closed-loop ALFWorld interface qualification

CPU-prepared, sealed and **proposed, not accepted or launched**, September21.
Main requested this combined interface after the completed screen had76 valid-JSON
but inadmissible-command errors. It is not an isolated causal ablation or new
hierarchy/RL method. Repeat all eight original seen-development games ×two seeds
×flat/manager-worker =32 slots; no favorable-game selection or held-out claim.

Actions now return exactly `{"action_index": integer}` with a zero-based index
into the **current** numbered native admissible list. The host maps that index
to the exact command; no repair/fallback/nearest match. Booleans, floats, strings,
duplicate/extra fields and out-of-range indices are invalid. Manager output stays
`{"goal": string}`. On any returned schema/index/manager error, the public history
gets the raw rejected response as data and explicit controller rejection feedback;
environment state does not advance. Both policies see the same information at an
identical state. This intentionally combines numbering and rejection feedback.

Retain original seeds, base4B weights, T=.5/top-p1/top-k0, manager refresh every
four executed actions,50-action/2048-generated-token episode caps,128 output
tokens/call and8192 context. Manager and rejected-response tokens are charged.
History trimming uses the common largest neutral role prompt plus512 goal-token
reserve and128 call-token reserve. Three consecutive invalid responses stop the
episode. Smoke requires one successfully executed action in each first-game/
first-seed policy, not task success. Native/inference failures remain unknown.

Implementation: separate `alfworld_closed_loop.py`, reusing unchanged native
`BaseClient` and `Bridge`; bounded owner/load lifecycle follows the original.
Existing source022 and live owners are unchanged. Six focused CPU fixtures passed,
including real saved command-list indexing, bool rejection, flat and manager
error-feedback round trips, matched public projection, native tiny-model response
decoding and proposal launch refusal. This is not a new4B GPU smoke result.

R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`:

- Sealed source: `R/source-026-alfworld-closed-loop/`.
- Immutable CPU PLAN: `R/alfworld-closed-loop-001/PLAN.json` (32slots).
- Proposed decision: `R/ALFWORLD-CLOSED-LOOP-DECISION-001.json`; hashes every sealed
  Python file, new PLAN and original screen PLAN. Main alone may accept/launch.
- Launcher waits for authenticated clean `R/hotpot-fresh-direct-vote-001` release,
  plus512/512 available new calls; refuses failed/capped/incomplete predecessor.
  One-hour cumulative owner cap, allocation-end minus600seconds, exclusive GPU
  lock, first native response check, no implicit scientific retry/resume.
- Conservative new-call ceiling6,144, constrained further by per-episode budgets;
  this is a safety bound, not a cost prediction or accepted work reservation.

CPU validation (already passed):

```bash
TRAINPY R/source-026-alfworld-closed-loop/launch_alfworld_closed_loop.py --root R --validate-only
```

Only after main acceptance, run the same launcher without `--validate-only`.
The old strict native analyzer intentionally expects string actions and must not
be silently reused for this changed schema; completed-result replay should use
the new parser/prompt/history contract. Collector SUMMARY retains all16 slots
per policy and native costs. Compare policies within this common new interface;
any comparison with original source022 is explicitly the **combined** interface
change. If invalid stops become long admissible loops without better native wins,
that does not qualify the agent for hierarchy training.
