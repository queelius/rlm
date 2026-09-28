# Compact-action exploratory comparison

Authorized CPU preparation; parent alone launches GPU. Existing isolated worktree;
no changes to active or pinned source files, no new environments or broad test gate.

Question: does generating only craft target/output quantity reduce generation cost
and improve a learned policy relative to emitting ingredient arguments then binding
them from public observations? This changes the interface and SFT token dose, not
merely the likelihood estimator; no RL or theorem novelty is claimed.

Design: adapt the existing collector/auditor through an experiment-local bridge
proxy. The proxy preserves strict JSON/schema checks and native get_info/finish;
craft lacks ingredient fields and requires a uniquely observed divisible recipe.
Only the native get_info reply can populate its per-frame recipe cache. Native
execution still checks inventory. Rejected attempts are charged and retained.

- [x] `compact_bridge.py` + `test_compact.py`: strict projection, public recipe
  binding, unchanged targets/counts, rejection of missing/ambiguous recipes,
  booleans/duplicate/extra keys/nondivisible output, no stock repair.
- [x] `prepare.py`: project all32task/366row public discovery examples; replay to
  rebuild compact instruction/history and remaining-token state; preserve original
  task bytes, action order/native feedback, and failures. Pin data/source/world,
  native qualification, tokenizer and original-vs-compact token doses.
- [x] `train.py`: thin unchanged SFT recipe with compact validator; same base,
  seed2026092208, one epoch,366rows,23updates,LoRA8/alpha16,LR1e-4; fixedcp23 and
 30min cap. Truthful separate compact contract; no old-manifest substitution.
- [x] `evaluate.py` + `fixture.py`: isolated collector/auditor modules sharing the
  compact proxy; first breadth fixed8/world42 and reconstructed fixed8/world50, two
  seeds,16episodes each. Fullformat+binder fixedpubliccp23 comparator; record exact
  model inputs, outputs, decode, costs, errors and replay. Saved-request CPU fixture
  traverses actual native episode and audit, including charged rejected actions.
- [x] README: runnable capped training/eval/audit commands, pins, expected3–6min
  training (original public SFT207s), up to45min each16episode readout, limitations
  and promote/retire criteria. Report changed prompt/output dose; equal epochs or
  steps are not equal FLOPs/tokens/wall time.

Success means ready, audited CPU assets and commands, not a GPU learning result.
Primary exploratory endpoint: paired task success plus returned completion/prompt
tokens and native call cost. Keep unavailable episodes unknown. Retire as a learning
direction if compact adaptation has no success gain and only trivial token savings;
retain as engineering if meaningful generation savings preserve success. One seed,
tiny fixed panels and same-world recipe overlap cannot support a robust claim.

Ruling: the explicit exploratory authorization and no-approval-gate instruction
supersede generic skill approval/commit gates. Use test-first local contracts and
actual saved-request verification; parent performs integration/launch review.

Ruling: use first breadth panel00/worlds42+50 instead of pilot42+43: it includes
depth5 and already has paired binder cp23 readouts; choose by index, not scores.
This avoids a new baseline run and does not claim matched difficulty across worlds.

Completed CPU checks: all32/366 projected rows pass native replay; five strict
interface tests pass; actual saved-request fixture passes both worlds including
charged schema/unknown-recipe/quantity/stock failures. Training and compact runtime
source pins remain unchanged after freeze. NoGPU launch/commit/push by this agent.
