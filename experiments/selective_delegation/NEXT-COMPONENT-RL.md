# Conditional next step: planner RL with a fixed helper contract

CPU readiness implementation is available; no new GPU launch is authorized by this
document. Existing sealed sources remain unchanged. The runner supports immutable
base, exact-format-reminder, or separate frozen helper-SFT36 contracts, with matched
T=.5 evaluation and base finals. Consecutive schedules now support up to16 updates;
the existing repeated16-parent, four-update default is unchanged. Choose the fixed
helper contract from the three-arm readout before accepting and sealing a run.
The question is whether fresh planner RL adds value **after**
fixing the helper, not whether changing two components beats the old system.

## Choose the cheapest supported helper contract first

Compare three helper conditions under identical frozen SFT48 root plans and base
finals: the original base helper, base weights with a trailing JSON-format reminder
appended after the documents, and helper-SFT36 with its unchanged original prompt.
Keep strict parsing, no repair/fallback, matched seeds and output caps. The reminder
is a separate prompt intervention; do not add it silently to the trained-helper
arm. Record its exact text/source hash and extra input tokens.

If `format_reminder` matches the trained helper's useful behavior on the paired
panel, prefer **base weights plus that fixed reminder** for the next planner-RL
test. Here “matches” is a practical small-panel judgment, not statistical
equivalence: compare protocol recovery, end-to-end EM/F1, the identities of gained
and lost parents, and costs. Require a concrete reason to retain SFT, such as
repeatable correct-answer gains where both arms return valid answers, rather than
an unexplained small aggregate difference. An interval crossing zero alone does
not establish equivalence, and a forced p<.05 gate is inappropriate for this pilot.
If the comparison remains ambiguous, report that uncertainty and retain the
cheaper default or predeclare one bounded replication; do not search prompts.

The reminder branch needs only the existing single 4B instance: root adapter
enabled for root generation/replay, disabled for reminder-helper and full-source
final calls. Start root from SFT48 with a fresh optimizer, not old RL4. No helper
adapter or second model is needed. Freeze the identical
reminder contract for both SFT-root baseline and subsequent RL-root evaluation;
never compare RL+reminder against SFT without the reminder and call it an RL gain.
Do not claim helper SFT was necessary if the cheaper prompt control explains its
benefit. Publish a price companion comparison: actual input/output tokens, calls,
latency and resident memory, including the reminder's added prompt tokens.

Specific pending hypothesis: terminal answer reward may favor plans that elicit
compliant helper formatting rather than better decomposition. In the prior held
comparison all three RL wins involved protocol recovery, while both losses had
valid finals. This motivates the hypothesis but does not prove a contract shortcut
or exclude semantic learning. Report wins/losses involving protocol failures
separately from both-valid outcomes; inspect whether intermediate answers actually
improve before labeling an effect semantic. Even both-valid final gains can arise
from downstream sampling, so parent-level examples and replication remain useful.

## Trained-helper branch: use two resident 4B instances

If helper SFT earns its additional complexity, recommend two independent copies
of the same pinned BF16 4B base on the one GPU:

| Role | Model instance | Adapter | Gradients |
|---|---|---|---|
| Root generation/replay | A | Root planner adapter | Only root LoRA during replay |
| Helper generation | B | Fixed helper-SFT adapter | None, including helper LoRA |
| Full-source final | A | All adapters disabled | None |

Initialize A from planner SFT48, **not** old RL4, with a fresh Adam optimizer.
Initialize B from the fixed, committed helper-SFT checkpoint36. If that checkpoint
does not exist or the accepted helper run is incomplete, do not silently substitute
another dose. The selected helper checkpoint is fixed before this RL run and stays
the same during training and evaluation.

This costs one extra 4B base copy but avoids switching named adapters inside the
trainable root model. The installed PEFT `set_adapter` implementation changes
`requires_grad` by default, including deactivating inactive adapters. A one-base,
two-adapter design is possible with careful inference-mode switching and explicit
gradient restoration, but adds a failure mode to the policy replay path. It is not
the smallest safe modification for this short experiment.

The implemented `rl_planner.Client` selects model and
adapter identity by role: root→A enabled; helper→B enabled; final→A disabled. Keep calls
sequential, local HF, and retain actual token IDs/usage. Root behavior logps remain
root-only; never request helper gradients or include helper tokens in the loss.

## Trained-helper branch: memory and freeze checks

The cached 4B safetensors total 8,045,591,552 bytes. Two BF16 copies therefore cost
16.09 GB decimal, approximately **14.99 GiB**. Root FP32 LoRA parameters, gradients
and two Adam moments add about 0.246 GiB for 16,515,072 parameters; frozen helper
LoRA adds about 0.062 GiB. This excludes allocator/workspace and activations.

The model has 36 layers, eight KV heads, head dimension128. A BF16 batch-one KV
cache costs `2 * 36 * 8 * 128 * 2 * sequence_length` bytes: about0.844 GiB at6144
tokens,1.125 GiB at8192. Calls are sequential and caches must not be retained in
receipts, so this is a per-active-call cost, not two simultaneous full caches.
Root gradient replay is much shorter than helper/full-source final generation:
title-index prompt plus at most128 root tokens, with activation checkpointing and
cache disabled.

A rough **18–26 GiB allocated** working estimate for inference or root-only
training is plausible; allow additional reserved-memory headroom. This is not a
measured fit guarantee. Verify actual load, a real full-source helper/final, and
one root backward before extending the run. Record allocated/reserved peaks;
preserve an OOM attempt rather than silently quantizing or dropping documents.
No new environment, download, server, or concurrent GPU owner is needed.

Required narrow invariants:

- A has exactly the expected504 trainable FP32 root-LoRA tensors; all its base
  parameters are frozen. Build the optimizer solely from that explicit A list.
- B loads with `is_trainable=False`; explicitly freeze every parameter, use eval
  mode and no-grad, and assert zero trainable parameters. Its parameter identities
  must be disjoint from the optimizer's list. Never call `set_adapter` on B after
  freezing it; its only adapter stays selected.
- A's final calls use its existing `disable_adapter()` context, then restore root
  state. Before replay assert root trainability/active adapter and check that B has
  no gradients. A tiny CPU fixture should show a root update changes root output
  but neither disabled-base final output nor trained-helper output.
- Root replay runs only A. Preserve the emitted-token/EOS alignment, T=.8 behavior
  and replay, and before/after full-forward likelihood diagnostics. Keep the known
  BF16 cached/full-forward discrepancy visible rather than claiming exact equality.

## Keep fresh RLOO and immutable identities

Keep four fresh root candidates per parent,16 parent groups per update, denominator
64 including zero advantages, one accumulation pass and one AdamW step. Use the
existing LR2e-5, weight decay0, clip1, and a declared maximum of up to16 consecutive
updates within three cumulative hours; no cached trajectories
across updates, PPO reuse, new reward shaping or helper/final optimization.
Root cap128, total helper output384 divided by plan length, final128; helpers and
final T=.5, common downstream seeds across candidates. Full-source final remains
unchanged, without a provisional answer or reference annotations. Missing
generations halt; returned malformed outputs receive zero with explicit status.

Admission remains a predeclared check for at least two groups with distinct valid
plans and mixed rewards among valid plans. Report fully scored versus
protocol-driven variation. A trained helper may remove syntax failures, but could
also make groups all-correct and eliminate gradient signal; neither outcome
justifies inventing reward variation or excluding zero-advantage trajectories.

For the trained-helper branch, PLAN must record separate root and helper paths,
COMMIT/config/STATE/adapter hashes,
base manifest for both copies, optimizer parameter scope, execution-source hashes,
case schedule, seeds and compute cap. Every call records instance/role and actual
adapter hash: evolving root hash, constant helper hash, or null final adapter.
The old `helpers_adapter_disabled=True` receipt is no longer accurate: replace it
with `helper_weights_frozen=True` plus explicit helper identity.

Save only A's evolving adapter, optimizer and RNG state in RL checkpoints, while
binding each checkpoint STATE to the immutable helper identity and PLAN. Resume
must reject changed helper/base/input/source identities and restore the committed
root optimizer. B is reloaded from its exact fixed checkpoint, not from a newly
selected helper run. The next fresh batch must use the just-committed A weights.
Preserve cumulative deadline/lease-minus600, exclusive ownership and no implicit
retry of an unfinished rollout batch.

For the reminder branch, bind PLAN/checkpoints to the frozen reminder source/text
hash instead of a helper checkpoint, retain one model and adapter-disabled helper
receipts, and preserve all the same fresh-rollout and optimizer constraints.

## Broaden coverage without increasing batch compute

It is sensible to broaden beyond the original16 parents: four were always wrong,
five always right, and only72/256 trajectories received nonzero advantages. F1
provided no additional mixed groups on that original panel. Do not select new
parents by their observed helper gains or rewards.

Conditional full-pass schedule: retain the existing deterministic shuffle of all256
train parents, then use consecutive disjoint16-parent blocks for updates1–16.
This covers each training parent once, for at most1024 fresh root trajectories,
with no within-pass parent duplicates. Four updates on the original16 parents
established gradient feasibility, not an adequate test of RL efficacy; this bounded
pass tests a larger training dose and broader coverage without adding a new reward,
KL term, sampling policy, or warmstart. Update1 preserves the old admission panel.
Declare all `case_ids_by_update` in PLAN before collection; retain the64 denominator,
existing admission/stop rule, and one committed checkpoint per completed update.
Do not skip failed admission blocks or extend the pass after inspecting rewards.
The hard bounds are16 updates and three cumulative hours (also allocation-minus600s);
an interrupted/capped run is not a completed pass. Repeated-panel schedules remain
limited to four updates. The default remains four repeated-panel updates.

The opt-in schedule is `--parent-schedule consecutive --updates 16 --hours 3`,
with the selected immutable helper contract and original SFT48 warmstart. This is
CPU-ready support, not an accepted GPU launch or a claim that16 updates suffice.

This changes coverage as well as the helper relative to the old four-update run;
it is not a pure old-versus-new training mechanism comparison. Per-update rewards
on changing parents are **not a learning curve**. The fixed held-out comparison
below is the decision metric. If a pure helper-mediated admission comparison is
the priority instead, retain the old16 for all four updates and label that choice
as the narrower diagnostic; do not change schedules after seeing outcomes.

## Separate helper gain from subsequent RL gain

Predeclare the same evaluation parents, repeats, sampling and source contract:

| Condition | Root | Helper | Final |
|---|---|---|---|
| A | SFT48 | Released base | Released base |
| B | SFT48 | Selected fixed helper contract | Released base |
| C | New planner-RL, predeclared completed checkpoint | Same fixed helper contract | Released base |

The selected contract is either base+format-reminder or helper-SFT36, fixed before
RL. **B−A is helper-contract gain; C−B is incremental planner-RL gain. C−A combines
both.** Keep the nonselected reminder/SFT condition's matched readout as the
companion control explaining this choice, not as a secretly tuned comparator.
Reuse identical frozen root plans for A/B when measuring helper effects. C must
generate its own root plans under the new policy; pair its root seeds and cases
with B, not its outputs. Use the evaluation temperature already fixed for both
root conditions, rather than silently using the RL rollout temperature for only C.
Keep strict parsers, all planned denominators, protocol/content decomposition,
parent-cluster intervals, and native call/token/latency costs.

Keep the earlier helper-effect panel separate. Its `eval_helper` downstream seed
base differs from `eval_planner`, so do not reuse it as the new C-versus-B baseline.
Collect matched SFT and new-RL evaluations with the same fixed helper, sealed source,
case inventory, and evaluator root/downstream seed formula on authoritative
`fresh-dev-inputs-003` (`split=development`,64 parents,32/32 panels). Record the
predeclared checkpoint/dose and stopping rule before inspecting those outcomes;
do not choose the best checkpoint on the final comparison panel. Historical
exclusions do not make an adaptively chosen experiment confirmatory. Freeze
component/checkpoint choices before touching remaining transfer outputs. Do not
declare planner improvement from training reward alone or count two repeats as
twice as many independent parents.
