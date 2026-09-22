---
question_id: textcraft_public_discovery
status: conditional_proposal
created_utc: 2026-09-22
depends_on: textcraft-trained-readout-001
gpu_accepted: false
claim_scope: demonstration_procedure_not_new_general_imitation_method
---

# Public root-first discovery question

Prospective CPU-only design, contingent on source052 action-SFT23. It accepts no
run and changes no frozen047 rows, tasks, collectors, or queues.

## Question

Can action SFT teach a flat TextCraft policy to *discover* prerequisites for a
new public goal, rather than only teach strict JSON, action/finish protocol,
and associations from a goal to an intermediate recipe? The fixed047 teacher
has 366 public-prompt action targets on the same 32 official TRAIN tasks, but
30/32 first actions query an intermediate. That intermediate is absent from
the initial public frame: its selection comes from the host gold trajectory.
The label is legitimate supervision, but it does not exhibit root-first
information gathering.

The contrast is a deterministic **public root-first teacher** on the same 32
frozen tasks. Its first action is `get_info` for a public root target.
Thereafter it may select only from (a) the public target quantities and current
inventory, and (b) recipe objects returned by earlier `get_info` actions. It
expands an unqueried currently needed product before crafting it, propagates
quantity demand using that public recipe's result count, crafts only after the
needed recipe and its ingredients are public/available, and finishes only once
current minus initial inventory meets the public target quantities. A deterministic
item-ID ordering resolves simultaneous prerequisites. Thus the teacher is an
algorithm over the same information an actor sees, not a planner supplied with
the gold dependency order.

## Leakage and construction contract

The constructor must not read `gold_trajectory`, `max_depth`, hidden recipe
database entries, evaluator details, native win state, or a target answer when
choosing a teacher action. It can call the pinned host only through the
existing public `get_info`, `craft`, `view_inventory`, and frame feedback
interfaces; prompts contain only the existing public goal, inventory, action
history, and those replies. Host scoring may qualify whether the resulting
teacher trajectory completed, but selected TRAIN task IDs remain all 32 and a
failed/capped trajectory is retained rather than replaced. No demonstration
or prompt may include a future recipe reply before its own public query.

TRAIN and VAL share seed42's recipe world, and 21/47 frozen VAL-required recipes
already occur in TRAIN demonstrations. This is an interface/ordering intervention,
not generic planning, recursion, or unseen-fact generalization.

## Smallest comparison

Freeze a second 32-task row set before any model outcome, use the same fresh
4B base, LoRA r8/alpha16/dropout0, target-only JSON+EOS loss, seed, one epoch,
and fixed final checkpoint policy as source048. Keep the current public prompt,
flat policy, 96-call/8192-token episode budget, and source052's original and
trailing-reminder readout profiles. Report rows, target tokens, calls, caps,
and incomplete teacher traces before training. Do not silently truncate,
duplicate, or add demonstrations to force a matching dose.

The current reference uses 32 tasks, one epoch, 23 updates, 366 rows and 8,821
supervised target tokens. Equal tasks and epochs do not imply equal compute:
public discovery may change both trajectory length and update count. Freeze
and disclose those counts before training; keep complete trajectories rather
than silently truncating them to manufacture a token match. The first screen
would compare demonstration procedures as packages, not isolate ordering at
identical compute. A later matched-exposure control is needed before attributing
an advantage specifically to information availability. The direct comparison is current privileged-order action SFT
versus root-first public-discovery SFT. Existing base and fixed trailing
instruction control (source049) remain cheap protocol/termination controls:
they test whether an exact-response/finish reminder alone explains apparent
progress. They do not test prerequisite discovery.

Use the already exposed eight-task panel only as a mechanism screen, paired by
task and seed; it is not fresh semantic generalization. Primary descriptive
outcomes are valid one-object JSON rate, rejected-action rate, root-first
`get_info` rate, successful craft progress, finish validity, native success,
and calls/tokens. A later genuinely fresh panel would require selection and
sealing before either new adapter is read out.

## Decision rule

Consider this if source052 leaves material discovery/order failures after
training and valid execution have been verified, whether or not aggregate
success improves. A null SFT result alone does not rule out a deficient teacher.
If remaining failures are only termination, prioritize the cheaper public-goal
termination baseline. Defer for unresolved protocol/runtime failures, or if
the public teacher cannot complete the fixed tasks within the budget without
hidden information. Retire the new procedure if it merely trades format or
termination for no improvement in discovery or execution. Even a positive screen would show
that a public discovery trace helps this fixed TextCraft interface; it would
not establish a new general hierarchy or memory method.

Sources: `TEXTCRAFT-TRAINING-COVERAGE.md`; frozen047 preparation
`R/source-047-textcraft-action-inputs/prepare_textcraft_sft.py`; public bridge
`textcraft_bridge.py`; fixed task/row identities recorded in the coverage note.

The [primary-literature note](LITERATURE-TEACHER-INFORMATION-20260922.md)
identifies LEAP's constrained privileged experts as directly relevant prior art;
the general student-information constraint is not a novelty claim.
