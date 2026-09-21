---
status: proposed_adaptive_amendment_not_gpu_accepted
created_utc: 2026-09-21T17:42:00Z
trigger: completed_joint_sft_readout
question: Can outcome training correct excessive refusal without restoring unsupported answers?
novelty_status: competency_and_objective_diagnostic_not_new_RL_algorithm
---

# A more specific reason to test outcome RL

The joint SFT endpoint is complete; its answer-only control is still training.
On the previously examined32-parent panel, the joint model answers11 of64
supported cases exactly, versus16 for base. It refuses37 supported cases,
versus24 for base; official-negative overanswers fall from23 to6. Paired exact
success only moves from9/64 to10/64. All128 responses are valid and available.
These are complete single-arm results, not the final two-arm comparison.

The original conditional proposal required a useful SFT improvement before RL.
That gate is too narrow for the newly observed question. A nonzero but overly
conservative policy may supply a useful starting point for asking whether the
**training objective**, rather than additional imitation, can improve the
answer-versus-refuse trade-off. An always-refuse policy gets zero paired exact
reward. We must still establish useful sampled reward variation on TRAIN; a
different objective cannot create missing correct candidate answers by assertion.

## Proposed revised decision

Finish both SFT readouts. Use the already prepared fresh paired panel to check
whether the observed trade-off persists, even if the result is a null rather
than a positive replication. This is a small diagnostic, not selecting a better
checkpoint. Freeze a separate later evaluation panel before any RL update.

If the starting policy produces valid reward differences on the new TRAIN
inventory, a bounded paired-outcome pilot is justified even without a demonstrated
SFT-over-base gain. Compare it with unchanged joint SFT and additional paired
SFT from the same initialization and participating training cases. The existing
proposal specifies one pass over at most128 new TRAIN parents, four paired
candidates each, eight16-parent blocks, and the matched-dose limitations.

Use the exact fixed step32 joint checkpoint, not a selected intermediate one.
Keep official paired exact reward and the original likelihood objective. Do not
add a refusal bonus, change to individual-label rewards after looking at flat
rewards, or silently extend training. Skip Adam on all-zero-advantage blocks;
continue through them until the finite eight-block cap, four consecutive zero
blocks, or the declared time bound. All sampled work remains in the cost account.

## What would change our mind

- More paired success with retained or recovered supported-answer accuracy on
  the later fresh panel would justify replication and an information-request task.
- A gain matched by extra SFT would point to additional supervision or dose,
  not a special benefit of RL.
- Higher TRAIN reward without fresh gains would remain fit, not transfer.
- Mostly uniform-zero rewards would identify an exploration/competence bottleneck;
  it would not justify more of the unchanged optimizer.

The official negative labels sometimes retain alternative evidence. This is
optimization against a dataset contract, not a perfect semantic verifier. No
claim about recursion, learned decomposition, general grounding, or publication
novelty follows from this proposed pilot alone. The full two-arm results and
the queued interactive-task control can still redirect the priority.
