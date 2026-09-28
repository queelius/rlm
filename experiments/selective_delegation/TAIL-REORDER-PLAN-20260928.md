# Front-load distinct questions without restarting scientific work

Approved scope: the user's autonomous, adaptive, broad experimentation request.
Design reviewed in `SCHEDULING-REVIEW-20260928.md`. No new scientific treatment
is introduced by this scheduling change; the separately qualified ALFWorld pilot
is added. Existing model owners and accepted sources remain untouched.

1. Test that reordering preserves all original job records exactly, retains
   within-block order, and rejects duplicate scientific outputs or absent screens.
2. Build one new tail after the unchanged interface/dose queue: three flexible
   screens, quantity/Phi, ALFWorld, fresh RL, reward diagnostics, compact RL.
   Preserve every original command, cap, output and source/input pin. Add only
   orchestration and old-receipt pins.
3. Dry-check actual receipts, ALFWorld preparation and all small pins. Before any
   process change, authenticate the four idle supervisors and confirm their queues
   contain only invocation records and no scientific owners/children.
4. Record intent, suspend downstream-to-upstream, recheck idle state, then stop
   only these four waiting supervisors. Resume suspended processes on exception.
   Preserve old receipts and record supersession. Launch the replacement through
   the existing queue executor. A live scientific stage cancels the operation.
5. Verify the new waiter, old waiter exits, unchanged active GPU owner and exact
   job conservation. Update resume pointers and push the scoped checkpoint.

Verification is limited to this seam; there is no broad runtime test campaign.
Individual jobs keep their own checkpoints, error accounting and allocation caps.

Completed at 12:09 UTC. Three ordering/conservation tests and scoped Ruff passed;
an independent read-only review verified all 84 inherited records and the six
new ALFWorld descriptors. All four old waiters exited; no scientific owner was
stopped. New supervisor 1105638 is waiting on the unchanged interface/dose queue.
Receipt SHA256:
`fac47014954360cc025e977b77d1c6d6f3fd6504ffd9f98eb410558d3fb22344`.
Parent separately reran all eight ALFWorld tests; native fixture evidence and
its limitations are in that package's verification document. External intent
and SUPERSEDED receipts record exact job conservation and process identities.
