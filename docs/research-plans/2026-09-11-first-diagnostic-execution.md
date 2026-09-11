# First RL diagnostic and strategy-choice pilot implementation plan

> **For agentic workers:** Use the applicable execution skill for code changes. Work task-by-task, keep GPU ownership exclusive, and do not let this document block the existing recovery run.

**Goal:** Determine what prevents useful RL updates, then test a small, learnable choice of decomposition strategy.

**Architecture:** Reuse the existing QS6 model, rollout collector and allocation-scoped runtime. First collect fresh attempts without updating weights. Prepare the strategy-choice comparison separately; a plan or a mathematical example is not a ready GPU experiment.

**Tech stack:** Existing Python rollout and LoRA stack; one A100 40 GB; append-only external attempt artifacts.

**Spec:** [Approved research direction](2026-09-11-learned-decomposition.md).

## Global constraints

- Use training/development source groups, not the advisor evaluation panel, for collecting training examples or selecting settings.
- Record immutable model, adapter, data, prompt and code identities before launch.
- Record every planned attempt, including unstarted, invalid and timed-out attempts. Never turn missing observations into valid answers.
- Keep final correctness separate from action validity and agreement with helper replies.
- Each launch has one owner, an absolute deadline and a fresh output directory. Do not retry the same failed inputs silently.
- The old runtime belongs to an27/allocation 5780. This session runs on an22/allocation 5801; preserve both histories.
- No production refactor, exhaustive test suite or presentation edit is a prerequisite for the exploratory launch.

## Task 1: Restore execution and finish the bounded interface diagnostic

**Owner:** `postmeeting_gpu_pilot`.

**Files:** External store `sidecars/runtime-an22-5801-v1/` and additive files under `sidecars/root-record-map-batch-handoff-v1/`.

- [x] Preserve the old allocation runtime and create a separate current-allocation binding.
- [x] Check the actual lazily loaded service path, not only the first imported runtime module.
- [x] Launch an additive recovery with the explicit synchronous helper API, the same frozen helper maps, 16 root endpoints and a 1,200-second outer cap.
- [ ] Record valid model calls, final outcomes and release status. A launched owner process is not proof that inference ran.
- [ ] Use this result only to qualify execution and interpret the existing interface question; do not call it RL training or learned decomposition.

Attempt 004 rejected the old allocation owner. Attempt 005 exposed a second late-loaded reference to that old runtime. Attempt 006 reached a live model server but its collector still used the old runtime, so it produced no model answers. Attempt 007 launched at 17:59:26 UTC on September 11 after checking both owner and collector runtime selection. These are distinct attempts, not an uninterrupted successful run. Consult the live queue for subsequent outcomes.

## Task 2: Collect fresh RL feedback diagnostics

**Owner:** `learned_recursion_prior_art`, reassigned to CPU preparation; coordinate GPU handoff with Task 1's owner.

**Artifacts:** External store `analyses/post-meeting-directions-2026-09-11/` for interpretation; the collector's new experiment directory contains the launch manifest, planned sample IDs and native attempts.

- [ ] Select 12 development tasks, covering several existing task families and source groups, with four sampled attempts each. Selection must be independent of new outcomes.
- [ ] Freeze the existing trained policy and tool interface. Set the collection seed and sampling settings explicitly; do not change them after inspecting early answers.
- [ ] Prefer a real-helper collection first. Add a paired exact-helper condition only if its intervention can be made without changing other task information. Label exact helper answers as an oracle.
- [ ] Launch a first collection capped at 1,800 seconds. Persist each completed attempt and the remaining planned IDs, so interruption does not require replaying completed calls.
- [ ] Summarize correctness, invalid actions, timeouts, all-success groups, all-failure groups and mixed-success groups. Report both all-planned and fully observed group denominators.
- [ ] Inspect a small sample of trajectories for genuinely different approaches. Different wording alone is not a different strategy.

**Decision rule:** If operational failures dominate, repair only the observed blocker before a longer collection. If valid attempts almost never succeed, make the task curriculum easier or strengthen the helper. If groups almost always succeed, increase difficulty. If both successes and failures occur, compare root-only RL with successful-trajectory self-SFT from the same starting checkpoint and matched collection budgets. The small diagnostic estimates readiness; it cannot establish a statistically reliable training improvement.

This is a diagnostic for root-only terminal-reward learning. Identical root rewards do not rule out useful independently verified child-task rewards; see the [published-code inspection](2026-09-11-rao-code-notes.md). Do not infer that all recursive RL is impossible from a low mixed-root-group fraction.

## Task 3: Make the strategy choice concrete before training it

**Question:** Can a policy choose what information a child must return, rather than always using the same short summary?

Start with two return choices: a count, or a set of entity–attribute relationships that the parent can merge. Include a direct whole-input baseline. Later expand to search and deeper delegation only if these first choices give a useful signal.

Example question: “How many customers bought both a bike and a helmet?”

| Evidence grouping | First child sees | Second child sees | Why the choice matters |
|---|---|---|---|
| Complete customer records | Maya bought both items. | Leo bought only a bike. | Counts of qualifying customers can be added: 1 + 0. |
| Split customer records | Maya bought a bike. | Maya bought a helmet. | Both local counts are 0, but the correct global answer is 1. |

Returning `Maya: {bike}` and `Maya: {helmet}` preserves the information needed to answer the second case. This is a mathematical illustration, not a model result. The cheap count strategy must only be presented as valid when customer groups really are disjoint and complete.

- [ ] Construct paired task instances with the same underlying facts and different valid partitions. Keep partition variants from one instance in the same split.
- [ ] Include event-counting tasks where scalar sums are sufficient, and cross-record relationship tasks where they are not. Include short tasks where delegation is unnecessary.
- [ ] First use exact child extraction to isolate the return-format decision. Then repeat with model-produced child answers; do not mix these conditions in a headline score.
- [ ] Compare each fixed return policy, direct processing, a simple rule based on observable grouping guarantees, and prompted choice before training a chooser.
- [ ] Keep the scientific task separate from hand-coded parsing: structured toy records establish correctness of the comparison, but do not establish a need for an LLM. Natural-language paraphrases and an external dataset are later generalization checks.
- [ ] Record final accuracy and actual tokens/calls/time separately. Structured returns may always be safer; a meaningful adaptive result must improve the accuracy–cost tradeoff against that strong fixed baseline.

**Promotion rule:** Train a chooser only after the fixed policies show a measurable tradeoff. If a cheap deterministic rule solves the whole selection problem, retain it as a baseline and make no claim of general learned planning. If always returning structured evidence wins without a meaningful cost, prefer that simpler harness change and redirect the learning experiment.

## Checkpoint and publication discipline

The live queue, not this fixed plan, owns changing execution status. Save launch receipts and failure diagnoses even when the experiment is inconclusive. After a completed diagnostic, record what changes next before increasing its compute budget. Do not revise the advisor slides just to report runtime repair; update them only when a scientific result changes the main story.
