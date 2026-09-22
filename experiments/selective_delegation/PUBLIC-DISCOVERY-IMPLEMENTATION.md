# Public discovery input prototype implementation plan

> **For agentic workers:** use superpowers:executing-plans and focused TDD for
> this single bounded CPU task. The user's autonomous research instructions
> override blocking approval and broad production-style qualification. Main
> reviews and seals completed inputs; no GPU run is accepted by this plan.

**Goal:** Determine whether public-query demonstrations can cover the same
32 frozen TRAIN tasks without privileged action selection.

**Architecture:** A deterministic teacher receives only a public goal, initial
and current inventories, and recipe replies it previously requested. A separate
bridge driver applies its JSON actions and records unchanged public prompts.
Task IDs and gold/depth metadata are never teacher inputs.

**Tech stack:** Existing Python, native trusted TextCraft bridge, native cached
Qwen tokenizer and `prepare_textcraft_sft.encode_row`; no dependencies or GPU.

**Spec:** `PUBLIC-DISCOVERY-QUESTION.md`, including its revised unmatched-token
dose limitation. This accepts CPU feasibility preparation only; training remains
conditional on source052 and a new decision receipt.

## Global constraints

- Work in the existing research worktree. Create only
  `prepare_textcraft_public.py` and `test_prepare_textcraft_public.py` in this directory.
- Reuse exact `R/textcraft-train-inputs-001/tasks.jsonl`, SHA256
  `390dff9bb19d0fe71c7bec0505c97608013c66c65aea90a821839f01b615ab30`.
- No task reselection, model sampling, checkpoint loading, or live-source edits.
- Same flat public prompt, native tools, 96 actions, 8,192 emitted tokens,
  256 tokens per action and 8,192 input-plus-output limit. No truncation.
- Strict JSON+EOS targets; observations and prompt tokens masked from loss.
- Preserve every task receipt and failed/capped trace. Never replace a task.
- Write CPU prototype artifacts to new `R/textcraft-public-discovery-prototype-001`.
  Mark them unaccepted for training; do not overwrite any existing path.

## Review focus

An action must use no unqueried recipe; shared prerequisites and native batch
sizes must not be treated as independent unit-yield trees; initial stock must
not be ignored; public net-goal readiness must trigger finish; unavailable
base resources or budget exhaustion must produce an explicit retained failure.

## Single task: observable teacher and replay qualification

- [ ] Read the spec, existing bridge and source047 encoder/replay behavior.
- [ ] Write and run failing focused fixtures for the teacher API
  `next_action(targets, initial_inventory, current_inventory, observed_recipes)`.
  Example: with goal `{'root': 1}`, stock `{'raw': 2}` and no recipes, its first
  action is exactly `{'action': 'get_info', 'items': ['root']}`. No gold object
  is supplied. A known root recipe requiring a missing intermediate must query
  that intermediate before any craft using its recipe. Add one shared-input,
  non-unit-yield fixture and one already-met net-goal finish fixture.
- [ ] Implement the smallest deterministic teacher over those arguments.
  Discover missing recipes through actual replies, aggregate demands across
  shared prerequisites before choosing batch quantities, and choose crafts
  only from observed recipes with currently sufficient ingredients. Derive any
  dependency ordering from queried edges, never hidden recipe tiers. Reject
  cycles or unreachable quantities explicitly rather than guessing a recipe.
- [ ] Implement the thin native driver using the existing public prompt and
  encoder. Call the native scorer only after the trace for qualification, not
  to choose actions. Log each action's public recipe provenance. No child calls.
- [ ] Run the focused fixtures to GREEN. Then replay all 32 fixed tasks on CPUs;
  count successes/failures, actions, target tokens, maximum context and coverage.
  Compare task identities, not model outcomes, with source047. Report dose
  differences honestly. If the algorithm fails, preserve the attempt; one
  small observed-seam correction is reasonable, not an open-ended planner build.
- [ ] Report source/input hashes, commands, tests and feasibility to main.
  Main performs review/sealing/commit. Do not edit trainers, create a GPU owner,
  or choose a trained endpoint. A successful teacher is not evidence that the
  student can learn its procedure or that this is a novel general method.

Self-review: one input-construction task; all five review risks have direct
teacher/native-replay checks. Existing encoded-row format is reused rather than
inventing a second trainer interface. Scope excludes model training, recovery
curricula, prompt changes, recursive teaching and benchmark selection.
