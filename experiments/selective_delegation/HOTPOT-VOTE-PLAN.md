# Hotpot fresh direct three-vote control

Status: proposed CPU-ready preparation; it is not accepted, sealed, or launched.

## Question

The fresh 128-parent Hotpot planner used about 3.44 times as many tokens as
the one-sample base-direct control without an observed accuracy advantage. This
control asks whether two additional independent base-direct samples plus a
predeclared majority vote are a cheaper competing use of test-time compute.
It is not an outcome-tuned vote rule and does not test planner learning.

## Frozen comparison

The input panel is exactly `hotpot-fresh-inputs-001/cases.jsonl`, with the
256 existing slots in `hotpot-fresh-direct-001` (128 parents x two repeats).
Each saved direct final is voter index 0. For every exact saved slot, collect
only voters 1 and 2 with the actual saved request seed plus 10,000 and 20,000.
The collector verifies the saved request was base/final, adapter-disabled,
temperature 0.5, top-p 1, top-k 0, cap 128, Qwen3-4B base, and the exact
current full-source direct prompt. Extra calls use the same prompt, base
weights, temperature, and cap. They do not load an adapter into generation.

Maximum new calls: 512. Expected collection cap: 20 minutes. This is a cap,
not an assertion of equal actual compute: the original responses are reused,
and physical tokens/calls are reported from receipts. The comparison retains
the planner and original direct results separately; it does not claim matched
actual compute.

## Vote and scoring

Strict `{"answer": string}` parsing and the cached official HotpotQA answer
normalizer define a vote. A malformed return is a single explicit `invalid`
ballot: two invalid ballots beat one valid answer, and a three-way tie resolves
to index 0 even if index 0 is invalid (then the slot is protocol-zero). Two
matching valid answers beat one invalid ballot. Any unavailable voter makes the
planned three-sample policy unobserved; it is reported as unavailable and scores
zero rather than silently falling back to a one- or two-sample vote. No fallback,
oracle, gold access, or outcome-dependent rule choice is allowed. A chosen valid
raw answer is regraded with official Hotpot EM/F1.

Every one of the 256 original parent/repeat slots remains in the denominator.
Missing extra calls remain unavailable—not votes and not known-zero cost. A
slot with any unavailable voter scores zero and is reported separately. Compare only
the predeclared three-vote direct result with original direct and existing
planner; do not choose among further voter counts.

## Seal and launch contract

Before acceptance, copy `hotpot_direct_vote.py`, `score_hotpot.py`,
`eval_planner.py`, `probe.py`, `prepare_plan_training.py`, this plan, and the
focused test into external `source-025-hotpot-vote`; record their hashes,
the source panel hash, original plan/call-tree hashes, and the model manifest
in `HOTPOT-DIRECT-VOTE-DECISION-001.json`. Its status must be `accepted` before
`launch_hotpot_vote.py` can run. The launcher waits for `sufficiency-001`,
collects once, then writes a separate immutable official-vote analysis.

Focused test coverage: normalized majority and original-index tie; invalid and
unavailable voters; no-valid-vote slots; actual-seed offsets; and rejection of
changed original sampling. It deliberately does not run a model or GPU.
