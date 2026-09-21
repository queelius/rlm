# Conditional outcome-RL pilot: paired sufficiency, not decomposition

**Status update, 18:08 UTC:** The historical go/no-go proposal below was superseded
by [the conservatism amendment](SUFFICIENCY-RL-CONSERVATISM-AMENDMENT.md).
Both SFT readouts are complete: the observed tradeoff motivates a bounded
objective comparison, not a claim that joint SFT succeeded. Main accepted sealed
sources037/038 after code review and six focused CPU tests; supervisor82971 waits
for the QAMPARI decoding control. Implementation and acceptance are now complete.
The earlier stricter gate is retained below as decision history, not current policy.

Design only; no optimizer implementation, data acquisition, acceptance, or GPU job.
Complete the fixed step32 SFT readouts first, then the fresh replication. Source033's
panel is under exposure audit and may be replaced; no source number promises a blind panel.
Neither a training-loss decrease nor a gain on the repeatedly inspected32-parent
panel is enough to accept this branch. Source032 ALF local deliberation is also
pending; no partial outcome has been used here.

## Go/no-go and selected component

Proceed only if joint SFT shows useful supported-answer competence and its paired
score/negative-decision gains survive the positive-only control and fresh replication.
A lower overanswer rate purchased by substantially more positive abstention is
not the desired signal. If positive-only is as useful, prioritize the simpler
reading-supervision explanation rather than inventing sufficiency RL.

The selected component is the **whole single-call answer+sufficiency policy**:
released4B base weights frozen, only its warm-started joint-SFT32 LoRA trainable.
There is no root planner, helper, additional final, router, or separately trained
abstention head. The policy is frozen throughout each rollout batch. Start both
comparison arms from the exact same fixed endpoint, but with fresh optimizer/RNG
for this explicitly new objective; this is not continuation of SFT Adam state.
Freeze initialization and hyperparameters before looking at any RL endpoint.

## New TRAIN inventory and fixed acquisition cap

Propose128 previously unused official TRAIN parents, each with its official positive
and negative variants. Select by SHA256(`2026092193:` + native parent ID), excluding
the SFT256, every known experiment panel, all official DEV/TEST, and overlaps in
native parent ID, normalized question, or atomic components (union of both variants).
No selection by answer, support count, hop count, output length, reward, or model
uncertainty. Record split provenance, exclusion manifests and native public-prompt
token audit before any model call. If128 cannot be obtained, report the shortfall;
do not relax exclusions or fill from DEV. Preserve the existing exact public prompt,
blind document ordering, strict JSON, no truncation and8192-token context cap.

CPU proposal now exists at `R/sufficiency-rl-input-proposal-001/`:128 parents/256
variants, selected from12,105 eligible official TRAIN parents. Its explicit inventory
excludes all splits of September21 cases, both historical breadth case files, prepared
SFT parent IDs expanded through the official TRAIN questions/components, and inherited
full official DEV/TEST exclusions. This is the agreed known inventory, not a claim
about unbounded history. CPU tokenization: maximum prompt4,393, prompt+128 generation
4,521, maximum supervised target26; zero context/target overflows and no truncation.
Cases SHA256 `774ceda6bbac38b23fec6f823909e606889e804e7201fd37f338ffb81e9f7f81`.
The directory preserves the exact preparer snapshot matching the manifest hash;
the worktree copy subsequently received only a line-wrap lint fix. No optimizer
or GPU action is authorized by this input proposal.

Use eight fixed consecutive blocks of16 parents. At each parent sample four
independent **paired candidates**: one response to the positive public context and
one to the negative context. No response sees its counterpart, annotation, gold
label, or gold answer. Separate variant/candidate seeds are fixed in the manifest.
Maximum128 parents x4 candidates x2 calls =1,024 new training calls, eight possible
optimizer updates, one pass and45 minutes. This is a finite learning-direction
pilot, not a sufficient-dose efficacy claim. Do not silently extend it.

## Reward, credit and finite sampling

For candidate k at parent p, reward is official **paired exact match**:1 only when
both answerability decisions are correct and the supported variant's answer is
officially exact (aliases included); otherwise0. Returned malformed JSON earns0.
Transport errors, missing calls and unresolved starts are unknown and invalidate
the batch for training; do not replace them with zeros or resample them away.

The joint policy factorizes across the two separate public observations. Thus a
candidate's score-function term is the **sum of emitted-token log probabilities
from both responses**, including EOS only when actually emitted. For four pairs,
`A_k = R_k - mean(R_other_three)`. Loss is
`-sum(A_k * (logp_positive_k + logp_negative_k))/64` over all64 paired candidates
in a16-parent block. There are128 generated responses, but64 reward units; do not
accidentally divide twice, treat positive/negative as independent examples, or
subtract the other variant's reward as an action-independent baseline.

Reuse native root RL's T.8 behavior/replay matching, sum-token logp, one pass,
FP32 LoRA-only parameters, LR2e-5, dropout0, clipping1 and checkpoint receipts.
Readout remains T.5. Reuse numerical/cache-vs-full-forward discrepancy reporting;
do not pretend BF16 generation/replay is bitwise identical. Reuse client/owner,
loss and checkpoint mechanics, **not** `rollout`/role routing/plan parser or the
hardcoded root-only denominator blindly. No PPO epochs, importance ratio, KL,
new shaping reward, or length penalty is proposed in this first diagnostic.

Inspect the first fixed block's public prompts, reward calculation and native/replay
likelihoods before optimizing, but impose no arbitrary two-mixed-parent gate.
One mixed fully valid group can supply a clipped step; report its effective group
count and whether reward variation is semantic or only malformed-output recovery.
Distinct strings alone do not establish useful alternatives. Retain all uniform
groups in the denominator. An entirely zero-advantage block advances only the
**sample cursor**, not Adam or committed update count; existing Adam momentum must
not move parameters on a skipped block. Continue to the next fixed unseen block.
Stop for insufficient signal only after **four consecutive all-zero blocks**
(64 parents), or at the eight-block/time cap. Reset the zero streak after a nonzero
block. An incomplete/error batch stops independently. Preserve all sampled groups,
costs and stop reasons; no outcome-conditioned refill or nonzero-group renormalizing.
This is a finite pilot, not evidence of convergence or absence of learnable signal.

Report each block's0/4…4/4 reward histogram, all-valid mixed groups, protocol-only
variation, positive-answer versus label errors, exact/normalized answer diversity,
nonzero advantage mass, sequence lengths, gradient norm, parameter delta, and
before/after logp on the saved outputs. Save optimizer/RNG/adapter after every real
update plus immutable skipped-boundary receipts. State stores both sampled cursor
and optimizer step. There is no requirement to manufacture eight nonzero updates.

Individual-call rewards answer a different question: averaging positive correctness
and negative abstention gives an always-abstain policy half credit, while the paired
reward gives it zero. It can nevertheless provide denser credit. Do not switch to
it after observing flat pair rewards. If fully valid pair reward is almost always
zero, report that failure and separately propose a predeclared individual-objective
comparison; F1 is not automatically useful when the outputs contain no partial answer.

## Minimum more-SFT control and fixed readout

Compare unchanged joint-SFT32, additional paired SFT, and outcome RL. Additional
SFT uses the same initialization, parent blocks, strict targets, LR2e-5 and sampled
block budget. For each actual RL update, teacher-force the corresponding positive
and negative gold targets four times (128 responses), one accumulated update; retain
the same updated/skipped block inventory. This matches participating parents,
response counts, and real optimizer steps, not target lengths, information, loss
normalization, or total compute. RL additionally pays for1,024 possible generations;
report those costs. The control is conditioned on RL's nonzero-update schedule and is a
**dose diagnostic**, not a fully independent policy-training algorithm comparison.
A later positive result needs a separate prespecified cost-matched SFT comparison.

Commit the terminal endpoint only; select no checkpoint using development scores.
If the pilot stops early, label the dose actually achieved. Freeze a new outcome-
independent32-parent paired DEV panel before training and compare all three policies
on its128 requests with common seeds. Preserve both variants and both repeats in
parent-clustered paired intervals, report component overlap and all unknowns, and
retain positive answer EM/F1, positive abstention, negative overanswer, protocol and
native cost alongside official paired metrics. This is384 readout calls; cap each
arm at10 minutes. The fresh replication is no longer an untouched held-out panel if it helped
make this go/no-go choice; do not rename its reuse a fresh confirmation.

One A10040GB should accommodate the existing4B+rank8 recipe with microbatch1 and
checkpointing; measure peak memory rather than promise a fit. Propose45min RL,
30min SFT control and30min readout total maximum, subject to remaining allocation.
This is a conditional cap, not an accepted reservation. Joint gain over unchanged
SFT but not extra SFT indicates supervision/dose, not a reason to favor RL. Better
TRAIN reward without fresh improvement indicates fit without demonstrated transfer.

## Would interactive ALF TRAIN RL be better?

Not yet. Source026 established executable indexed actions (no invalid responses)
but only1/16 flat and6/16 manager wins, all gains on placement, with heating/cleaning
still unsolved. Source032's one-call local-reason control tests whether extra local
deliberation can explain that advantage. Its ordered reason+action schema and token
allocation are not a pure reasoning mechanism intervention.

If032 matches manager performance, prefer the simpler local-reason policy for any
later TRAIN competence study. If manager retains a useful advantage, a manager-only
adapter with frozen worker is a plausible selected component, but first require
mixed native-success rewards on independently selected official TRAIN games. Never
train on these eight exposed seen-development games. Select TRAIN games without
outcome filtering, keep task metadata/PDDL/expert plans host-only, reset exact native
initial states, and use identical affordance assistance and budgets across policies.

A four-candidate ALF RLOO group needs four full environment trajectories rather than
eight short QA responses. Even16 TRAIN games x4 candidates permits3,200 executed
actions, plus up to832 manager calls, with sparse terminal success and long-horizon
credit. The current all-valid action loops suggest semantic progress is a serious
bottleneck; terminal RL may simply produce uniform-zero groups. Outcome-only code
cannot safely reuse the one-call root loop without handling episode token/action
likelihoods, resets, unavailable environments and variable-length trajectories.

Therefore prefer the paired-sufficiency pilot **only if** SFT and fresh replication establish useful
competence plus mixed TRAIN reward; it is the smaller identifiable optimization
test. Prefer a bounded interactive TRAIN rollout-variation screen (not immediate
optimizer work) if032 shows a repeatable semantic advantage and sufficiency gains
are only abstention/annotation effects. If neither condition holds, neither RL job
is justified. Adaptive interaction, RLOO and abstention learning are established
ideas; this proposal makes no novelty or hierarchical-decomposition claim.

Inspected actual `rl_planner.py` RLOO/denominator/admission/replay/optimizer loop,
`train_sufficiency.py`, strict sufficiency metric, and source032's local-reason
contract. Existing root trainer conflates update and sample cursor, so skip support
above is a real future implementation seam, not something already implemented.
