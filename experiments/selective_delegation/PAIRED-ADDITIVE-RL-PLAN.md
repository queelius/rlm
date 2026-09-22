# Additive versus product sufficiency reward: bounded objective control

**Status update, September 22, 06:30 UTC:** main accepted this fixed comparison
in `R/INDEPENDENT-TRAINING-QUEUE-001.json`; it follows the crafting action-SFT
run. The deeper readout subsequently reported warm/RL/SFT correct pairs 6/7/6
of 64. Frozen panel050 was prepared before new optimizer updates and before
the final deeper-panel review, not after that review as the prospective text
below proposed. No deeper-panel outcome was used to select its questions.
The sealed source051 proposal remains unchanged; this is the working decision
history, not a retroactive alteration of the original plan.

CPU proposal, 2026-09-22; no GPU acceptance or new panel selection. Existing
039 readout gives warm joint-SFT32 / product-RL8 / matched extra-SFT8 paired
EM 10/18/19 of64. RL has no established advantage over extra SFT. Await the
separately frozen046 compositional replication; do not select new TRAIN
parents or tune an endpoint using its outcomes.

## Question and smallest comparison

Does removing the product reward's marginal-success gate improve held paired
correctness, rather than merely make refusal easier to learn?

Start again from the same joint32 adapter with fresh Adam, not from RL8.
Keep the same128 TRAIN parents/order, eight16-parent blocks, four independently
generated positive/negative candidate pairs per parent, prompts, seeds, T=.8,
rank8, LR2e-5, clip1, zero weight decay, actual-emitted-token sum log-probability,
and division by64. Change only the scalar reward:

- Product (completed037): `P * N`.
- Additive candidate: `(P + N) / 2`, diagonal pair RLOO, with the same pair
  advantage applied to both response log-probabilities.

`P` means positive label true **and** official alias-aware exact answer;
`N` means negative label false. A wrong positive answer remains zero even if
its answerability label is right. A malformed returned variant has marginal
success zero; the other variant retains its own success. Missing/native-failed
responses are unknown and halt, never score zero. Keep every returned candidate
and the full64-pair loss denominator. No repairs, rejection resampling, reward
normalization, KL change, separate marginal advantages, or mixed
`pairing_mean`+additive option in this comparison.

Cap at eight sampled blocks /1,024 calls /45minutes. Skip Adam on all-zero-credit
blocks but advance the sampled cursor; stop after four consecutive such blocks
or the cap. Save each committed update/skip and use the declared terminal
endpoint, not the best observed checkpoint. Distinct future policies require
fresh on-policy collection; later saved037 trajectories are not additive-policy
training data. The common first block is a same-warmstart counterfactual audit.

## Actual saved037 credit coverage

All eight committed blocks,128 parents,512 paired candidates and1,024 native
responses were retained. Exactly two pairs include a malformed numeric answer;
zero calls are unavailable. Official marginal reconstruction reproduces all512
saved product rewards. No gold or outcome-based sample selection was performed.

| Block | Product active groups | Pairing-mean active groups | Additive active groups |
|---|---:|---:|---:|
| 1 | 2 | 4 | 10 |
| 2 | 4 | 6 | 13 |
| 3 | 6 | 7 | 12 |
| 4 | 5 | 8 | 14 |
| 5 | 8 | 8 | 12 |
| 6 | 8 | 10 | 13 |
| 7 | 4 | 4 | 12 |
| 8 | 5 | 5 | 12 |
| Total /128 | 42 | 52 | 98 |

| Descriptive credit quantity | Product | Pairing mean | Additive |
|---|---:|---:|---:|
| Active groups among126 fully valid groups | 41 | 51 | 96 |
| Responses with nonzero coefficient /1,024 | 336 | 340 | 756 |
| Actual emitted tokens in those responses | 4,165 | 4,374 | 9,316 |
| Sum absolute response coefficients, before common `/64` | 192 | 118 | 236.6667 |

The additive scalar histogram is123 zero,311 half,78 one. Product has434 zero
and78 one. Additive restores56 product-inactive groups, but **46 of those56
contain no correct positive answer at all**: their new signal is negative
refusal variation. Two contain no correct negative refusal; eight have successes
on both marginals. More active groups therefore do not establish useful answer
learning. Additive retains all42 product-active groups, but loses two of the52
pairing-mean-active groups because their `P+N` is constant.

`pairing_mean` is a different control: conditional averaging over24 pairings
preserves the expected product objective and restores10 product-inactive
groups. Its coefficient mass118 here equals the earlier report's29.5 multiplied
by4, because that audit included a per-parent `/4`; the training denominator
already accounts for it. Neither coefficient mass nor active-group count is a
gradient-variance estimate or a prediction of post-clipping/Adam improvement.
Do not change the reward and pairing estimator together and attribute the result
to only one of them.

## Readout and decision

Freeze a small new held panel before any new updates, after reviewing046; use
outcome-independent IDs and explicit overlap provenance. Read the same panel
with joint32, completed product-RL8, completed matched extra-SFT8 and the new
additive endpoint. Existing039/046 are exposed exploratory evidence, not fresh
selection sets. Primary metric remains official paired EM, with paired F1,
positive exact accuracy/abstention, negative overanswer, both-label accuracy,
protocol/unavailable counts, and parent/component-cluster uncertainty reported
together. Match endpoints and record all native costs; equal samples/updates
are not equal nonzero-credit tokens, gradients, information, or FLOPs.

Promote only if increased coverage accompanies a useful held paired/content
improvement without simply exchanging positive competence for more abstention.
Do not require an arbitrary p<.05 threshold in this small exploratory pilot.
If only easy-negative accuracy or TRAIN credit grows, retire this objective
explanation rather than extending training indefinitely. This is a standard
objective/competency diagnostic, not novel abstention, decomposition, or RL.

## Reproducible evidence

`R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
The CPU calculation uses each committed `sufficiency-rl-001/batches/sample-*/`
`pairs/pNN-kK.json`, its two `calls/*.json` native output token lists, and
`analysis-paired-reward-pairing-002/REPORT.json`'s independently reconstructed
positive/negative successes. For each four-candidate group compute
`rloo(x)[i] = x[i] - sum(x[j] for j != i)/3`; use `x=P*N` or `x=(P+N)/2`.
Pairing-mean response coefficients are `mean(N)*rloo(P)` and
`mean(P)*rloo(N)`. Count exact nonzeros; do not remove malformed rows.

- Product PLAN SHA256: `5950ee3c4bc84c7c6bf8c3fe2ee56001bdfd824c5b4fd556952b52ed93f5e020`.
- TRAIN cases SHA256: `774ceda6bbac38b23fec6f823909e606889e804e7201fd37f338ffb81e9f7f81`.
- Sealed037 trainer SHA256: `3a5586425b1ee99062c9ec22dc4df6d81473a57bf69b4e4317fa5debc5c20288`.
- Pairing audit002 REPORT SHA256: `0a39243283d0e42d8610d994b75ad07084f5d17f665069d22c2d265be359ed43`.
- Existing full TRAIN native/endpoint audit: `analysis-sufficiency-rl-training-001.json`.

Only small retained receipts were read; no weights were rehashed, no GPU calls
were made, and no completed output or sealed source was changed.
