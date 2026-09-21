# Helper training is not a demonstrated general answering improvement

On the fresh64-question MuSiQue panel, applying the helper-trained adapter
directly to the original question produces **49/128 correct answers**, versus
**54/128** for the base model. Both receive identical full public documents and
the same direct-answer prompt, seeds and generation limits. Neither uses a
planner, helper call or extra final answer. All256 calls return valid answers.

The paired difference is −3.91percentage points, connected-component95% interval
[−11.76,+4.03]; F1 falls from48.64% to47.25%, difference−1.39points
[−8.72,+6.18]. There are six wins and eleven losses, all both-valid. This is not
an established decline, but it does not support a generic answering benefit on
this panel. Training on subquestions and testing original composed questions
also changes the task distribution; this is a transfer control, not a direct
answerer trained on original-question targets.

The new base control exactly reproduces all128 historical base-direct emitted
token sequences and scientific requests. These are repeated same-seed
acquisitions, not extra independent evidence. There are64parents and47connected
atomic-component clusters. Actual new inference cost is746,688tokens/256calls;
base and trained arms use373,440 and373,248 respectively. No missing outcomes,
unresolved starts or unknown usage are reported.

The matched planner policies score53/128 for SFT and56/128 for RL, versus this
adapted direct model's49/128. Those differences remain uncertain; selecting
the weaker direct arm alone would overstate planning's value. Base direct54/128
remains an essential control and uses about28% of planner inference tokens.

## Secondary views do not replace the full panel

The completed predeclared stratum analysis retains all64 as primary:

| Questions | Base direct | Helper-trained direct | SFT planner | RL planner |
|---|---:|---:|---:|---:|
| Two-hop,32parents x2 | 26/64 | 27/64 | 31/64 | 29/64 |
| Three-hop,32parents x2 | 28/64 | 22/64 | 22/64 | 27/64 |

On three-hop examples the direct-adapter difference is−9.38points, exploratory
component interval[−20.4,−1.6]; F1 difference−4.6points still spans zero.
These are unadjusted subgroup views, not a tested interaction or evidence that
specialization necessarily harms composition. Exact TRAIN-paragraph exposure
and hop count are strongly associated; no exact-overlap subgroup is not a
pretraining-clean or semantically unseen dataset.

## Decision and evidence

Keep the unadapted full-source final in the current controlled planner studies.
Do not replace it with the helper adapter based solely on helper-role gains.
Run the analogous small direct-adapter control on Hotpot, where helper training
transfers more strongly. Finish the queued plan-only comparison before deciding
whether extra helper execution earns its cost. These results motivate role-aware
controls; they do not yet establish a new architecture or publishable advantage.

All external paths are under
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`.

- `analysis-direct-adapted-001.json/.md`: authoritative native audit and primary
  20,000-draw component bootstrap, seed2026092116. JSON SHA256
  `c2774c6a011376585937f994247d5efdb55d6c6c74002b1bfa77f01b9b4ab791`.
- `analysis-fresh-strata-001.json/.md`: completed secondary analysis,
  20,000draws, seed2026092121. JSON SHA256
  `246ccc66b4f656db623555b0352e3a9bfe3319d91ba7eac0c08f71c8337e6f2a`.
- `direct-adapted-001`: sealed source015, native episodes/calls and terminal
  owner06c23c7e9470. Fixed helper checkpoint36; no evaluation-time learning.
