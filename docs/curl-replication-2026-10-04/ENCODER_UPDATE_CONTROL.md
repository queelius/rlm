---
date: 2026-10-05
status: five_complete_final_running
question: Does the effect of correct or wrong image matching depend on the strength of its encoder update?
task: cartpole/swingup
training_seeds: [123, 456, 789]
primary_endpoint: 100000_training_simulator_steps
new_arms: [single_encoder_curl, single_encoder_shuffled_curl]
---

# Does the strength of the extra learning exercise matter?

**Execution update, October 5 at 18:08 UTC:** five of six models are complete.
Correct matching with the smaller update helped seed 123 (795.84 versus 678.02)
but hurt seed 456 (316.12 versus 446.17) and seed 789 (531.20 versus 587.99).
All three remain above no matching; there is no consistent advantage over
the original update rule.
The smaller-update wrong-target scores are 138.08 and 72.01, compared with
0.11 and 68.10 originally. Both remain below no matching (454.47 and 240.54).
Wrong matching for seed 789 is running. Finish it unchanged. A separate
[fresh-seed comparison](CARTPOLE_FRESH_SEEDS.md) is admitted and waiting behind
that final job under the shared lock. No fresh training has begun at this cutoff.
Actual returns, finite learning values, native completion and saved checkpoint
sizes were checked. Do not infer equivalence or a reliable interaction yet.
The [dated findings](FINDINGS.md) and [native records](encoder-strength-data)
give the evidence.
Implementation and focused real-upstream CPU tests passed independent review
before admission. The first model launched 0.875 seconds after the predecessor
queue ended; subsequent jobs launched immediately under the same owner.

The design below was fixed at 16:28 UTC, when only the first wrong-matching
endpoint was known. All three original wrong-target models have since finished;
their [complete results](CORRESPONDENCE_CONTROL.md) do not change this fixed design.

At the time of that decision, the first model taught to match the wrong pictures had finished with almost no
test reward. The original model with correct matches, and the model without
the matching exercise, had both learned much more. Only one new training seed
was complete, and the other two were required to finish unchanged.

That result raises a narrower question: **are incorrect targets particularly
harmful because the extra exercise changes the image encoder too strongly?**
The encoder is the part of the model that turns pictures into useful features.
This question concerns the strength of learning, not just whether pictures
are matched correctly.

## What we will change

The pinned official CURL code has two separate optimizers that both update
the same image-encoder parameters from one matching-loss gradient. They have
the same learning rate and Adam settings. The second optimizer also updates
the learned comparison between image features. Our reference runs preserve
this behavior. We are not calling it a proven bug.

For the new conditions, skip only the dedicated encoder optimizer's step.
The other optimizer still updates the encoder and the image-comparison
parameters. Keep its full parameter membership, the loss, labels, image crops,
reward-learning updates and target-network schedule unchanged. Both original
gradient-clearing calls remain. This makes a smaller encoder update for that
exercise; it does not remove image matching or change reward learning's own
optimizer. Subsequent trajectories can diverge, so this is not a claim that
the total training effect is exactly halved.

| Matching exercise | Original update rule | Smaller encoder update |
|---|---|---|
| Correct picture pairs | Three existing models | Three new models |
| Wrong picture pairs | Three completed models | Three new models |

Keep the original three models without matching in the report as context.
They are not part of the two-by-two comparison above.

## Fixed experiment and interpretation

Run all six new models from fresh weights with seeds 123, 456 and 789. The
decision to prepare all six was made after the first wrong-matching endpoint,
before the other two endpoints. Do not select conditions or seeds using those
remaining results. Use the original cartpole settings: action repeat 8,
12,500 decisions, 11,500 learning updates, ten fixed test starts 10000--10009,
and the same evaluation schedule. Each model has a fixed 100k training-step
endpoint and 260k additional test interactions. No best-checkpoint selection.

1. Compare correct matching with the smaller update against correct matching
   with the original update. Does the early benefit depend on this strength?
2. Make the same comparison for wrong matching. Does a smaller update reduce
   the harm from incorrect targets?
3. Show the correct-minus-wrong difference under both update rules, with all
   individual seeds. Do not infer a reliable interaction from one favorable run.

Recovery with the smaller wrong-matching update would support sensitivity to
update strength. It would not show that correct correspondence is unnecessary.
Continued poor performance would show that wrong targets remain harmful at
this smaller strength, not that every possible strength behaves that way.
Three seeds, one task and an adaptively chosen follow-up do not establish a
general mechanism or a new algorithm.

## Checks, resources and handoff

Before admission, use the actual upstream learner in focused CPU tests: one
update against an independent oracle, unchanged original arms, unchanged
comparison-parameter update and random draws at a shared initial state,
private permutation-state restoration, and next-update checkpoint restoration.
Use distinct arm identities to reject resumes across different update rules.
Preserve all five optimizer objects and the existing checkpoint format.

Estimated total time is 90--120 minutes, with a 2,700-second cap per model and
about 18 GB peak additional checkpoint space. Save full state periodically and
at the endpoint. Prepare a separate source snapshot; never edit the current
live batch. Admit a waiting owner only after its final seed has launched, or
after the entire predecessor batch ends, so it cannot displace a queued seed.
The allocation deadline is epoch 1791387366. The external run store is
`/project/alex_phd/runs/curl-cartpole-encoder-strength-20261005`.
Its SOURCE.json, ADMISSION.json, LAUNCH.json and HANDOFF.md record hashes,
verification, resource checks and owner identity. Admission is not evidence
that the six new models have trained or succeeded.

See the [wrong-matching protocol](CORRESPONDENCE_CONTROL.md),
[running findings](FINDINGS.md), and
[ranked follow-ups](NEXT_COMPARISONS.md).
