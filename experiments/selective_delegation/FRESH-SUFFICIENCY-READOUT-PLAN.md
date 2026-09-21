# Canonical paired sufficiency readout proposal

The current candidate is `sufficiency-canonical-inputs-003`: 32 official
MuSiQue DEV parents / 64 paired variants, selected label-blind as the first 32
SHA256(`2026092192:` + original ID) after excluding recorded non-TRAIN study
inputs and breadth exposures, plus all official TRAIN parents/components. It has
303 eligible parents and is naturally all two-hop; it is therefore an unexamined
identity-disjoint panel, not a depth-balanced generalization panel.

The selection audit finds zero selected intersections with those known inputs by
parent, normalized question, or atomic component. It is not document- or
pretraining-clean: 183 of 729 selected unique documents exactly overlap official
TRAIN, across 58/64 variants (title overlap in every variant). Paired variants
retain their natural document/length changes, so the endpoint tests official
answerability labels rather than a causal support-deletion intervention.

The fixed readout is base, joint checkpoint-0032, and positive-only
checkpoint-0032, with the original full-public-context prompt, temperature .5,
top-p 1, top-k 0, output cap 128, and seeds 2026092181/2026092182: 64 variants
× 2 seeds × 3 arms = 384 new calls. Official group answer+sufficiency EM/F1 is
primary; parent-clustered analysis treats repeated variants/seeds as dependent.

`sufficiency-fresh-inputs-001` is retained as the initial preflight candidate:
its 15/10/7 two/three/four-hop mix was not valid for this canonical comparison
because later exposure audit found prior-study intersections. `-002` is retained
as an unexecuted failed CPU preflight (missing official metric hashes). Neither
has outcome data.

Main accepted the fixed comparison at 17:51 UTC. Validation then caught a stale
source034 launcher pointer to nonexistent inputs-004. No GPU job ran with that
pointer. The additive source034b launcher and `SUFFICIENCY-FRESH-DECISION-002.json`
preserve the failed attempt and bind the same authoritative inputs-003 and exact
committed endpoints. Repaired validation passed; supervisor34405 waits for the
local-reason ALF control to release the GPU. The checked-in launcher reflects
this repair, not the abandoned source034 pointer.
