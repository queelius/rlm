# Event-driven research review

The user requires periodic **reasoning and adaptation**, not just a running GPU
queue. The last checkpoint left no mechanism to resume Codex after its final
answer. GPU work continued, but findings went without model review for hours.

Use the installed `codex queue --thread UUID --message TEXT` command to enqueue
reviews in this exact session. A local stdlib dispatcher observes the two
explicit accepted queues. It queues on changed completed-stage receipts or
active service failures. Routine events are coalesced to at most once per five
minutes, with a 20-minute fallback. Newly observed service faults bypass that
delay; repeated instances of the same fault do not. It never controls GPUs,
mutates accepted queues, or starts
another model session. Existing scientific owners remain independent.

Only one review may be pending. A review is complete only after Codex writes an
explicit acknowledgement containing an interpretation, a decision, and evidence
pointers. Admission to the native queue is not proof of delivery. The initial
already-queued live probe must be acknowledged by a subsequent model turn before
we claim end-to-end operation. A lost/ambiguous submission stops new submissions
until inspected, rather than duplicating possibly accepted work.

Check the supported main account quota before submission (reuse a read no older
than 15 minutes). Stop new reviews at 12% remaining to leave headroom for the
user-supplied 10% reserve; unknown quota pauses submissions and is retried. End
ten minutes before the current GPU allocation, or upon a STOP sentinel. Local
scientific jobs continue under their own caps. This depends on the existing
Codex client/session remaining available; it is not an always-on cloud service.

Every review checks actual model returns, outcomes/failures, scientific meaning,
the ranked next experiments, resource limits, and publication/slide relevance.
It may prepare and launch new scoped experiments using existing owner rules.
It must preserve diagnostic B exposure limits and immutable live sources.

Store dispatcher config, PID, source hash, state, request/acknowledgement records,
and log under the external research operations store. Publish source and concise
operating instructions. Use focused fixtures and one real native-queue probe,
not a broad regression campaign.
