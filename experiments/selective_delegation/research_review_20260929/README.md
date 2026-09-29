# Actual periodic research review, not just a status logger

The user requires Codex to interpret new findings and adapt the research plan
without another human prompt. Before this change, GPU jobs kept running after
the final answer, but no model review was scheduled. That distinction matters.

## Mechanism and current verification

The installed Codex0.158.0 CLI provides `codex queue --thread UUID --message TEXT`.
The observer in [monitor.py](monitor.py) queues into the exact existing research
session, never `--last`, a different project, or a new model instance.

- Poll explicit queue-stage receipts and active service STATUS files every30s.
- Queue on new completed stages, coalescing routine events for five minutes.
- New service faults bypass that delay; identical alerts are deduplicated.
- Review after20minutes even without a changed terminal receipt.
- Permit only one pending review. It remains pending until the model records an
  interpretation, next decision, and evidence pointer. Events arriving during a
  review remain eligible for the following review.
- Quota is read from the supported main account bucket, no more than15minutes
  old. Unknown quota pauses admissions and retries. No new admissions at≤12%,
  leaving headroom for the user-supplied10% reserve.
- STOP or the allocation deadline ends the observer; scientific owners remain
  independent. Failed/ambiguous queue admission cannot silently duplicate work.

Live observer PID1257601 started05:39:43UTC. **The first same-session queued
wakeup was delivered and reviewed at05:48UTC.** Probe
`rlm-wake-20260929-0528`, native message
`01a0eba2-10d2-72b0-85dd-39d842ec7733`, actually started a subsequent model turn.
After checking the native results and GPU returns, Codex wrote its decision and
evidence acknowledgement at05:48:42; the observer consumed it at05:48:44, clearing
the pending request and setting `delivery_verified:true`.

This verifies native queued-turn delivery plus the review/acknowledgement cycle,
not perpetual client availability. The first observer-generated event request,
`review-1790661224-a88aaa50`, was queued at 05:53:44 UTC and actually reached
the model around 06:24 UTC, after the active presentation turn finished. Its
[scientific review and next decisions](reviews/2026-09-29-0624.md) check live
returns, native results, recovery-data qualification and relevant prior art.
The 70 receipts in that initial request include already-reviewed historical
events; they are not 70 newly completed experiments. Queueing does not interrupt
an active turn. The 20-minute fallback applies when no earlier event is due.

Ten focused tests pass, including quota bounds, coalescing, exact UUID targeting,
incomplete JSON, ambiguous admission, pending/restart behavior, acknowledgement,
evidence requirement, and STOP/deadline arriving during quota sampling. A separate
review caught the last timing gap; it was reproduced before being fixed. No full
RLM regression suite was needed for this independent operational tool.

## Live artifacts and stopping

External store:
`/project/alex_phd/runs/rlm-research-r4/operations/2026-09-29-research-review/`.

`CONFIG.json` pins the exact thread, two accepted queues, quota sampler, and
deadline. `LAUNCH.json` records PID/source hash. `STATE.json`, `STATUS.json` and
`QUOTA.json` explain pending work and budget. Requests and model acknowledgements
are saved separately under `requests/` and `acknowledgements/`.

Thread: `01a049e7-2027-7253-97c4-c70906238762`.
Current allocation5932 endsOctober1 09:50:35UTC; observer stops09:40:35UTC.
Creating `STOP` in that exact operations directory ends only this observer.
No accepted jobs, owners, locks, or scientific input sources are changed.

After a real review, record its decision with:

```sh
/project/alex_phd/envs/rlm/bin/python experiments/selective_delegation/research_review_20260929/monitor.py \
  --config /project/alex_phd/runs/rlm-research-r4/operations/2026-09-29-research-review/CONFIG.json \
  --ack REVIEW_ID --decision 'Interpretation; next research decision' --evidence RESULT_PATH
```

Then yield to allow the next native queued turn. Do not acknowledge from the
observer or claim a review because a message was merely queued. If the pending
message is not consumed, diagnose native delivery; do not stack replacements.

## Limits and sources

This is local same-session queueing, not a cloud automation. It depends on the
existing client/session staying available. Quota is shared with another project;
the reserve is a conservative gate, not an account-wide lock. New accepted queues
must be added to a reviewed replacement observer configuration; periodic reviews
also inspect the canonical current research checkpoint.

Installed `codex queue --help` is the source for the exact supported invocation.
The [official automation guide](https://learn.chatgpt.com/docs/automations?surface=app)
distinguishes local tasks, which depend on the local application, from cloud tasks;
it does not establish this CLI probe's end-to-end delivery. The current CLI
[reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli) describes
queued follow-up turns. Live delivery still requires an observed model turn.
