# Publication handoff: October 7, 2026

The October 5 restrictions no longer apply in the current session. GitHub
remote reads work again. The user requested publication of the learning guide
and latest results. All nine fresh models finished on October 5 at 20:22 UTC;
there is no remaining work in that queue. See the [latest findings](FINDINGS.md)
and [complete native-data summary](fresh-seed-data/SUMMARY.json).

The updated fifteen-page guide reports both matching methods winning twice
and losing once against no matching. Smaller-update CURL beats original CURL
once and loses twice. All nine endpoints and 234 evaluation averages were
checked against native episode records. Publication includes the PDF, its
source, supporting explanations and compact native records, not model weights.

Current allocation is job 5990 on an22. Its environment reports end epoch
1791621574, different from the old observer deadline of 1791387366. Do not
reuse the old allocation assumptions for a new campaign. The observer's
latest saved main-account quota was 100% at 08:32 UTC on October 7. Preserve
the reserve and query live quota before launching optional work.

The old observer still has pending request `review-1791226637-bc5542de`.
Publication does not by itself repair periodic dispatch, renew the observer
deadline or start GPU work. The current GPU was idle. Reconcile observer and
allocation state explicitly before claiming autonomous research has resumed.

The historical handoff below explains the interruption and is retained as
history, not a description of current running jobs or current permissions.

## Historical local handoff: October 5, 2026

The scientific review of `review-1791226637-bc5542de` is complete locally.
Read [the review](reviews/REVIEW-1791226637-bc5542de.md) before the external
checkpoint, which still describes the preceding, incomplete comparison.

The first fresh seed favors no matching (823.97) over smaller-update CURL
(668.22) and original CURL (280.41). This is one seed, not a reliable ranking.
The [15-page learning PDF](learning-guide.pdf), source and supporting records
are updated locally, but **not committed or pushed**. Last previously verified
GitHub main commit: `59d2f88ac901cc8d90718160187e3cd9ce9de39d`.

## What continues

The existing owner 1860062 continues the nine-model fresh-seed queue, without
any process or source changes from this review. At the last live check, the
fourth model, smaller-update seed 567 (child 1875480), was learning and five
jobs followed. Read current native logs before acting; do not duplicate it.

- Native root: `/project/alex_phd/runs/curl-cartpole-fresh-seeds-20261005`
- Observer config: `/project/alex_phd/runs/curl-replication-20261004/reviews/CONFIG-v9.json`
- Existing observer PID: 1861292. Its pending request is not cleared.
- Allocation deadline: epoch 1791387366 (October 7, 15:36:06 UTC).
- Preserve the full cohort and all three comparisons per seed. No new probe
  or tuning sweep is admitted by this review.

## What could not be completed

Current developer permissions prohibit writes to external run stores and Git
metadata; network access is also restricted. Do not bypass those restrictions
through another process or transport. No acknowledgment, external STOP file,
checkpoint update, commit or push was made. The existing GPU jobs do not
depend on those writes. Automatic review dispatch may remain pending until
the acknowledgment can legitimately be written.

The live quota query failed at 18:58 UTC. Last known main-account quota was
78% around 18:44 UTC, not a current guarantee. Query again before admitting
new optional work; preserve the 10% reserve and 12% dispatch headroom.

## Resume only when the relevant operations are permitted

1. Inspect actual new native results and the existing owner. Complete the
   remaining fresh-seed comparisons without changing seeds or selecting peaks.
2. Read the main-account quota. Recheck the local diff, native-copy equality,
   PDF hash and current remote before an ordinary non-force publication.
   Include the new control evidence, JSON summary, guide, supporting documents
   and review record, but no external model weights.
3. Reconcile the external session checkpoint with this review. Acknowledge
   the exact pending request using the command below, then read back its
   receipt. Do not claim this happened merely because the review exists.
4. Inspect observer state before expecting automatic future reviews. Record
   additional completed runs as new evidence, not as duplicate receipts.

Pending acknowledgment command, for an environment that permits its writes:

```sh
/project/alex_phd/envs/rlm/bin/python \
  /project/alex_phd/runs/curl-replication-20261004/reviews/code-v2/review_monitor.py \
  --config /project/alex_phd/runs/curl-replication-20261004/reviews/CONFIG-v9.json \
  --ack review-1791226637-bc5542de \
  --decision "Fresh seed 234 favors no matching over both CURL update rules; continue all declared fresh seeds unchanged and report the added cohort separately." \
  --evidence /project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/reviews/REVIEW-1791226637-bc5542de.md
```
