# Completed short LLM RL reproduction

## Updated result and lesson-centered presentation

Scientific cutoff: 7 October, 19:40 UTC. The scheduled update-160 monitor is
**48/64**, following 20, 42, 42, 43, 43. All 64 new outputs independently
regrade and all ordered questions, references and inputs align. Six gains and
one regression since128; 28 gains and no losses versus the initial monitor.
See [the receipt](interim-monitor-1940.json). No final500 result is inferred.
Training health was checked at160 learning calls, with finite gradients and
saved model/optimizer labels152/160. Current owner continues unchanged.

The user now prioritizes lessons and an RLM research direction over an exact
paper-score chase. The six-slide deck gives the reproduction one slide, then
explains reward contrast, prompt/interface fit, a four-stage curriculum diagram, relevant prior work,
and a portfolio of small controlled experiments rather than one settled question.
The fifteen-page learning PDF adds lessons, concrete examples, related work,
possible blind spots and the experiment portfolio. Read
[RESEARCH-QUESTIONS.md](RESEARCH-QUESTIONS.md) for the broader discussion.

The hypothesis is not that we never taught RLM basics: the earlier audit already
records SFT learning executable routines with limited flexible transfer. Test
which prerequisites are missing, whether teaching order matters beyond the
same examples shuffled, and whether RL adds gains beyond SFT alone. DAPO,
RLM training, DeepSeek-R1 cold start and STaR are linked primary precedents.
No new RLM experiment or live source change was made for this discussion.

Three CPU literature agents completed five scoped reports: RLM training,
curriculum, competence/exploration, parent/helper credit, and teaching/repair.
The synthesis distinguishes protocol from dependent composition, example order
from information availability, reward from observations, and teacher correctness
from student learnability. Reports remain in the external brainstorm-20261007
folder; public prose links the primary sources directly. These findings support
experiments, not a claim that curriculum explains all earlier failures.

Continue the fixed255 training and six final conditions under the same owner.
If the final endpoint confirms meaningful gains, fresh training replication
outranks blind budget expansion. The smaller learning-rate/reuse screen remains
ready for diagnosis after whole-owner release. RLM curriculum experiments remain
proposed and require a bounded protocol before launch.

## New monitor point, 19:00 UTC

The update-128 check is **43/64**, unchanged from update 96. One question improved
and one regressed. All 64 answers independently regrade; question, reference and
formatted-input checks agree across all five points: **20, 42, 42, 43, 43**.
The longer run's early gain persists but further monitored improvement is small.
This is not a final MATH500 score or a demonstrated gain over the stronger base.
See [the new receipt](interim-monitor-1900.json).

At the cutoff, training had completed 129 learning calls, with 129 finite positive
gradient norms and no logged training traceback or CUDA OOM. Model and optimizer
checkpoint labels 120 and 128 were present. Continue the existing whole owner,
its four final-evaluation jobs, and the prepared smaller-run queue below.
The overview, chart, slides, notes and learning guide now use the 19:00 cutoff.

## Reading package and next experiments, 18:40 UTC

Start with [the self-contained overview](START-HERE.md). It explains our longer-term
RLM goal, why we are reproducing a math-RL paper first, how the training works,
what the scores mean, and what we still have not established. The overview and
five-slide PDF are published on GitHub main at `8292ebe`. The two charts separate
the prompt-format comparison from the current training-progress curve. Scientific
results still have the 18:20 cutoff; there is no new final benchmark result.

The user now prioritizes faster, smaller exploratory comparisons. A three-run
[512-question screen](../../experiments/r1_replication_20261007/quickscreen512-readiness.json)
is prepared and reviewed, **not launched**. Each run starts from the original model:
the current learning rate, a five-times-larger rate, and two learning passes over
each response batch. These make 32, 32, and 64 optimizer updates, respectively.
All three use the same 512 training questions and 4,096 sampled answers. A fresh
baseline and every final model will answer the same 128 evaluation questions in
both input formats. We keep all outcomes, not just the most favorable one.

Eleven focused CPU tests pass. Native data loading confirms 512 usable questions,
32 complete batches, and no exact overlap with the evaluation questions. Data and
source hashes are verified. The exact executed preparation script is archived;
a separate receipt records subsequent source formatting without changing the data.
This is preparation evidence, not proof that the new training runs will succeed.

Current attempt 4 and its four final-evaluation jobs remain unchanged under whole
owner PID 1861599. Its native log has advanced beyond update 112 without a training
traceback. The short screen may start only after that entire owner releases the
GPU, not merely when training exits. No waiting successor process exists. The older
255-update learning-rate follow-up remains available but is now lower priority.
The existing observer continues to request reviews; read the external session
checkpoint and queue before admitting a successor. Model weights are not in GitHub.

## Latest review, 18:20 UTC

Attempt 4 continues unchanged. At 18:20:58 it had 99 completed learning calls,
latest printed counter 98, 98 finite positive gradient norms and no logged
training errors. Saved model labels 88 and 96 were present. Whole owner
1861599 and observer 1884898 remain authenticated and active.

The complete monitor curve is now **20, 42, 42, 43 out of 64**, at updates
0, 32, 64, 96. The latest point has five gains and four losses relative to 64;
all answers independently regrade. Collection 96 also passed a fresh audit:
128 responses, 71 rewarded, 11 of 16 questions with mixed outcomes, four
no-EOS answers correctly zeroed, finite aligned logprobs and zero grading
disagreements. These changing-question rewards are not a test learning curve.
See [new monitoring and training evidence](interim-monitor-1820.json).

The five-slide and ten-page learning PDFs now have the 18:20 cutoff. A new
guide page explains a corrected dice answer and a wrong claimed code output,
with arithmetic anyone can check. No generated code was executed. Full-run
benchmarks remain pending; no best-checkpoint selection or early stopping.

Future LR5e-6 launchers have passed five scoped CPU cleanup tests under the
pinned Python environment and independent review with no material blocker.
The documented launch command preserves any previous owner log. See the
[readiness receipt](../../experiments/r1_replication_20261007/lr5e6-readiness.json).
Nothing new has been launched. Current training and four final-evaluation jobs keep their
owner. Main quota45% at18:22:56; reserve10%, dispatchheadroom12%. The supplied
allocation epoch1791621574 is **10 October08:39:34UTC**; earlier calendar
summaries saying9October20:39 were wrong. Observer cutoff is five minutes earlier.

## Current research decision at 17:55 UTC

Attempt 4 remains healthy and unchanged: 77 completed learning calls at 17:52,
latest printed optimizer counter 76 (logging can lag), no reported errors.
Full owner PID1861599/startticks1319284714 still owns training and all six
final test conditions. Observer PID1884898/startticks1319852094 is active.
Main account quota49% at17:50; preserve10%, dispatchheadroom12%.

The five-slide deck and nine-page learning guide now carry the 17:45 evidence
cutoff and the complete20/42/42 monitoring curve. Their short-run endpoint
table remains unchanged. Both PDFs compile and fit their pages; revised slides
and guide pages have been visually checked. See the presentation verification.

The [training signal audit](training-signals-1740.json) found no reward
mismatches in1,536 responses and no numerical gradient failure in67 logged
rounds. Mixed-success questions remain common. Late sampled failures are
mostly mathematical errors after valid formatting. Zero clipping is expected
under this one-update collection schedule, not evidence of ineffective steps.
Changing-question reward trends are descriptive, not a causal learning curve.

The [recipe audit and two fixed-endpoint plans](recipe-opportunities-1745.json)
rank fresh-base LR5e-6 first, then independently two proximal learning epochs
atLR1e-6. The first preserves255 updates; the second preserves255 collections
but requires510 updates and a different completion guard. Neither is launched.
A CPU agent is preparing separate LR5e-6 launchers; the existing training and
observer source are sealed. No new GPU admission until the WHOLE current owner
has completed, including evaluations, and enough allocation remains.
Do not conflate proposals with a new improvement or select favorable checkpoints.
Next review should prioritize fresh training replication if the current endpoint
already supplies a meaningful gain, otherwise use the ranked contrast.

## Access restored and research continuing, 17:37 UTC

The user restored full access in this same session. Training was not restarted
or interrupted: its authenticated owner and GPU children remain active.
At 17:36, 65 learning calls had completed without reported errors. Latest
saved model labels were 56 and 64. The old review observer had exited for an
unknown reason; its unchanged source/config/state have been restarted under
detached PID1884898, startticks1319852094, with redirected streams and no GPU
actions. A new review was successfully queued to this exact chat. Main-account
quota is 50%. The earlier restricted-access notes below are historical now.

The fixed monitoring scores are **20/64, 42/64, 42/64** at updates0,32,64.
All64 answers at both later points were independently regraded with no
mismatches. Between32 and64, five answers improved and five regressed, leaving
the total unchanged. See [intermediate evidence](interim-monitor-1737.json).
No full-run endpoint exists yet. The user accepts clearly labeled interim
results for the report due around **20:00 UTC today**. Preserve all scheduled
monitor points, continue the255-update run, and keep its final comparisons.

## Resume here: 7 October, 17:32 UTC

**Restart safety correction:** Do not assume the training process survives
closing Codex. It was launched through a Codex-managed exec session; the
sequence script runs its jobs in the foreground and does not itself detach.
The restricted session cannot inspect the original process ancestry, and
polling exec30612 returns "Unknown process id" even while native logs advance.
Thus job survival across assistant shutdown is unverified. Leave the current
session open; if permission changes require another session, open it separately
and inspect the existing owner before closing anything or starting another job.
Saved checkpoint files persist independently, but that is not a guarantee of
uninterrupted training or exact restart from optimizer state.

The user is considering restarting this Codex session to restore its earlier
permissions. Do not restart training or relinquish the GPU allocation just to
restart the assistant. First identify the existing full training/evaluation
owner. A missing PID in a restricted process view is not proof that it exited.

**Latest actual progress:** attempt 4 has 62 completed learning calls and native
optimizer counter 62, no reported training errors, and a log updated at 17:31:04
UTC. Saved model directories currently include `step_00048` and `step_00056`;
the owner rotates checkpoints, retaining the latest two. Earlier checkpoint
labels in this document are historical, not guaranteed to remain on disk.

**Intermediate result:** on the same 64 monitoring questions, the starting
model scored 20/64 and the update-32 model scored 42/64. The 22 changed outcomes
are all improvements, with no regressions. All 64 update-32 answers were
independently rescored using the full verifier at 17:30 UTC, with zero
mismatches; prompts, references and formatted inputs match the baseline.
The update-32 JSON SHA256 is
`7f422ea1e1f9c52e0de494a49389f60a02f999847e9e7dcf62527b7b73b0711f`.
This is a repeatedly observed development monitor, not a new MATH500 final
test, training replication, or evidence that the larger run has already beaten
the strong question-only starting model. The earlier short chat run scored
38/64 at update32 and 35/64 at its final step33 with the native fast checker.
Full regrading gives 39/64 and 36/64 respectively (and 20/64 before training).
Use those full-checker values when comparing with the current full-checker
monitor; preserve both evaluations, not the better one.

**New reporting target:** the user requested reportable results in about 2.5
hours, around **20:00 UTC on 7 October**, and explicitly accepted intermediate,
non-full-run results. Keep the complete fixed-monitor learning curve, label its
64-question denominator and input format, and retain every scheduled point.
Do not select a favorable checkpoint or shorten the accepted 255-update run
just to obtain a headline result. At the current pace, the full endpoint is
likely later than this reporting target. Prepare a readable interim synthesis;
the existing final MATH500/AMC/Minerva evaluations remain queued in the same
owner. No new GPU job or training-source edit was made in the restricted session.

**Monitoring/access caveat:** this session changed from unrestricted access to
managed workspace-write, with restricted network and no approval escalation.
Cause unknown; the assistant did not change the setting. Native logs remain
readable and continue to advance, but GPU driver access and the external owner
processes are not visible here. Do not infer that the GPU stopped. The current
observer's STATUS.json last updated at 16:48 UTC, with no pending review then;
unattended review delivery is not currently verified. Main-account quota read
at 17:30 failed (unknown, not zero); last successful value was 52% at 16:34.
Preserve the 10% reserve and 12% dispatch headroom. Restore permitted access,
check the real existing observer/owner, and repair monitoring only if necessary.
Do not launch a duplicate observer or edit sealed live source.

Key resume paths:

- Source worktree: `/project/alex_phd/repos/rlm/.worktrees/r1-replication-20261007`.
- External checkpoint: `/project/alex_phd/runs/r1-zero-replication-20261007/SESSION_CHECKPOINT.md`
  (last updated 16:36; this section contains the newer state).
- Run: `/home/atowell/research-runs/r1-zero-replication-20261007/larger4096-chat-drgrpo42-attempt4/debug_1007T16:02:41`.
- Log: `/project/alex_phd/runs/r1-zero-replication-20261007/larger4096-chat-drgrpo42-attempt4.log`.
- Observer: `/project/alex_phd/runs/r1-zero-replication-20261007/reviews-longer-v4/CONFIG.json`.
- Full owner: exec30612, session1861590, outer timeout1861599/startticks1319284714,
  trainer1861602/startticks1319284715; authenticate afresh before any intervention.
- Last verified GitHub main: `17e609d0725e05caaa911ff960120dc9b092d346`.
  This newer resume section is saved locally, not yet committed or pushed.

Resume the live owner, analyze new monitor points, update the <=5-slide deck and
learning guide if the new evidence changes the story, and publish a verified
milestone once permitted. GitHub holds documents and source, not model weights.

**Progress check, 16:36 UTC:** The longer run has completed 22 learning calls
and saved model and optimizer files at `step_00008` and `step_00016`.
Twenty-one reported gradient norms are finite and positive. An audit of 1,920
new responses found no prompt/reference or token/reward alignment errors;
eight position-selected answers were independently regraded, with agreement.
See the [bounded audit](longer-attempt4-progress-1636.json). No final score
exists yet. Continue owner exec30612 unchanged through training and its final
evaluations; do not admit another job at trainer exit. Checkpoints are saved,
but exact training-resume semantics remain unvalidated.

**Latest operational update, 16:03 UTC:** Attempt 4 started at 16:02 under
exclusive sequence owner exec30612, with its own training and all six final
evaluation conditions. The old observer was replaced by `reviews-longer-v4`;
read external `SESSION_CHECKPOINT.md` for authenticated process identities.
The small A100 memory check passed; it does not yet establish full-run stability.
All four broader reference conditions are complete and independently checked;
see [results](README.md) and [receipt](broader-reference-receipt.json). No
larger-run trained-model endpoint exists yet. The PDFs remain dated 15:55.

**Earlier phase, 7 October at 15:55 UTC:** The user requested continued work toward
reproducing the paper's training result. A fresh-base run of roughly 4,000
questions and 250 weight updates is prepared, not yet launched, with fixed final
evaluations planned after completion. Attempt 1 failed with GPU memory exhaustion;
the allocator retry failed at model initialization. Attempt 3 completed 12 updates
and sampled 13 collections before another memory failure; it saved no checkpoint.
The [failure receipt](longer-attempt3-failure.json) is retained, not a benchmark result.
Prepared attempt 4 uses a private memory adapter with three passing focused CPU
tests, including matching actual learning-step updates. The shared environment
and official source remain unchanged; GPU memory stability is not yet established.
Its loader-derived budget stays at 4,080 questions and 255 updates, with saving
every eight updates. The owner will handle all six final tests: MATH500, AMC,
and Minerva in both chat and question-only formats. Its caps are ten hours for
training and thirteen hours for the whole sequence. Separate broader checks of
the base and authors' released models are underway, not our training results.
The authors' MATH500 score is 366/500 (73.2%), versus 74.2% reported; those are
their released weights, not a model we trained. Read the
[longer-run plan](longer-training-plan.md) and current
external `SESSION_CHECKPOINT.md` before touching GPU processes. The completion
notice below describes only the earlier presentation batch. Actual allocation
ends 9 October at 20:39 UTC; the 14:00 window below is historical.

Historical batch status: 7 October 2026, 13:51 UTC. The requested five-slide presentation was
published before 13:00 UTC. The optional experiment batch is complete before
14:00 UTC. No training or evaluation job remains active from this batch.

## What to take away

We ran the authors' LLM RL code, verified sampled answers, reward calculations,
nonzero updates and saved weight changes, and evaluated prescribed final
checkpoints. This validates a working implementation path, not the paper's
published numerical results or RL for recursive language models.

The strongest lesson is to check the starting prompt. Without any training,
the base model scored 154/500 in chat style and 305/500 with the question alone.
Chat-style Dr. GRPO training raised the chat score to 308, but that large gain
does not by itself show newly acquired mathematical ability.

Starting from the original model and training on questions alone gave 314 and 306
in two Dr. GRPO runs. Standard GRPO gave 318 in one further run. Each used 512
questions and 32 updates. These are small exploratory gains over 305, with
insufficient training replications to establish dependable gains or rank the
algorithms. Standard GRPO also scored 169 in chat style, so prompt sensitivity
persists. All results, including regressions, are retained.

Six fixed-weight repeated evaluations were byte-identical to their originals.
That checks evaluation repeatability under the same settings, not training
replication. Earlier monitoring outputs did vary at unchanged weights, so we
do not claim that all greedy GPU generation is deterministic.

## What to try next, in order

1. Run a longer fresh-base comparison against the strong question-only
   baseline, using the paper's schedule as the reference. Fix the training
   budget and final-checkpoint rule before inspecting results.
2. Repeat a meaningful gain with fresh training realizations. More test
   questions and repeated tests of the same weights cannot replace this.
3. If an algorithm difference persists, separate the two normalization
   changes and effective update scale. The present GRPO switch changes them
   together, so it cannot identify the cause of a score difference.
4. Return to RLM training with these checks in place: inspect actual inputs,
   rewards and learned behavior; distinguish a stronger starting setup from
   a training gain; grade tool execution separately from plausible code text.

The longer retry is prepared and broader reference checks are underway, as
recorded above. The remaining items are ranked future experiments, not claims
of success. The earlier presentation window is no longer the research deadline.

## Preserve and resume

- Read the [learning PDF](learning-guide.pdf), [research record](README.md),
  and [five-slide presentation](../../slides/2026-10-07-r1-replication/research-update.pdf).
- Operational checkpoint and queue:
  `/project/alex_phd/runs/r1-zero-replication-20261007/SESSION_CHECKPOINT.md`
  and `NEXT_JOBS.md` in that directory.
- Training outputs and full final model/optimizer checkpoints:
  `/home/atowell/research-runs/r1-zero-replication-20261007/`.
- Native evaluations, immutable protocols, and review receipts:
  `/project/alex_phd/runs/r1-zero-replication-20261007/`.
- Official source and environment are pinned in the research record. Do not
  modify sealed source or reuse transformed dataset caches between conditions.
- Saved optimizer state exists, but native resume has unverified continuation
  semantics, including actor synchronization and data position. Prefer a fresh
  base run for the next scientific comparison; do not call resume equivalent
  to uninterrupted training without testing it.

GitHub contains the PDFs, source and lightweight evidence, not the heavy
checkpoint files. Keep both external stores when changing allocations.
