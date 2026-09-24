---
date: 2026-09-24
evidence_cutoff_utc: "07:15"
status: exploratory
questions: [learnable-demonstrations, memory-interface, terminal-credit]
---

# Completed RL comparison: no advantage from positive-only credit

The four previously unavailable cases were collected separately and all failed
the task. The complete result is now **15/32 for both positive-only and signed
credit**. There are three paired wins, three losses and26 ties. The task-group
bootstrap interval for the difference is−12.5 to+12.5 percentage points. This is
not evidence that the methods are equivalent; it is no detected improvement in
this small comparison.

The original one-hour result remains preserved. The supplemental evaluation used
about16 minutes of additional collection wall time and restarted the interrupted
case from its initial state. It did not retry any of the28 completed cases. All
original and supplemental calls are charged, including the interrupted prefix.
This is complete coverage under the same per-episode limits, not equal total
evaluation time. [Compact evidence and source hashes](morning-results-20260924.json).

## The third memory comparison also gives no clear gain

For the RL-trained actor, success was14/32 with full history and14/32 with full
history plus the recipe notebook. It was13/32 with the latest four interactions
and13/32 with those interactions plus the notebook. All128 outcomes are known.
Tied counts can conceal changed individual outcomes; use the paired records
before making any equivalence claim. Across all three actors, the notebook is
not a reliably beneficial change without further training or interface design.

## What happened while Codex was inactive?

Queue020 completed the memory audits, supplemental RL collection and RL audit.
The two world47 quantity-control collectors failed before loading the model.
The saved plan referenced the working-copy wrapper path, but execution generated
the sealed-copy path. The wrapper bytes matched; the path keys did not. The
preflight had not exercised the actual sealed entry point in a fresh process.

The queue then ended without an independent fallback. This left roughly97 minutes
of the old allocation unused, from02:22 until03:59 UTC. That is an operational
failure, not scientific compute. The failed attempts and sealed source003 remain
unchanged. Source004 was prepared and then rechecked through two separate actual
sealed CLI invocations. It runs only the corrected arm and reuses the existing
public comparator, avoiding duplicate rollouts.

At07:08 we found a new allocation5902 and an idle GPU. Queue021 launched the
already-prepared world47 teacher comparisons for two training seeds. Real model
responses were confirmed promptly, with zero transport failures. Queue022 waits
for those collectors and runs the corrected control plus its paired analysis.
The corrected arm has a30-minute total cap versus45 minutes for the public arm;
only complete coverage supports the full-panel comparison. Unknown cases must
not be treated as failures or silently dropped.

## Meeting-focused decision

The five-slide [advisor draft](../../slides/2026-09-25-advisor-meeting/README.md)
centers the repeated teaching advantage. It does not claim RL improvement or
recursion benefits. The next changed-world control directly tests an important
limitation in that story. Update slide4, notes and evidence together when its
native audit finishes; include contrary results if the advantage disappears.

Beyond this control, a promising next experiment separates the model's choice
of item from exact recipe-argument construction. Keep this labeled proposed.
The new user priority is supporting a clear, credible five-slide report for
September25 at13:00; the meeting's timezone was not specified.
