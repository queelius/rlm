# Does executing the plan add value beyond presenting the plan?

September 21, 14:55 UTC (rounded). Accepted and queued after CPU preparation;
source018 and `R/PLAN-ONLY-DECISION-001.json` bind the experiment. Supervisor32879
waits the next-question screen before launch. This follows the fresh fixed-helper
result, not a new dataset search or a claim that recursion is unnecessary.

The final answerer sees all public documents in our current architecture. The
root's questions may therefore help as a plan or reformulation even when the
helper answers add little. The earlier source-removal experiment asked a
different question: could a final answerer work without the original documents?

## Smallest comparison

Use all64 fresh003 parents and both saved repeats for both SFT48 and RL16 roots.
Keep the actual generated root question list. Give the unchanged base final
model its original final prompt, replacing only the helper report with
`{"execution":"isolated","steps":[]}`. Keep full original public documents,
original question, original final seed, temperature .5 and128-token cap. Do not
include resolved questions, helper answers, annotations, or hidden gold fields.
No root or helper generation occurs. This is **plan-only**, not direct answering.

Collect256 new plan-only finals and compare with the saved factual executions.
There are255 factual finals and one SFT dependency failure. Retain that failure
as zero in the planned policy comparison, but attribute any recovery separately
from changes on the255 both-final-available slots. It is not an unknown final.

As a small reproducibility check, replay the original factual final request on
the first four sorted parent IDs, both repeats and root policies: at most16
additional calls. Selection is independent of outcomes. If a selected factual
final does not exist, record that control as unavailable; do not replace it.
Compare native emitted tokens and official grades against their saved source.
Any disagreement is a limitation to report, not a reason for silent retries,
selective replacement, or an unannounced larger experiment.

## Cost, accounting and interpretation

Maximum272 new calls, one resident frozen4B model, one exclusive owner and a
20-minute collection cap. Use the existing native HF final-generation path with
all adapters disabled. Loading an inert adapter for the existing client is an
implementation detail, not use of its trained weights. Save prompts, token IDs,
seeds, model/source hashes and owner/terminal receipts. No optimizer is created.
Verify an actual decoded response promptly. Unknown calls remain missing.

Primary: paired EM/F1 differences, separately for SFT and RL, over all64parents
x2repeats; connected-component bootstrap with20,000 draws, seed2026092173.
Report protocol-involved changes separately from both-valid final changes,
component/parent counts and observed versus missing denominators. Direct54/128
remains an unchanged descriptive control, not a newly sampled comparison.

New physical collection cost is separate from reused acquisition. Hypothetical
plan-only inference charges its original root once plus the new final, omitting
helper calls. Do not claim measured deployment wall-time speedup from saved
root reuse or summed service latencies. Count extra reproducibility controls
as research cost, not policy inference cost.

Removing a trace changes prompt content and length; this comparison does not
isolate each answer's causal contribution. One saved plan and final seed per
repeat does not establish an expected-value effect. If plan-only preserves
quality while removing most calls, investigate cheaper planning/answering or
a task where information must actually be delegated. If factual execution
helps, that supports studying helper reliability and feedback. Neither result
alone decides whether long-context RLMs or learned recursion are useful.

The actual sealed CPU preflight verified256planned slots,255factual finals and
all16fixed replay controls. Earlier CPU-preflight001/002 are preserved staging
receipts; `R/plan-only-001/PLAN.json` is the authoritative sealed manifest.
