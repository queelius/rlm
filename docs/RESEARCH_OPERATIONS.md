# GPU-First Research Operations

## Governing priority

Our accelerator allocation is temporary. Every avoidable idle GPU-hour is information we can never
recover. The default is therefore to run informative experiments aggressively, learn quickly, and
adapt. This is research software: production-style hardening is not a prerequisite for exploratory
runs.

## Two evidence lanes

1. **Exploratory:** launch quickly with a question, immutable input snapshot, metric, seed, compute
   cap, checkpoint policy, and artifact directory. Failures are evidence. Diagnose, repair, rerun, or
   pivot without ceremony.
2. **Confirmatory:** promote only promising findings. Add held-out data, multiple seeds, stronger
   provenance, independent review, and statistical uncertainty when the result may support a claim.

Exploration must not wait for confirmatory infrastructure.

## Continuous operating loop

1. Keep the GPUs working on the highest-information ready jobs.
2. Prepare and validate later jobs on CPUs while GPU jobs execute.
3. Checkpoint long training so a lease loss or session restart does not erase progress.
4. Analyze every completed or failed run promptly, including error categories, learning curves,
   resource use, and unexpected behavior.
5. Update the live queue with the decision changed by the result and ranked follow-up questions.
6. Replicate strong signals, ablate plausible causes, deepen promising mechanisms, and retire or
   reformulate directions that stop producing information.
7. Search current primary literature, especially arXiv and official implementations, when findings
   suggest a connection, mechanism, or alternative worth testing.

## Experimental defaults

- Use cheap pilots and multi-fidelity sweeps before committing to long jobs.
- Prefer paired comparisons, matched data, and focused ablations over unrelated headline scores.
- Calibrate task difficulty so RLVR rollouts include both valid successes and valid failures.
- Track training and evaluation curves; do not rely only on final loss.
- Run additional seeds after evidence of a signal, not as a prerequisite to obtaining one.
- Measure correctness, trace validity, cost, latency, model calls, and GPU utilization separately.
- Preserve negative and invalid results with an explanation; do not silently relabel them.
- Compare model-weight changes, harness changes, and their interaction so co-adaptation is measurable.

## Ready-queue rule

Maintain at least two useful follow-on jobs whenever practical. If a reserved GPU is idle and no job
is ready, preparing the smallest decision-relevant job becomes the immediate priority. Documentation,
general refactoring, broad test suites, and presentation work do not outrank an independent ready GPU
experiment.

## Readable research records

Use the [results reading guide](RESEARCH_RESULTS.md) to find the structured dossier,
later analyses and live queue. Keep plain-language conclusions, detailed evidence,
limitations and future comparisons in separate, cross-linked documents. Preserve
raw artifacts and distinguish fixed-cutoff syntheses from changing execution status.
