# Research notes

RLM is an experimental kernel, not a benchmark, verifier, trainer, or general
agent framework. It keeps a large public request in persistent Python working
memory while a controller produces exact Python actions and may make bounded
model calls. The runtime deliberately preserves a small, inspectable
Responses-only contract so experimental changes are attributable.

## Experimental boundary

The kernel owns controller context, the ABI, typed recovery/fatal semantics,
deadlines and budgets, executor/IPC behavior, and canonical causal traces.
Datasets, task conditioning, splits, verifiers, rewards, training, and reports
remain external. Verifier answers and rewards never enter controller context.

Harness behavior is explicit and fingerprinted: frozen structural `HarnessSpec`
contains the prompt, exact rendered-prompt digests, typed first-turn bootstrap,
ABI identity, recovery, observation, and context settings. Bootstrap schema v2
tells the controller that caller content is absent, names the ABI-bound request,
and derives the allowed explicit final kinds from the request. Observation
schema v2 separates runtime feedback from caller turns and distinguishes cell
execution from final submission. Neither envelope copies task data into the
private conversation. The fixed global
`RunLimits` are not harness-search variables. This prevents an optimizer from
claiming reward by enlarging time, token, call, recursion, or concurrency
budgets, weakening tracing, or changing fatal behavior. Controller options may
change between runs, but each run owns a deep strict-JSON snapshot before
rollout; no option mutation can alter an in-flight run.

The controller protocol is intentionally narrow: one exact Python cell per
turn, a choice between inspecting needed request fields and deliberately
delegating the bound request, no mandatory separate inspection turn, persistent
IPython state, retained reasoning output, strict
`FINAL_TEXT`/`FINAL_RESPONSE` submissions, and typed observations. Controller
and Python faults can be repaired by a normal next turn; infrastructure,
validation, deadline, budget, executor, and trace faults are fatal. There is no
automatic retry, direct answer fallback, or conversion of a failure to an empty
response. Final text is also exact: the executor rejects non-string values for
self-correction instead of silently applying `str(...)`.

## Evidence and limitations

Recursive language model work motivates programmatic interaction with external
context, but recursion itself is not presumed beneficial. Fixed depth and
shared budgets make the non-recursive/controller-only baseline measurable.
Process isolation improves operational containment but is not a security
sandbox: generated code retains the current user's OS permissions.

The current benchmark implementation is `benchmarks/oolong.py` (runner schema
version 1), an optional stress runner rather than evidence of general
capability. Its small locally pinned Oolong subset can contain very few unique
long contexts, so row-level scores would overstate independence. It therefore
groups train/development/test splits by context digest, reports row and unique
context counts, uses repeated rollouts for stochastic conditions, and retains
HTTP, transport, timeout, decode, protocol, provenance, and budget failures
separately from verifier scores.

Each RLM-minimal or RLM-oracle condition has a paired direct-model baseline.
Both arms derive from one immutable comparison with the same task, context,
model, seed, sampling options, and declared budget. Prompts remain arm-specific:
the direct arm receives no RLM ABI or retry scaffold. Provenance includes exact
prompt text/profile/digest, dataset/source/revision, context digest, verifier
identity, model/checkpoint identity, harness fingerprint, repetition, pair ID,
and aggregate usage. A successful RLM arm must attest its controller model,
canonical option digest, and harness fingerprint; its token budget is audited
from aggregate run counters, not from a leaf response.

## Research sequence

1. Measure plain direct and fixed-harness RLM baselines on held-out,
   context-grouped conditions.
2. Collect verified successful fixed-harness trajectories and run self-SFT on
   controller actions, with a matched plain-model self-training control.
3. Apply later fixed-harness RL with verifiable rewards using fresh on-policy
   rollouts from both base and self-SFT checkpoints.
4. Freeze model weights and search only symbolic `HarnessSpec` candidates on
   paired development tasks under fixed budgets; then evaluate once on held-out
   test tasks.
5. Alternate fixed-harness model updates and fixed-model harness search only
   after the preceding controls are measured.

Trace exporters use exact model-visible requests as states and model-authored
reasoning/assistant content as targets. Environment output, observations,
verifier output, reward, and transport metadata remain context or provenance,
not SFT targets. On-policy RL additionally needs token IDs, action masks,
sampling metadata, and old-policy log probabilities; ordinary Responses
objects alone are not enough.
