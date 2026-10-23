# rlm

`rlm` is a small, request-preserving Recursive Language Model (RLM) runtime for the
non-streaming OpenAI **Responses** API, and the research codebase built on it. A private
controller writes exactly one Python cell per turn in a persistent IPython process; that cell can
inspect the unchanged public request, call helper models through a versioned ABI, use a prebound
library of routines, and submit a final Responses result.

**Documentation, results and lab notebook:** https://queelius.github.io/rlm/

## What the research has found (October 2026)

The question is how a *small* controller (Qwen3-4B) should learn to decompose long-context tasks,
and in particular when to decompose at all. Short version:

- **Teaching then RL works for the routine.** Imitating a scripted expert and then RL with a
  verifiable reward lifts held-out OOLONG-synth from 0.567 to 0.652 (direct answering: 0.208).
- **RL does not hold the route choice.** Every policy-gradient variant we tried collapses to a
  single route: cost penalties, KL to the warm start, stratified sampling, short-task weighting
  and a balance term. Starting from a model that already routes correctly does not help either.
- **Put the routine in a library instead.** With the routines bound through the kernel's harness
  prelude, the *untrained* 4B controller matches every trained controller. On the 293-row held-out
  OOLONG set it scores 0.672 with no errors, equal to the scripted expert.
- **Put the route choice in a table.** Outcomes of past verified runs, keyed by a signature
  computed from the input (RAG-style retrieval), are rendered as a short note. The untrained
  controller follows it on every task and gets the best accuracy at the lowest cost.
- **Grow the library by verification.** A proposer model writes a routine for a new question
  form, and it is accepted only if it verifies on teaching data. Four held-out forms went from 0
  to 22 to 24 of 24 with no training. Verification kept out a buggy 4B routine that a stronger
  (8B) proposer then fixed.

Added on 22 and 23 October (new families: OpenAI MRCR, MuSiQue in long haystacks, LongBench-v2, and 37 generated
families for library discovery):

- **Libraries can be bootstrapped, by a capable enough proposer.** A 4-bit Qwen3-8B with thinking wrote a
  verified MRCR library from three examples (2 of 5 seeds) that scores 1.000 held out up to 1M tokens; a 0.6B
  caller that never saw MRCR calls it correctly. A 4-bit 14B does no better (2 of 5). No 4B proposer found it,
  however trained.
- **Training the proposer works on families like its training ones.** SFT on the 8B's verified code, then RL
  with the verified teaching score as reward, lifts a 4B from 9 to 30 of 74 held-out family attempts
  (replicated; RL without SFT does not help), and to 42 with more varied teacher data, level with the 8B teacher on
  code-solvable families. Three times the helper-family teacher data brings it to the 8B on families that need the
  helper's judgment, at a small, non-significant cost on code; more than that lowers held-out code success further
  without helping helper families (113 pairs: 29 against 42 of 74, p 0.007). It does not transfer to MRCR.
- **Where answers are exact, bootstrapping needs three labels.** A candidate that runs cleanly and reproduces the
  three examples in its prompt is almost always correct (127 of 129, all generalizing to fresh, longer inputs).
  Pre-registered on 30 unseen families, the trained 4B with eight samples and three labels bootstrapped as many
  correct libraries as the thinking 8B with thirty (12 against 13) at 7 percent of the tokens. Agreement among
  candidates is no substitute, and noisy families (MuSiQue, OOLONG) need tens of labels.
- **Boundary: broad families.** No proposer bootstraps OOLONG, the study's own family, as one library (eleven
  question templates from thirty examples); narrowed to one task group, the 14B writes a counting library level
  with the hand-written one, but not the timeline or user groups.
- **Choose evidence, do not summarize it.** Two-round retrieval is the best and cheapest MuSiQue library (also
  at 512k tokens); on all 503 LongBench-v2 questions a retrieve-then-read library beats direct reading, 0.43 against
  0.36, at a quarter of the tokens, also through the 0.6B caller. Bigger readers, map-reduce notes and recursion do not help.
- **One line of documentation decides whether an untrained caller uses a library** (0.18 to 0.95 on MRCR);
  a distilled caller does not need it.

Evidence for each claim, with run locations and caveats, is in the
[results ledger](docs/RESULTS.md); the narrative is in the
[lab notebook](docs/notebook/) and the plan in [docs/research-plans/](docs/research-plans/).
The experiment code is in `experiments/selective_recursion_20261019/`; released models and data
are listed in [docs/hf/RELEASE_LIST.md](docs/hf/RELEASE_LIST.md).

## The runtime

The public surface is deliberately narrow:

- `GET /v1/models` is passed through to the configured upstream.
- `POST /v1/responses` runs the RLM loop.
- Chat Completions, streaming, and background Responses are unsupported.
- `RLM.run_direct(request)` is the explicit, attested direct-model baseline. The RLM loop
  never falls back to it, retries a request, or substitutes an alternate answer.

## Install

```bash
uv sync --extra dev
uv run pytest
```

For a GPU-hosted Hugging Face workflow, including vLLM serving, an A100 smoke test, and a
trace-to-LoRA iteration loop, see [A100 Experiment Quickstart](docs/A100_EXPERIMENTS.md).


## Library usage

Configuration is nested. `HarnessSpec`, limits, execution, and trace structure
are frozen; `ControllerConfig.options` is intentionally mutable between runs.
Each run takes one owned deep snapshot before its first model call, so later
caller mutations cannot alter that run. `HarnessSpec` defines model-visible,
fingerprinted behavior—including the typed first-turn bootstrap contract—while
controller selection remains a separate experiment control.

```python
from rlm import (
    ControllerConfig,
    ExecutionConfig,
    HarnessSpec,
    RunAttestation,
    OpenAIEndpoint,
    RLM,
    RLMConfig,
    RunLimits,
    TraceConfig,
    load_trace,
    response_output_text,
)
from rlm.prompts import default_harness_spec

harness: HarnessSpec = default_harness_spec()
config = RLMConfig(
    harness=harness,
    limits=RunLimits(max_turns=12, deadline_seconds=600, max_depth=0),
    controller=ControllerConfig(model="controller-model", options={"temperature": 0}),
    execution=ExecutionConfig(working_directory=None),
    tracing=TraceConfig(enabled=True, directory="runs", markdown=True),
)
rlm = RLM(OpenAIEndpoint(base_url="https://api.openai.com/v1"), config=config)

request = {"model": "public-model", "input": "Explain recursion in one paragraph."}
run = rlm.run(request)
assert run.response["object"] == "response"
assert response_output_text(run.response)
attestation = RunAttestation.from_result(run)
trace = load_trace(run.trace_directory, expected_run_id=attestation.run_id)

# An attested evaluation baseline, never an automatic recovery path:
direct_run = rlm.run_direct(request)
```

`RLM.complete(request)` returns only the completed Responses object.
`RLM.run(request)` additionally returns the run ID, aggregate reported usage,
turn count, controller model/options identity, harness fingerprint, duration,
and optional trace directory.

Python orchestration is preferred for the first A100 experiment because
`RunResult.trace_directory` names its artifact unambiguously. `RunAttestation`
provides the same typed execution identity for in-process and HTTP consumers,
while `load_trace` verifies the manifest, JSONL bytes, schema, causal graph,
model-call outcomes, and terminal provenance before returning owned values.

## Kernel contract

The controller's first user item is a typed bootstrap schema-v2 envelope. It
declares the ABI-derived `request` binding, compact request metadata, and the
required final kinds: `text` plus `response` for ordinary requests, or only
`response` when a synthetic text response would lose request semantics. Caller
content is absent from the envelope; it is runtime metadata, never the task.
The controller determines the task from bound `request` by inspecting the
fields it needs or deliberately delegating the exact request with
`model_complete(request)`; neither path requires a separate host-mandated
inspection turn. Each controller response must contain one assistant
`output_text` item whose whole content is exactly one:

````text
```python
# one non-empty Python cell
FINAL_TEXT("answer")
```
````

No prose, extra fences, extra messages, or empty cells are accepted. Each
nonterminal turn receives a strict-JSON `rlm.controller_observation` schema-v2
item with explicit request binding, absent/required submission state, and a
bounded nested execution record. `execution.status: "ok"` means only that the
cell ran; `submission.status: "absent"` means no valid final was accepted. An
accepted final terminates without another observation. Printed or displayed
values are private and do not submit an answer. Configured observation budgets
must be at least 512 characters so every built-in repair identity can be shown.
The latest controller response retains its reasoning and message output items,
followed by this runtime item; IPython itself is durable working memory.

The versioned environment ABI provides `model_complete`, `ask`, their batch
variants, `FINAL_TEXT`, `FINAL_RESPONSE`, and `SHOW_VARS`; recursive helpers
are present only when the depth limit permits them. A harness may also carry a
`PreludeSpec`: library source executed once before the first cell, whose exported
names are listed in the prompt by contract, hidden from `SHOW_VARS`, restored
before every cell, and included in the harness fingerprint (see `DESIGN.md`). `FINAL_TEXT` synthesizes a
minimal completed text response only when that can faithfully satisfy the
public request. Its argument must already be a string; invalid values become a
typed repair observation and are never coerced. `FINAL_RESPONSE` accepts an
existing, fully validated Responses object.

`ask_batch` returns an `AskBatchResult`, with successful texts aligned to the
input indexes and failures retained separately. A controller can inspect
`failed_indexes` and call `require_texts()` when complete text is required;
there is no global pending-error flag and no automatic recomputation.

Malformed controller output, Python exceptions, empty strict-text helper
results, and invalid final submissions are typed controller faults. The
configured recovery policy returns a bounded observation and consumes a normal
turn (or explicitly aborts). Invalid public requests, malformed/failed
upstream responses, backend deadline failures, exhausted limits, host or
executor failures, and enabled-trace failures are fatal. Captured `stderr` is
ordinary output, not a failure signal.

Every backend call receives the remaining timeout and a typed call context
(`controller`, `public`, or `subcall`). Backends must honor that timeout; a
custom backend that blocks beyond it violates the contract. The shared ledger
also rejects calls that return after the run deadline.

## Server usage

```bash
uv run rlm serve \
  --upstream-base-url https://api.openai.com/v1 \
  --controller-model controller-model \
  --controller-option temperature=0 \
  --trace-dir runs
```

Controller option values are strict JSON. They cannot replace runtime-owned
model, instructions, or input fields. Successful proxy responses carry
`X-RLM-*` run metadata, including controller identity, harness fingerprint,
and aggregate usage; their body remains a normal Responses object. HTTP clients
can decode those case-insensitive headers with
`RunAttestation.from_headers(response.headers)`. No header exposes an absolute
path on the server filesystem; artifact transfer remains an explicit
orchestration responsibility.

## Traces and research use

With tracing enabled, JSONL is the canonical append-only trace. Markdown and
the manifest are derived views. The trace schema is strict and records the
exact public request, effective configuration, complete harness and its
fingerprint, exact rendered-prompt digests, ABI identity, raw model requests
and responses, typed call roles, causal IDs, observations, recovery decisions,
usage completeness, limits, timings, and a final response or typed failure.
Unknown trace values are rejected rather than serialized with `repr`.
Post-hoc consumers should call `load_trace(path, expected_run_id=...)` rather
than parsing individual lines. Corruption, truncation, run-ID disagreement,
causal errors, orphaned model calls, and manifest/final-event disagreement are
fatal `TraceError` failures; the reader never repairs or skips an event.

Task conditioning, datasets, verifiers, rewards, and benchmark recipes belong
outside `src/rlm/`. The optional `benchmarks/oolong.py` runner performs paired
RLM/direct evaluation: arm-specific prompt profiles are derived from one
immutable comparison, with identical task, context, model, seed, sampling, and
budget. It keeps transport/decode/provenance failures separate from verifier
scores, groups split data by context digest, records repetitions and complete
provenance, and audits RLM aggregate usage rather than a leaf response.

The original plan was fixed-harness evaluation, then self-SFT on successful
trajectories, then RL with verifiable rewards, and only then harness search. The
evidence above revised it: routines and route selection now live outside the
weights (prelude library, outcome statistics, verified growth), and training is
reserved for what verification cannot supply. Whatever optimizes the harness or
library never receives verifier answers for held-out splits, and cannot alter
budgets, trace semantics, or fatal-error handling.

## Security

The IPython executor is process-isolated but **not** a security sandbox.
Generated code has the file, network, and process permissions of the current
OS user. Run only trusted generated code and treat traces as sensitive.

## Related reproductions

Three paper reproductions done as training-pipeline exercises in October 2026 live in their own
repositories: [SPO on GSM8K](https://github.com/queelius/spo-gsm8k-reproduction) (audited,
58.19 percent versus the paper's 56.7) and [R1-style Dr. GRPO and CURL](https://github.com/queelius/rl-reproductions).
See `docs/PATH_MAP.json` for where their former folders went.
