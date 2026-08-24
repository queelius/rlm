# OOLONG Bootstrap Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move OOLONG experiment orchestration into `rlm-bootstrap` without losing
the current runner's identities, provenance, failure semantics, or resumability,
while leaving `rlm` as a benchmark-agnostic execution harness.

**Architecture:** Preserve runner schema v1 as a read/replay compatibility
contract, establish a small public RLM result/attestation/trace boundary, and
then transplant OOLONG data, prompting, verification, pairing, and scheduling
into benchmark-specific `rlm-bootstrap` modules. Keep the original runner until
old/new parity and one live smoke test pass; remove it only in a separate cutover
change.

**Tech Stack:** Python 3.10+, frozen dataclasses and enums, strict JSON, PyArrow,
pytest/pytest-cov, Ruff, Hatchling/uv, OpenAI Responses-compatible transports.

**Specs:**

- `docs/superpowers/specs/2026-08-23-rlm-kernel-and-experiment-contract-design.md`
- `../rlm-bootstrap/rlm_harness_integration_contract.md`
- `../rlm-bootstrap/rlm_self_training_experiment_handoff.md`

## Global Constraints

- Do not modify or remove `benchmarks/oolong.py` until Task 8's parity gate passes.
- `rlm` owns execution, typed ABI, limits, failures, traces, fingerprints, and
  explicit `run_direct`; it must not acquire datasets, verifiers, rewards,
  training, or experiment scheduling.
- `rlm-bootstrap` owns benchmark prompts, datasets, splits, paired conditions,
  rewards, checkpoints, exporters, training, and reports.
- Copy required OOLONG code and data specifications from `harness-rl`; never add
  `harness-rl` as an `rlm` or `rlm-bootstrap` runtime dependency.
- Do not add `codex-api` imports, configuration, or process management.
- Preserve strict failure separation: infrastructure/provenance/trace failures
  are never converted into empty answers or ordinary verifier scores.
- Keep v1 prompt text, IDs, and checkpoint readers immutable. Any prompt,
  verifier, selection, or artifact-layout correction receives a v2 identity.
- Keep raw datasets, rollout artifacts, traces, checkpoints, and reports outside
  Git. Commit only typed specifications, code, synthetic fixtures, and configs.

---

## Audited Current State

### Dependency proof

- `src/rlm/`, `pyproject.toml`, and `uv.lock` contain no reference to OOLONG,
  `harness-rl`, or PyArrow.
- `benchmarks/oolong.py` depends on RLM in only three places:
  `validate_controller_options`, strict-JSON helpers, and
  `validate_terminal_response`.
- The wheel contains only `rlm/**` and distribution metadata. The current source
  distribution also includes `benchmarks/oolong.py`, tests, and internal
  `.superpowers/sdd` artifacts; Task 8 closes that packaging leak.
- The local RLM checkout has two commits on `main` but no remote. The local
  `../rlm-bootstrap` directory is not yet a Git repository, and neither
  `queelius/rlm` nor `queelius/rlm-bootstrap` currently exists on GitHub. Connect
  those repositories before executing cross-repository commit steps; no remote
  action is part of this audit.

### Migration inventory

| Source | Preserve or replace |
|---|---|
| `benchmarks/oolong.py` | Preserve schema-v1 prompts, pair/episode/run identities, seven rollout failure kinds, signed records, signed checkpoints, resume behavior, summaries, and CLI semantics. |
| `tests/test_benchmark.py` | Move all 29 collected cases; add missing live-transport, main-path, trace-integrity, and pinned-data coverage. |
| `tests/fixtures/trace-schema-v1.json` | Keep as the RLM trace golden; publish a machine-readable copy with the runtime. |
| `harness_rl/oolong.py` | Migrate the pinned dataset specification, checksum verification, atomic download, Arrow row schema, deterministic selection, and verifier behavior. Do not migrate its legacy RLM/client runtime. |
| `harness_rl/tests/test_oolong.py` | Preserve selection, label/comparison/numeric scoring, direct-prompt, token-usage, and truncation cases. |
| OOLONG Parquet + sidecar | Keep outside Git; pin repository revisions, 5,901,318-byte size, SHA-256 `8cdd8ef5a01320d924271d7c9e0b1bb73b8198a53273f3f568e1e2c8750b8ee3`, and 650-row count. |

The pinned file has 13 context lengths, 50 rows at each length, and only two
unique context strings per length. It is useful for long-context stress testing,
but it is not by itself a credible broad-generalization training corpus.

### Known v1 defects to preserve for replay, not perpetuate

- Both v1 prompts ask for one aggregate answer and also exactly N per-question
  labels. A live Qwen3.5 run exhausted output on this contradiction and returned
  no answer text. Correct wording must use new prompt-profile IDs.
- The legacy scorer keeps only the first value in a multi-answer gold list.
- Current docs claim context-group train/dev/test splits, but the runner performs
  selection only; no split is implemented.
- The JSON checkpoint repeats full long contexts and complete responses for each
  arm and rewrites the whole file after every result. Preserve v1 import/export,
  then use content-addressed v2 artifacts for scale.
- Benchmark unit coverage is 78%; live HTTP error mapping and the real CLI/data
  assembly path are the largest uncovered regions.

### Smallest stable RLM boundary

The external runner should consume only:

```python
RLM.run(request: Mapping[str, Any]) -> RunResult
RLM.run_direct(request: Mapping[str, Any]) -> RunResult
RunResult.response
RunResult.run_id
RunResult.stop_reason
RunResult.usage
RunResult.turns
RunResult.duration_seconds
RunResult.controller_identity
RunResult.harness_fingerprint
RunResult.trace_directory
```

For HTTP compatibility it also needs a typed round-trip for the existing
`X-RLM-*` attestation headers. For trajectory ingestion it needs a read-only
loader that validates `manifest.json` and `trace.jsonl` against trace schema v1.
The runner must never import the engine loop, executor, trace writer, recovery
implementation, or private canonical-JSON functions.

---

### Task 1: Freeze OOLONG Runner v1 Compatibility

**Files:**

- Create: `tests/fixtures/oolong-runner-v1/comparison.json`
- Create: `tests/fixtures/oolong-runner-v1/checkpoint.json`
- Modify: `tests/test_benchmark.py`

**Interfaces:**

- Consumes: current `BenchmarkComparison`, `BenchmarkRunSpec`, `_signed_record`,
  and checkpoint serialization.
- Produces: immutable golden artifacts that both repositories must read and
  reproduce before cutover.

- [ ] **Step 1: Add exact identity assertions for the synthetic comparison**

```python
assert value.pair_id() == "de7550607f282095a078ca64d129600d3ddf38017801ef5b7966a7cbe6434b76"
assert episode_id(value.pair_id(), BenchmarkCondition.RLM) == (
    "22f8cdbe556aceb4a3bdffdfc00272c25bbbbb4cbd0dd1d5a8be03e34fdbc312"
)
assert episode_id(value.pair_id(), BenchmarkCondition.DIRECT) == (
    "a48c56a7c1dc1dfc863229f054e3d5e88789596101f17d8846f79d26a1246d37"
)
assert value.prompts.rlm.sha256 == (
    "5c45be8256e927b395b4b034cf9f57ba8c708cf4d70a0037ba41350a7815d8de"
)
assert value.prompts.direct.sha256 == (
    "43f6d9fd91076a13eb633e63fac382908c8124c03cc3aa6af5eee3b452afc8cc"
)
```

- [ ] **Step 2: Write deterministic golden comparison and checkpoint fixtures**

Use the existing synthetic `comparison()` and fixed `BenchmarkCall` values.
Normalize only `elapsed_seconds`; do not omit response, headers, prompt text,
failure, usage, or record/checkpoint SHA-256 fields.

- [ ] **Step 3: Prove the fixtures round-trip and reject one-field drift**

Run:

```bash
uv run pytest tests/test_benchmark.py -q
```

Expected: all 29 tests plus the new golden tests pass.

- [ ] **Step 4: Commit the compatibility boundary**

```bash
git add tests/test_benchmark.py tests/fixtures/oolong-runner-v1
git commit -m "test: freeze OOLONG runner v1 artifacts"
```

### Task 2: Publish the Narrow RLM Consumer Contract

**Files:**

- Create: `src/rlm/attestation.py`
- Create: `src/rlm/trace_reader.py`
- Create: `src/rlm/schemas/trace-v1.json`
- Create: `tests/test_attestation.py`
- Create: `tests/test_trace_reader.py`
- Modify: `src/rlm/response.py`
- Modify: `src/rlm/protocol.py`
- Modify: `src/rlm/executor.py`
- Modify: `src/rlm/server.py`
- Modify: `src/rlm/types.py`
- Modify: `src/rlm/__init__.py`
- Modify: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: `RunResult`, `ControllerRunIdentity`, current `X-RLM-*` headers,
  `TRACE_SCHEMA_VERSION`, `TraceEventKind`, and trace payload schemas.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class RunAttestation:
    run_id: str
    stop_reason: str
    turns: int
    duration_seconds: float
    usage: TokenUsage
    controller_identity: ControllerRunIdentity
    harness_fingerprint: str

    @classmethod
    def from_result(cls, result: RunResult) -> RunAttestation: ...

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> RunAttestation: ...

    def to_headers(self) -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class TraceArtifact:
    directory: Path
    manifest: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    manifest_sha256: str
    jsonl_sha256: str


def load_trace(directory: Path, *, expected_run_id: str | None = None) -> TraceArtifact: ...
def response_output_text(response: Mapping[str, Any]) -> str: ...
```

Both public dataclasses must defensively copy mutable `TokenUsage`, manifest,
event, and response values so frozen wrappers cannot be mutated indirectly.

- [ ] **Step 1: Test attestation result/header round-trips and malformed headers**

Require case-insensitive input headers, exact current header names, strict
nonnegative integers, finite nonnegative duration, a positive model-call count,
valid SHA-256 values, and no ignored missing field. Preserve the current
six-decimal duration representation; identity and usage round-trip exactly while
duration round-trips to that documented precision.

- [ ] **Step 2: Implement `RunAttestation` and derive server headers from it**

`server._run_headers` must become a one-line delegation to
`RunAttestation.from_result(result).to_headers()`. Do not maintain a second
header-name table in the server or experiment runner.

- [ ] **Step 3: Test public response validation and text extraction**

Move the permissive standard Responses text extraction out of the private
controller protocol. Preserve unknown output items and empty text; whether an
empty answer is scoreable remains an experiment decision.

- [ ] **Step 4: Implement and export `response_output_text`**

Update executor imports to use the public helper. Export
`response_output_text`, `validate_response_envelope`, and
`validate_terminal_response` from `rlm`.

- [ ] **Step 5: Test post-hoc trace integrity**

Cover exact schema and payload fields, strict JSON, run-ID agreement, unique
ordered event IDs, backward-only causal parents, exactly one final event, and
manifest/final-event agreement. Every model request must have a response or
failure with the same `call_id`; permit a response followed by a failure only
for the existing post-response validation path. Include completed, repaired,
recursive, failed, truncated, tampered, and missing-file fixtures.

- [ ] **Step 6: Implement `load_trace` without exposing the writer**

Read the two files once, validate against the packaged v1 contract, deep-own the
returned JSON values, and hash the exact bytes separately. Never repair,
reinterpret, or skip an invalid event.

- [ ] **Step 7: Document `run_direct`, attestation, and trace consumption**

State that Python orchestration is preferred for the first A100 experiment
because `RunResult.trace_directory` is an unambiguous artifact reference. HTTP
remains supported through typed attestation, but no absolute server filesystem
path is exposed in a header.

- [ ] **Step 8: Run the core contract suite and commit**

```bash
uv run pytest tests/test_attestation.py tests/test_trace_reader.py \
  tests/test_server.py tests/test_response.py tests/test_trace.py -q
uv run ruff check src tests
uv run ruff format --check src tests
git add src tests README.md pyproject.toml
git commit -m "feat: publish rollout attestation and trace contract"
```

### Task 3: Scaffold `rlm-bootstrap` Without Training Infrastructure

**Files:**

- Create: `../rlm-bootstrap/pyproject.toml`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/__init__.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/canonical_json.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/rollouts.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/checkpoints.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/__init__.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/__init__.py`
- Create: `../rlm-bootstrap/tests/test_package.py`

**Interfaces:**

- Consumes: released or sibling `rlm==0.1.0` public API only.
- Produces: typed rollout and artifact primitives used by OOLONG now and later
  exporters; no trainer, reward framework, or generic workflow engine.

- [ ] **Step 0: Establish the bootstrap repository before moving code**

Initialize `../rlm-bootstrap` on `main`, commit its existing `AGENTS.md` and two
contract documents, then create/connect the intended GitHub remote. Perform this
as an explicit repository-release operation before the first transplanted file;
do not bury it inside a copy script.

```bash
cd ../rlm-bootstrap
git init -b main
git add AGENTS.md rlm_harness_integration_contract.md \
  rlm_self_training_experiment_handoff.md
git commit -m "docs: define bootstrap experiment contracts"
gh repo create queelius/rlm-bootstrap --public --source=. --remote=origin --push
```

- [ ] **Step 1: Add package metadata and explicit dependencies**

Use Python `>=3.10`, `rlm==0.1.0`, and `pyarrow>=18`; put pytest, pytest-cov,
and Ruff in a dev extra. During sibling development, use uv's path source for
`../rlm`, but record the exact RLM Git revision in each experiment manifest.

- [ ] **Step 2: Add strict canonical JSON helpers with golden hash tests**

Reject duplicate keys, NaN/infinity, cycles, non-string mapping keys, and
non-JSON objects. Do not import `rlm.json` or silently use `repr`.

- [ ] **Step 3: Define typed rollout outcomes**

```python
class RolloutCondition(str, Enum):
    RLM = "rlm"
    DIRECT = "direct"


class RolloutFailureKind(str, Enum):
    HTTP = "http"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    DECODE = "decode"
    PROTOCOL = "protocol"
    PROVENANCE = "provenance"
    BUDGET = "budget"


@dataclass(frozen=True, slots=True)
class RolloutFailure:
    kind: RolloutFailureKind
    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class RolloutOutcome:
    condition: RolloutCondition
    response: dict[str, Any] | None
    attestation: RunAttestation | None
    trace: TraceArtifact | None
    failure: RolloutFailure | None
```

Enforce exactly one of a successful response or failure. An RLM success requires
attestation and trace. A raw-provider direct success does not require RLM
attestation; a direct run made through `RLM.run_direct` retains its run ID and
trace as additional provenance. Deep-copy any retained mutable response or
artifact metadata in `__post_init__`.

- [ ] **Step 4: Add the package smoke test and commit**

```bash
cd ../rlm-bootstrap
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
ruff format --check .
git add pyproject.toml src tests
git commit -m "feat: scaffold typed rollout artifacts"
```

### Task 4: Transplant the Pinned OOLONG Dataset and Verifier

**Files:**

- Create: `../rlm-bootstrap/configs/benchmarks/oolong-trec-coarse-v1.json`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/data.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/verifier.py`
- Create: `../rlm-bootstrap/tests/benchmarks/test_oolong_data.py`
- Create: `../rlm-bootstrap/tests/benchmarks/test_oolong_verifier.py`

**Interfaces:**

- Consumes: pinned Parquet schema and behavior from `harness_rl.oolong`.
- Produces: `DatasetSpec`, `OolongExample`, `OolongGold`, `OolongScore`,
  `load_examples`, `select_examples`, and versioned verifier implementations.

- [ ] **Step 1: Encode and validate the complete dataset identity**

The config must contain repository `lsteno/RLM-Evals`, revision
`a6aea6d06da9f08d701038b64195049cf71e1997`, source repository
`oolongbench/oolong-synth`, source revision
`f0d59eaf0febf130664cfceb710436c8e3216b2b`, relative Parquet path,
byte count, SHA-256, row count, and all 13 expected Arrow columns.

- [ ] **Step 2: Port atomic download, checksum, Arrow loading, and row types**

Preserve every source column, including `context_window_id`, `input_subset`,
`task`, `task_group`, `answer_type`, and `num_labels`. Fail before selection if
the checksum, size, schema, or row count differs.

- [ ] **Step 3: Preserve v1 selection and verifier behavior under explicit IDs**

Name them `oolong_selection_v1` and `oolong_verifier_v1`. Their golden tests
must match `harness-rl` for label, comparison, numeric partial credit, seeded
selection, multi-answer-first behavior, and malformed output.

- [ ] **Step 4: Add a structured v2 gold parser and verifier**

Parse the dataset's string-encoded list with `ast.literal_eval`, then require a
non-empty list whose member types agree with a typed `OolongAnswerType` enum.
Parse outputs by answer type and exact requested prefix/enum/integer grammar;
do not use regex to parse structured values. Treat every value in a gold list as
accepted and give numeric partial credit only for the numeric answer type.

- [ ] **Step 5: Derive verifier provenance from its implementation module**

Keep parsing and scoring helpers in `verifier.py` and hash that module's exact
bytes, deriving its path from the callable rather than maintaining a file map.
Record package version, semantic verifier ID, module SHA-256, Git revision, and
dependency-lock SHA-256. Moving behavior into another module requires a new
verifier version and an updated provenance strategy rather than an untracked
helper dependency.

- [ ] **Step 6: Run unit and pinned-data integration tests and commit**

```bash
python -m pytest tests/benchmarks/test_oolong_data.py \
  tests/benchmarks/test_oolong_verifier.py -q
git add configs src/rlm_bootstrap/benchmarks/oolong tests/benchmarks
git commit -m "feat: migrate pinned OOLONG data and verifier"
```

### Task 5: Transplant Identities, Records, and Resumption

**Files:**

- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/schema.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/artifacts.py`
- Create: `../rlm-bootstrap/tests/benchmarks/test_oolong_schema.py`
- Create: `../rlm-bootstrap/tests/benchmarks/test_oolong_resume.py`
- Copy: `tests/fixtures/oolong-runner-v1/**` to
  `../rlm-bootstrap/tests/fixtures/oolong-runner-v1/**`

**Interfaces:**

- Consumes: Task 1 goldens and Task 4 examples/verifier identity.
- Produces: `ComparisonSpecV1`, `ComparisonSpecV2`, `EpisodeRecord`,
  `ExperimentManifest`, `ArtifactStore`, and lossless v1 readers/exporters.

- [ ] **Step 1: Read v1 goldens without changing any identity**

Assert the exact pair, episode, prompt, options, run-spec, record, and checkpoint
hashes from Task 1. Reject unknown schema versions rather than guessing.

- [ ] **Step 2: Define v2 identity from one immutable comparison**

Bind dataset/source/split/example identity, model and checkpoint digest,
tokenizer revision, both exact prompt specs, harness fingerprint and RLM source
revision, seed, repetition, strict sampling, declared budget, verifier identity,
environment/lock digest, and selection version. Both arms share one `pair_id`;
`episode_id` additionally binds the typed arm.

- [ ] **Step 3: Implement content-addressed artifacts**

Use this layout under a caller-supplied artifact root:

```text
run.json
contexts/<context_sha256>.txt
examples/<example_id>.json
episodes/<episode_id>.json
traces/<run_id>/manifest.json
traces/<run_id>/trace.jsonl
```

Write each object through a same-directory temporary file, fsync the file,
replace atomically, and fsync the directory. Episode records reference immutable
context/example objects; the v1 exporter materializes the old full record
without losing any field.

- [ ] **Step 4: Resume only absent valid episode IDs**

On startup, validate `run.json`, every referenced object hash, every episode's
pair/condition/prompt/provenance, and duplicate IDs before any model call. A
corrupt artifact aborts the run; it is never overwritten or treated as missing.

- [ ] **Step 5: Port all checkpoint tamper and drift tests**

Cover prompt, verifier, reference, selection, dataset revision, model revision,
budget, record deletion/reordering, duplicate arms, schema version, context
blob, and trace digest drift.

- [ ] **Step 6: Run coverage and commit**

```bash
python -m pytest tests/benchmarks/test_oolong_schema.py \
  tests/benchmarks/test_oolong_resume.py \
  --cov=rlm_bootstrap.benchmarks.oolong --cov-report=term-missing -q
git add src/rlm_bootstrap tests
git commit -m "feat: preserve OOLONG identities and resumption"
```

### Task 6: Version Correct OOLONG Conditions and Splits

**Files:**

- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/prompts.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/selection.py`
- Create: `../rlm-bootstrap/tests/benchmarks/test_oolong_prompts.py`
- Create: `../rlm-bootstrap/tests/benchmarks/test_oolong_selection.py`

**Interfaces:**

- Consumes: typed examples and comparison identities.
- Produces: immutable v1/v2 `PromptSpec` values and deterministic
  `DatasetSplit` assignments.

- [ ] **Step 1: Preserve all three v1 prompt profiles byte-for-byte**

Keep `rlm_minimal_v1`, `rlm_oracle_v1`, and `direct_v1` only for replay and
equivalence. Their SHA-256 values remain identity-bearing.

- [ ] **Step 2: Add non-contradictory v2 profiles**

All v2 prompts request exactly the aggregate answer and explicitly prohibit a
per-question label list as the final response. `direct_v2` provides the dataset
question without ABI or retry guidance; `rlm_minimal_v2` adds only taxonomy and
answer-format context; `rlm_oracle_v2` may prescribe typed batching/retry code
but still submits only the aggregate answer.

- [ ] **Step 3: Test paired-request invariants**

Assert equal context, task, model identity, seed, sampling, repetition, split,
and declared budget. Assert distinct typed prompts, no RLM-only scaffold in the
direct arm, and a changed pair ID whenever either exact prompt changes.

- [ ] **Step 4: Implement deterministic group splitting before selection**

Group exact duplicate contexts by context SHA-256, assign groups by a versioned
SHA-256 rank of `(dataset_revision, split_seed, group_id)`, and prove no digest
crosses train/development/test. Report rows and unique groups per split and
refuse requested stratification when there are too few groups; do not silently
fall back to row splitting.

- [ ] **Step 5: Add the OOLONG sufficiency warning as structured metadata**

Record that the pinned subset has 26 exact context digests total and two per
context length. Mark it `stress_only` unless an experiment config supplies a
separate dataset with enough independent groups for the declared split.

- [ ] **Step 6: Run tests and commit**

```bash
python -m pytest tests/benchmarks/test_oolong_prompts.py \
  tests/benchmarks/test_oolong_selection.py -q
git add src/rlm_bootstrap/benchmarks/oolong tests/benchmarks
git commit -m "feat: version OOLONG conditions and context splits"
```

### Task 7: Add External Paired Rollout Orchestration

**Files:**

- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/runner.py`
- Create: `../rlm-bootstrap/src/rlm_bootstrap/benchmarks/oolong/cli.py`
- Create: `../rlm-bootstrap/tests/benchmarks/test_oolong_runner.py`
- Create: `../rlm-bootstrap/tests/integration/test_oolong_end_to_end.py`
- Modify: `../rlm-bootstrap/pyproject.toml`

**Interfaces:**

- Consumes: public `RLM.run`, `RLM.run_direct`, `RunAttestation`, `load_trace`,
  typed OOLONG comparison specs, verifier, and artifact store.
- Produces: resumable paired records and summaries; no training behavior.

- [ ] **Step 1: Implement the preferred in-process RLM adapter**

Create one configured `RLM` around the Responses backend. Dispatch the RLM arm
with `run` and the direct arm with `run_direct`; retain each arm's own response,
failure, usage, run ID, and trace. Never call the direct arm as recovery for an
RLM failure.

- [ ] **Step 2: Implement a compatibility HTTP adapter**

Use standard Responses JSON plus `RunAttestation.from_headers` for the RLM arm.
Keep HTTP/transport/timeout/decode/protocol/provenance/budget as the seven v1
episode failure kinds. Dataset, checkpoint, verifier, and trace corruption are
fatal experiment errors, not zero-reward episodes.

- [ ] **Step 3: Validate every successful RLM trace before scoring**

Require run-ID, controller model/options, harness fingerprint, aggregate usage,
zero unreported calls, schema version, terminal response, and trace/manifest
agreement. Store both exact trace-file digests in the episode record.

- [ ] **Step 4: Add explicit target-eligibility metadata**

Record each controller response's causal outcome. For initial SFT, default to
masking controller actions that caused a typed repair while retaining them as
later context; keep `all_controller` as an explicit ablation. This policy lives
in `rlm-bootstrap` and never changes the canonical RLM trace.

- [ ] **Step 5: Port summary and resumption tests, then cover missing paths**

Add fake-server tests for every transport failure, malformed/empty successful
response, missing/malformed attestation, direct usage failure, over-budget run,
score exception, trace tampering, interrupt after either arm, and exact resume.

- [ ] **Step 6: Add a synthetic end-to-end test**

Run one successful and one repaired scripted RLM pair through artifact writing,
trace validation, scoring, restart, and summary. Assert restart performs zero
model calls and that no observation/verifier/reward data entered a model request.

- [ ] **Step 7: Add the CLI and commit**

Expose thin `rlm-bootstrap oolong download`, `verify`, `run`, `resume`, and
`summarize` commands. Every run command requires a versioned config and explicit
artifact root; no sibling dataset path is a default.

```bash
python -m pytest tests/benchmarks/test_oolong_runner.py \
  tests/integration/test_oolong_end_to_end.py -q
git add pyproject.toml src/rlm_bootstrap tests
git commit -m "feat: run resumable paired OOLONG experiments"
```

### Task 8: Prove Parity, Cut Over, and Clean Release Artifacts

**Files:**

- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `RESEARCH.md`
- Modify: `pyproject.toml`
- Delete only after parity: `benchmarks/oolong.py`
- Delete only after parity: `tests/test_benchmark.py`
- Modify: `../rlm-bootstrap/README.md`
- Create: `tests/test_distribution.py`

**Interfaces:**

- Consumes: old/new v1 golden behavior and the complete bootstrap runner.
- Produces: a benchmark-free RLM repository and a self-contained OOLONG
  experiment in `rlm-bootstrap`.

- [ ] **Step 1: Run old/new v1 differential tests**

Feed both implementations identical synthetic comparisons and captured call
results. Require exact comparison payloads, IDs, prompts, records, checkpoint
digests, resume calls, failures, scores, and summaries.

- [ ] **Step 2: Run pinned-data equivalence**

Require identical 650-row loading, checksum/schema validation, selected source
IDs for fixed seeds/context lengths, and v1 verifier outputs. Then separately
test v2 prompts, multi-gold scoring, split grouping, and content-addressed resume.

- [ ] **Step 3: Run one live v2 smoke test**

Run one OOLONG pair against Qwen3.5 with tracing. Require a non-empty aggregate
answer or an explicit typed episode failure, a valid trace, exact provenance,
and a restart that performs no completed-arm call. Do not require a positive
score as a migration gate.

- [ ] **Step 4: Remove the old runner in a separate RLM commit**

Only after Steps 1-3 pass, delete the two old files and update RLM docs to point
to `rlm-bootstrap`. Do not move any bootstrap implementation back into `rlm`.

- [ ] **Step 5: Restrict source-distribution contents**

Configure Hatch's sdist target so releases include runtime sources, license,
README, and public schemas/docs, but exclude `benchmarks`, tests, `.superpowers`,
`runs`, `dist`, and handoff scratch artifacts. `tests/test_distribution.py`
must inspect both wheel and sdist members and reject OOLONG, `harness_rl`,
`.superpowers/sdd`, and `codex` paths.

- [ ] **Step 6: Run final verification in both repositories**

```bash
cd ../rlm
uv run pytest --cov=rlm --cov-report=term-missing -q
uv run ruff check .
uv run ruff format --check .
uv build

cd ../rlm-bootstrap
python -m pytest --cov=rlm_bootstrap --cov-report=term-missing -q
ruff check .
ruff format --check .
```

Expected: zero failures; neither built RLM artifact contains OOLONG,
`harness-rl`, Codex-specific code, or internal SDD snapshots.

- [ ] **Step 7: Commit the cutover independently in each repository**

```bash
git add README.md DESIGN.md RESEARCH.md pyproject.toml tests/test_distribution.py \
  benchmarks/oolong.py tests/test_benchmark.py
git commit -m "refactor: move OOLONG experiments out of RLM"
cd ../rlm-bootstrap
git add README.md
git commit -m "docs: document OOLONG experiment workflow"
```

## Explicitly Deferred

- LoRA/QLoRA, SFT exporters, held-out statistics, RL, and harness optimization
  remain later `rlm-bootstrap` plans.
- No Chat Completions compatibility, Codex adapter, trace-serving HTTP API,
  dataset registry, generic workflow engine, or RLM core-loop rewrite is needed
  for this extraction.
- If experiments must run the RLM server on a different host from artifact
  storage, design a separate authenticated trace-artifact transport. Do not leak
  absolute filesystem paths through HTTP headers.
