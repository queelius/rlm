# RLM Implementation Design and Review Brief

## Purpose

Use this document to review and improve the existing RLM implementation without turning it into a large, opaque agent framework.

The target is a **small, reproducible, inspectable, and trainable agentic harness**. It should preserve the ordinary model interface while adding an ephemeral inner computation loop, Python state, recursive calls, and complete observable traces.

The implementation should prioritize:

- pedagogy and readable code;
- a small and stable core;
- explicit semantics rather than hidden behavior;
- reproducible experiments;
- robust telemetry for debugging, evaluation, and training;
- clean separation between the model, harness, task conditioning, and verifier.

Do not rewrite working code merely to match this document. First identify the smallest changes that materially improve the design.

---

## 1. Core contract

If the base model has the interface

\[
m : X \rightarrow Y,
\]

then the wrapped model should have the same interface:

\[
\operatorname{RLM}_{H}(m) : X \rightarrow Y.
\]

Conceptually:

```python
base_answer = model(x)
rlm_answer = RLM(model, harness=H)(x)
```

The caller supplies the same kind of input `x` and receives the same kind of output `y`.

The RLM's trace, metrics, and artifacts should be recorded **out of band** rather than changing the return type. A separate debugging API may return `(answer, trace)`, but the primary model-compatible interface should return only the answer.

---

## 2. One RLM invocation

A single invocation should behave as follows:

1. Create a fresh, ephemeral Python/IPython environment.
2. Place the exact input in a variable named `context`.
3. Prompt the model to generate one executable Python cell or finish the task.
4. Execute the cell in the persistent environment for this invocation.
5. Return a structured execution observation to the model.
6. Repeat until the model calls `finish(...)` or a host-enforced limit is reached.
7. Return the final answer.
8. Save the complete observable trace.
9. Destroy the environment.

The environment persists **within one invocation**, but not across independent calls unless a later experiment deliberately adds persistence.

The basic loop is:

\[
\text{model generates code}
\rightarrow
\text{execute code}
\rightarrow
\text{return observation}
\rightarrow
\text{repeat until finished}.
\]

---

## 3. Minimal separation of concerns

Use the following conceptual decomposition:

\[
z = A(x),
\qquad
(y, \tau) = \operatorname{RLM}_{M,H}(z),
\qquad
r = V(x,y).
\]

### `M`: model

The model chooses the next code cell or final response.

### `H`: harness

The harness defines how computation occurs:

- the system prompt;
- the Python environment;
- available functions;
- execution and observation formatting;
- recursion;
- step, depth, token, and time budgets;
- termination;
- trace collection.

### `A`: optional task conditioner

`A` transforms the raw task into useful task-specific conditioning:

- project instructions analogous to `CLAUDE.md` or `AGENTS.md`;
- selected examples;
- schemas;
- resources;
- skill descriptions;
- task-family-specific guidance.

`A` should remain outside the RLM kernel. The RLM simply receives its result as input.

### `V`: verifier

The verifier evaluates the final answer independently of the model and harness. It may be a unit-test suite, symbolic checker, exact answer, simulator reward, or human feedback.

The verifier, held-out data, and global budget enforcement must not be editable by an automated harness optimizer.

---

## 4. Stable model-facing environment

Treat the Python environment as a small, versioned ABI shared by teacher and student models.

### Required primitives

```python
context  # exact input for this invocation
finish(response)  # terminate and return response
```

### Recommended primitives

```python
model_call(request)  # one ordinary call to the base model
rlm_call(request)  # one fresh recursive RLM invocation
describe(name)  # signature, type, docstring, and safe metadata
```

### Optional skill interface

```python
list_skills()  # names and short descriptions only
load_skill(name)  # load the selected skill's full instructions
```

Avoid a large mandatory API. Add a primitive only when it supports a concrete experiment or removes recurring prompt complexity.

Recursive `rlm_call(...)` should create a fresh ephemeral child environment. The host must enforce total recursion depth and shared compute budgets.

---

## 5. Reference loop

The exact implementation may differ, but its semantics should remain this simple:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

X = TypeVar("X")
Y = TypeVar("Y")


@dataclass(frozen=True)
class Observation:
    stdout: str
    stderr: str
    result: str | None
    exception: str | None


@dataclass(frozen=True)
class HarnessConfig:
    max_steps: int = 16
    max_depth: int = 2
    execution_timeout_seconds: float = 10.0
    observation_char_limit: int = 16_000


class RLM(Generic[X, Y]):
    def __init__(
        self,
        model: Callable[[list[dict[str, Any]]], str],
        *,
        config: HarnessConfig,
        trace_sink: "TraceSink",
    ) -> None:
        self.model = model
        self.config = config
        self.trace_sink = trace_sink

    def __call__(self, x: X) -> Y:
        run_id = self.trace_sink.start_run(x=x, config=self.config)
        env = self._new_environment(context=x, run_id=run_id, depth=0)
        history: list[dict[str, Any]] = []

        try:
            for step in range(self.config.max_steps):
                messages = self._build_messages(
                    context=x,
                    history=history,
                    step=step,
                    env=env,
                )

                code = self.model(messages)
                self.trace_sink.record_model_output(
                    run_id=run_id,
                    step=step,
                    output=code,
                )

                observation = self._execute(code, env)
                self.trace_sink.record_execution(
                    run_id=run_id,
                    step=step,
                    code=code,
                    observation=observation,
                )

                history.extend(
                    [
                        {"role": "assistant", "content": code},
                        {
                            "role": "tool",
                            "content": self._format_observation(observation),
                        },
                    ]
                )

                if env.finished:
                    answer = env.final_response
                    self.trace_sink.finish_run(run_id=run_id, answer=answer)
                    return answer

            raise RuntimeError("RLM reached its step limit without finishing")
        except BaseException as exc:
            self.trace_sink.fail_run(run_id=run_id, exception=exc)
            raise
        finally:
            env.close()
```

This is a semantic skeleton, not a demand for these exact classes.

Important properties:

- the model remains the action policy;
- the same environment persists across inner steps;
- the model receives execution observations;
- `finish(...)` is explicit;
- host limits cannot be bypassed by the model;
- the environment always closes;
- telemetry is independent of the returned answer.

---

## 6. Input representation

Do not force every caller into one new schema merely for the RLM.

`context` should contain the exact input type accepted by the wrapped model:

```python
context = "Solve this problem..."
```

or:

```python
context = [
    {"role": "user", "content": "Hey, how are you?"},
    {"role": "assistant", "content": "I'm doing well."},
    {"role": "user", "content": "What did I just ask?"},
]
```

or a structured JSON-compatible object:

```python
context = {
    "problem": problem,
    "instructions": project_instructions,
    "resources": resource_descriptors,
}
```

Large files and non-JSON objects should usually be passed through explicit resource handles rather than embedded into one giant prompt.

---

## 7. Prompt design

The universal system prompt should explain only the RLM protocol:

- `context` contains the input;
- generate executable Python;
- inspect execution results before proceeding;
- state persists during this invocation;
- use `model_call(...)` or `rlm_call(...)` when useful;
- call `finish(response)` when done;
- do not invent tool results;
- respect budgets shown in observations.

Task-specific instructions belong in `context` or in an external task-conditioning step `A(x)`.

Avoid continuously expanding the universal prompt with domain-specific advice. That makes the harness harder to interpret and encourages accidental benchmark overfitting.

---

## 8. What is part of the trainable harness `H`?

A useful first search space is small and explicit.

### Reasonable trainable components

```python
HarnessGenome = {
    "system_prompt": str,
    "max_steps": int,
    "max_depth": int,
    "max_subcalls": int,
    "observation_char_limit": int,
    "history_window": int,
    "include_tracebacks": bool,
    "function_registry_version": str,
    "skill_catalog_version": str,
}
```

Later, an optimizer may edit selected hooks:

```python
def build_messages(...): ...
def format_observation(...): ...
def build_child_context(...): ...
def select_environment_functions(...): ...
```

### Keep immutable during harness search

- verifier implementation;
- answers and test labels;
- train/validation/test partitioning;
- sandbox and permission boundary;
- global compute accounting;
- telemetry schema;
- model client semantics;
- experiment identifiers and version recording.

Use a fixed global compute budget when comparing candidates. Otherwise, a harness may appear better merely because it increases recursion or token use.

---

## 9. Environment functions and skills

The initial environment is itself part of `H`.

Functions should be discoverable through a controlled registry rather than requiring the model to inspect arbitrary internal objects.

```python
@tool(
    description="Solve independent subproblems and combine their results.",
    tags=["decomposition", "aggregation"],
)
def map_reduce(items, mapper, reducer):
    """Apply mapper to each item and combine the results with reducer."""
    ...
```

`describe("map_reduce")` should expose safe metadata such as:

```json
{
  "name": "map_reduce",
  "signature": "(items, mapper, reducer)",
  "docstring": "Apply mapper to each item and combine the results with reducer.",
  "tags": ["decomposition", "aggregation"],
  "version": "1.2.0"
}
```

Variables generally do not have docstrings. Use a metadata registry for important variables such as `context`.

Skills should be loaded lazily:

```python
list_skills()
load_skill("decompose_and_compose")
```

Do not place the full text of every skill into every prompt. Skill selection and skill overload are themselves useful research questions.

Generated functions or skills must not become permanent automatically. Promotion should require:

- tests;
- versioning;
- provenance;
- held-out evidence of improvement;
- rollback or revocation support.

---

## 10. Telemetry and observable traces

Record observable computation, not hidden chain-of-thought.

A trace should be a tree of runs because recursive calls create child RLM invocations.

Recommended JSONL events:

```json
{"event":"run_start","run_id":"...","parent_run_id":null,"input_hash":"..."}
{"event":"model_request","run_id":"...","step":0,"messages":[...]}
{"event":"model_output","run_id":"...","step":0,"code":"..."}
{"event":"execution","run_id":"...","step":0,"stdout":"...","stderr":"...","result":"...","exception":null}
{"event":"subcall_start","run_id":"...","child_run_id":"...","depth":1}
{"event":"subcall_finish","run_id":"...","child_run_id":"...","result":"..."}
{"event":"finish","run_id":"...","answer":"..."}
{"event":"verifier","run_id":"...","score":1.0,"details":{}}
{"event":"run_end","run_id":"...","usage":{},"duration_seconds":0.0}
```

For reproducibility, record at least:

- model name and exact revision;
- sampling parameters and random seed where supported;
- harness version or source commit;
- system prompt hash;
- function and skill versions;
- Python and dependency versions;
- input hash and dataset example ID;
- step, depth, token, wall-clock, and subcall usage;
- exact generated code;
- execution observations and exceptions;
- verifier result;
- parent/child run relationships.

Scrub secrets and sensitive environment values before persistence.

---

## 11. Teacher-to-student adaptation

A central use case is:

1. Wrap a large teacher model `M` in the RLM.
2. Run `RLM(M)` on a large verified task set.
3. Store complete observable trajectories.
4. Filter or rank trajectories using an external verifier.
5. Adapt a smaller model `m` to operate the same RLM ABI.

The training examples should emphasize the next observable action:

```text
input/context
+ previous generated cells
+ execution observations
→ next Python cell or finish call
```

Useful training targets include:

- valid Python generation;
- appropriate inspection of `context`;
- reacting to errors;
- deciding when to recurse;
- constructing good child contexts;
- integrating sub-results;
- verifying answers;
- terminating efficiently.

Hidden prose reasoning is not required. Executed code, observations, subcalls, and final verified answers form a reproducible supervision signal.

The same trace format should support:

- supervised fine-tuning;
- preference training over better and worse trajectories;
- RL with verifiable rewards;
- behavior cloning from a larger teacher;
- harness debugging and ablation analysis.

---

## 12. Harness optimization

Keep model weights fixed and search over a constrained harness space:

\[
H^* = \arg\max_H J(H, \theta_0).
\]

A practical iteration is:

1. Run the current harness on a training batch.
2. Give an optimizer the current source, metrics, and selected success/failure traces.
3. Request one small patch and an explicit prediction.
4. Evaluate the patch under the same compute budget.
5. Accept it only if it improves validation performance or the accuracy-cost Pareto frontier.

Use distinct datasets:

```text
D_train       used to propose changes
D_validation  used to accept/reject changes
D_test        untouched until final evaluation
```

Start with three search spaces:

1. system prompt only;
2. prompt plus bounded hyperparameters;
3. prompt, hyperparameters, and a few selected code hooks.

Do not begin by allowing arbitrary rewrites of the entire runtime.

---

## 13. Basic 2×2 experiment

Compare harness adaptation and weight adaptation separately:

| | Original model `m₀` | Adapted model `m*` |
|---|---:|---:|
| Original harness `H₀` | baseline | weight adaptation |
| Optimized harness `H*` | harness adaptation | joint adaptation |

This tests:

- how much improvement comes from the harness alone;
- how much comes from model adaptation;
- whether the gains are additive;
- whether `H*` transfers to other models;
- whether `m*` remains effective under a different harness;
- whether the pair merely co-adapts to one benchmark.

Cross-test models and harnesses rather than evaluating only matched pairs.

---

## 14. Recommended implementation priorities

### P0: preserve the clean contract

- `RLM(model)(x)` accepts and returns the same types as `model(x)`.
- traces are out of band;
- each invocation gets a fresh environment;
- termination is explicit;
- limits are host enforced;
- environment cleanup is guaranteed.

### P1: make experiments reproducible

- version the prompt and function registry;
- emit structured JSONL traces;
- record model, sampling, environment, and budget metadata;
- represent recursive calls as a trace tree;
- add deterministic unit tests for loop semantics.

### P2: make the system trainable

- expose stable action/observation formatting;
- retain exact prompts and model outputs;
- export verified trajectories as training examples;
- keep teacher and student on the same environment ABI;
- make verifier results joinable to traces by `run_id`.

### P3: add controlled extensibility

- `describe(name)`;
- versioned function registry;
- lazy skills;
- recursive `rlm_call(...)`;
- bounded harness hyperparameters;
- selected optimizer-editable hooks.

Avoid adding persistence, autonomous background execution, plugin ecosystems, or elaborate memory systems until an experiment specifically requires them.

---

## 15. Suggested tests

At minimum, add tests for:

1. `context` is available unchanged inside the environment.
2. Variables persist across inner steps.
3. Independent RLM calls do not share state.
4. `finish(...)` returns the correct value and ends the loop.
5. The environment closes after success and failure.
6. Syntax errors and runtime errors become structured observations.
7. Output truncation is deterministic and visible to the model.
8. Step, time, token, and recursion limits are enforced by the host.
9. Recursive calls receive fresh environments and correct depth metadata.
10. Trace events are complete, ordered, and linked by run IDs.
11. The ordinary interface returns only the answer.
12. Debug mode can retrieve the corresponding trace.

---

## 16. Non-goals for the core

The core RLM does not need to become:

- a full replacement for Codex or Claude Code;
- a persistent personal agent;
- a plugin manager;
- a distributed task scheduler;
- a general memory architecture;
- an IDE;
- a hidden planning system;
- a framework with many mandatory abstractions.

Those systems provide valuable lessons, but the RLM's advantage is that it is a small experimental object whose behavior can be inspected, reproduced, trained, and modified.

---

## 17. Review instructions for the existing implementation

When reviewing the current repository:

1. Read the implementation and tests before proposing changes.
2. Map existing components onto `M`, `H`, optional `A`, `V`, environment, and telemetry.
3. Identify where the current behavior already satisfies this design.
4. List concrete divergences, ordered by impact.
5. Prefer small patches over architectural rewrites.
6. Preserve compatibility unless a change is clearly justified.
7. Add tests for every semantic change.
8. Keep the central loop readable in one place.
9. Avoid abstractions that obscure the action-observation loop.
10. Produce a brief implementation note explaining the final architecture and remaining limitations.

A successful revision should make it easy for a researcher to answer:

- What exact prompt did the model receive?
- What code did it generate?
- What happened when that code executed?
- What state persisted during the invocation?
- Why did the invocation stop?
- How much computation did it use?
- Which harness, model, skill, and environment versions were active?
- Can the trajectory be converted directly into training data?
- Can the experiment be repeated?

---

## Summary

The intended design is:

\[
\boxed{
\text{same model signature}
+
\text{ephemeral persistent-within-call Python state}
+
\text{model-action/execution-observation loop}
+
\text{optional recursion}
+
\text{small versioned harness}
+
\text{external task conditioning and verification}
+
\text{complete observable traces}
}
\]

Keep the kernel simple. Put task-specific complexity in explicit input data. Put mandatory guarantees in the host. Record enough telemetry to debug, evaluate, optimize, and train the entire system.
