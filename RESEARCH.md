# Research notes

This design was reviewed against public work available on 2026-08-10. The
central conclusion is deliberately conservative: make programmatic interaction
with external context reliable first; make recursion optional and measurable.

## Primary RLM sources

- [Recursive Language Models, v3](https://arxiv.org/html/2512.24601v3)
- [Official RLM implementation](https://github.com/alexzhang13/rlm)
- [Prime Intellect rlm-harness](https://github.com/PrimeIntellect-ai/rlm-harness)
- [Prime Agent](https://www.primeintellect.ai/blog/prime-agent)

The paper's durable idea is not a particular `FINAL(...)` string protocol. It
is keeping large context in a persistent environment while generated programs
can transform it and invoke models over values constructed at runtime. The
official implementation and paper expose several practical lessons:

- Root prompts are model-specific policy and must be versioned.
- Qwen variants can generate pathological fan-out without explicit limits.
- `FINAL`/`FINAL_VAR` was brittle enough to require substantial training-data
  repair; typed final submission is safer.
- Higher recursive depth is not monotonically useful and compounds syntax and
  planning failures.
- Captured observations belong in immutable logs even when the controller sees
  a more aggressively truncated view; capture itself needs a hard bound.
- A subprocess kernel with deadlines is operationally preferable to in-process
  `exec`, but neither one is a security sandbox by itself.

This repository therefore uses complete response objects, prompt hashes,
shared budgets, a host-owned model broker, fresh recursive kernels, and a
canonical event stream.

## Evidence against recursion by default

- [SRLM](https://arxiv.org/html/2603.15653) separates gains from programmatic
  context interaction from gains due specifically to recursive model calls.
- [Think, But Don't Overthink](https://arxiv.org/abs/2603.02615) reports that
  additional RLM depth can sharply increase latency and reduce accuracy.
- [TimeRLM](https://arxiv.org/html/2608.03391) shows strong structured-context
  interaction and RL results under hard turn and efficiency budgets, often
  without recursive subcalls.
- [lambda-RLM](https://arxiv.org/abs/2603.20105) motivates typed, bounded
  composition over unconstrained recursive fan-out.

Accordingly, `max_depth=0` is the default. Ordinary `model_complete` leaf calls
remain available; recursive `rlm_complete` is an explicit experiment.

## Action and context design

- [CodeAct](https://arxiv.org/abs/2402.01030) supports Python as a compositional
  action space.
- [Recursive Models](https://arxiv.org/html/2603.02112) and
  [Recursive Agent Harnesses](https://arxiv.org/html/2606.13643) reinforce
  isolated child frames and compact parent-visible returns.
- [LCM](https://arxiv.org/html/2605.04050) motivates immutable raw information
  and lossless pointers when summaries or compaction are introduced.

The default action is one official `ipython` function tool. A strict fenced-code
encoding exists because compatible endpoints do not all implement tool history
equally. Natural-language controller output is a final answer; Python is used
only when it helps. The public request never becomes a flattened user string.

## Prime ecosystem boundary

- [Continual Harness](https://arxiv.org/html/2605.09998)
- [Verifiers](https://github.com/PrimeIntellect-ai/verifiers)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)

Prime Agent demonstrates useful host/kernel separation and same-engine child
agents, but it also includes daemon sessions, schedules, mutable skills,
memories, and asynchronous messaging. Prime-RL is a distributed vLLM/FSDP
training system. Neither is the right dependency for this small model adapter.

The useful architectural boundary comes from Verifiers:

- taskset: data, tools, and scoring;
- harness: how the agent rolls out;
- runtime: where execution occurs.

`rlm` is the harness/engine. Tasks, graders, rewards, Continual Harness prompt
mutation, and distributed training stay outside it. Trace events include stable
run, branch, event, parent, depth, and model-call identifiers so downstream
systems can reconstruct trajectories without changing inference behavior.

## RL implications

The official RLM training work emphasizes root-policy trajectories, filtering
malformed or trivial episodes, protocol repair, and eventually on-policy RL.
TimeRLM separately suggests decomposed reward components and applying
efficiency pressure primarily to successful trajectories.

The recent [single-layer RL study](https://arxiv.org/html/2607.01232) is the
strongest direct evidence for the proposed middle-layer experiment. Across
seven Qwen3/Qwen2.5-derived models, three RL algorithms, and math, code, and
agentic tasks, its highest-contribution layers were usually around 40--60% of
network depth. A profiling-free middle-five-layer heuristic beat the paper's
full-parameter RL baseline on all three tested Qwen3 sizes. This is promising,
but not yet a universal rule for Qwen3.5 or RLM trajectories. We should make the
trainable layer set explicit and benchmark at least middle-five, a selected
single layer, and an appropriate PEFT/full-update control.

Freezing parameters does reduce trainable weights, optimizer state, and weight
gradient storage. It does not eliminate rollout generation or the full forward
path, and gradients for a middle block still backpropagate through later frozen
blocks. We should therefore measure peak memory and step time rather than infer
an end-to-end savings factor from the fraction of trainable parameters.

For any future trainer, ordinary API response objects are insufficient. It
must preserve exact sampled token IDs and log probabilities, the renderer that
produced the sampled context, generated-token masks, branch visibility, and
weight/version provenance. This is why no RL-specific normalization is hidden
inside the inference library.

## Codex subscriptions

Official OpenAI documentation distinguishes ChatGPT subscription access from
usage-billed API access:

- [Codex authentication](https://learn.chatgpt.com/docs/auth)
- [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk)
- [Codex app-server](https://learn.chatgpt.com/docs/app-server)
- [Non-interactive Codex](https://learn.chatgpt.com/docs/non-interactive-mode)

Those surfaces support programmatic local Codex agents. They do not turn a
subscription into `/v1/chat/completions`, expose hidden chain-of-thought, or
provide policy token/log-probability telemetry. The current
[Terms of Use](https://openai.com/policies/terms-of-use/) also restrict
programmatic output extraction and using output to develop competing models.
Any Codex integration must therefore be labeled as an agent adapter and kept
separate from the exact OpenAI-compatible transport.
